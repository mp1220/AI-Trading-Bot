"""Autonomous decision engine that bridges scheduled signals with real-time data."""
from __future__ import annotations

import asyncio
import json
import os
import random
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence, Tuple

import pandas as pd

from app.reasoning import Reasoner
from app.storage import DB
from app.trading import TradeExecutor
from app.utils.discord_utils import notify_discord
from app.utils.logging import get_logger


def _as_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "on"}


def _ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _parse_details(details: Any) -> Dict[str, Any]:
    if isinstance(details, dict):
        return details
    if isinstance(details, str):
        try:
            return json.loads(details)
        except json.JSONDecodeError:
            return {}
    return {}


class DecisionEngine:
    """Continuously evaluates trading signals against live data."""

    def __init__(
        self,
        db: DB,
        cfg: Dict[str, Any],
        executor: TradeExecutor,
        focus: Sequence[str],
        discord_threshold: str,
    ) -> None:
        self.db = db
        self.cfg = cfg
        self.executor = executor

        self.logger = get_logger("DecisionEngine")
        self.reasoner = Reasoner(cfg)

        decision_cfg = cfg.get("decision_engine") or {}
        runtime_cfg = cfg.get("runtime") or {}

        env_flag = os.getenv("AI_AUTONOMOUS_TRADING")
        self.enabled = _as_bool(runtime_cfg.get("autonomous_trading", True)) and _as_bool(env_flag, True)

        conf_default = decision_cfg.get("confidence_threshold", 0.45)
        self.confidence_threshold = self._coerce_float(os.getenv("DECISION_CONFIDENCE_THRESHOLD"), conf_default)

        cooldown_default = decision_cfg.get("cooldown_seconds", 60)
        self.cooldown_seconds = max(0, self._coerce_int(os.getenv("DECISION_COOLDOWN_SECONDS"), cooldown_default))

        max_pos_default = decision_cfg.get("max_position_per_symbol", 1)
        self.max_position_per_symbol = max(0, self._coerce_int(os.getenv("DECISION_MAX_POSITION_PER_SYMBOL"), max_pos_default))

        dynamic_cfg = decision_cfg.get("dynamic_sleep") or {}
        self.sleep_high = max(1, self._coerce_int(os.getenv("DECISION_SLEEP_HIGH"), dynamic_cfg.get("high_volatility", 30)))
        self.sleep_low_min = max(1, self._coerce_int(os.getenv("DECISION_SLEEP_LOW_MIN"), dynamic_cfg.get("normal_min", 60)))
        self.sleep_low_max = max(
            self.sleep_low_min,
            self._coerce_int(os.getenv("DECISION_SLEEP_LOW_MAX"), dynamic_cfg.get("normal_max", 120)),
        )

        vol_cfg = decision_cfg.get("volatility") or {}
        self.vol_lookback = max(3, self._coerce_int(os.getenv("DECISION_VOL_LOOKBACK"), vol_cfg.get("lookback_points", 8)))
        self.vol_threshold = self._coerce_float(os.getenv("DECISION_VOL_THRESHOLD"), vol_cfg.get("high_threshold", 0.015))

        order_notional_default = decision_cfg.get("order_notional", 1000.0)
        self.order_notional = max(0.0, self._coerce_float(os.getenv("DECISION_ORDER_NOTIONAL"), order_notional_default))

        explore_default = decision_cfg.get("exploration_probability", 0.10)
        self.exploration_probability = min(1.0, max(0.0, self._coerce_float(os.getenv("DECISION_EXPLORATION_PROB"), explore_default)))

        max_states_default = decision_cfg.get("max_active_states", 20)
        env_max_states = os.getenv("DECISION_MAX_ACTIVE_STATES") or os.getenv("MAX_EVALUATION_STATES")
        self.max_active_states = max(1, self._coerce_int(env_max_states, max_states_default))

        self.discord_threshold = discord_threshold or "info"

        self.focus = [symbol.upper() for symbol in focus if symbol]

        # Track last action acknowledged for each symbol to avoid duplicate notifications.
        self._state: Dict[str, Tuple[Optional[datetime], str]] = {}
        self._state_order: deque[str] = deque()
        self._evaluation_log: deque[Dict[str, Any]] = deque(maxlen=self.max_active_states)

    async def run(self, stop_event: asyncio.Event) -> None:
        if not self.enabled:
            self.logger.info("🤖 Autonomous trading disabled; decision engine idle.")
            return
        if not self.focus:
            self.logger.warning("No focus tickers configured; decision engine exiting.")
            return

        self.logger.info("🤖 Decision engine monitoring: %s", ", ".join(self.focus))

        while not stop_event.is_set():
            quotes_df = await asyncio.to_thread(self.db.recent_live_quotes, self.focus, self.vol_lookback)
            sentiments = await asyncio.to_thread(self.db.aggregate_sentiment, self.focus)

            for symbol in self.focus:
                await self._evaluate_symbol(symbol, sentiments, quotes_df)

            sleep_seconds = self._dynamic_sleep(quotes_df)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=sleep_seconds)
            except asyncio.TimeoutError:
                continue

        self.logger.info("Decision engine shutdown complete.")

    async def _evaluate_symbol(
        self,
        symbol: str,
        sentiments: Dict[str, float],
        quotes_df: pd.DataFrame,
    ) -> None:
        exploratory = False
        confidence: Optional[float] = None

        signal = await asyncio.to_thread(self.db.latest_signal, symbol)
        if not signal:
            self._track_outcome(symbol, None, "no-signal", None, None, signal_id=None, confidence=None, exploratory=False)
            return

        signal_ts = _ensure_utc(signal["ts"])
        signal_id = signal.get("id")
        action = str(signal.get("action") or "").upper()
        prob_up = signal.get("prob_up")
        prob_value = float(prob_up) if prob_up is not None else None
        details = _parse_details(signal.get("details"))
        signal["details"] = details

        quote_info = self._latest_quote(quotes_df, symbol)
        if not quote_info:
            self._track_outcome(
                symbol,
                signal_ts,
                "skip:no-price",
                f"⚠️ Skipped {symbol} — no recent quote.",
                "warn",
                signal_id=signal_id,
                confidence=None,
                exploratory=False,
            )
            return
        price_ts, price = quote_info

        if action not in {"BUY", "SELL"}:
            self._track_outcome(symbol, signal_ts, "hold", None, None, signal_id=signal_id, confidence=None, exploratory=False)
            return

        confidence = self._confidence_for_action(action, prob_value, details)
        below_threshold = (confidence is None) or (confidence < self.confidence_threshold)

        if below_threshold:
            if confidence is not None and self._should_explore(confidence):
                exploratory = True
                self.logger.debug(
                    "Exploration triggered for %s (conf=%.2f < %.2f)",
                    symbol,
                    confidence,
                    self.confidence_threshold,
                )
            else:
                conf_display = "n/a" if confidence is None else f"{confidence:.2f}"
                self._track_outcome(
                    symbol,
                    signal_ts,
                    "skip:confidence",
                    f"⚠️ Skipped {symbol} — confidence {conf_display} < {self.confidence_threshold:.2f}",
                    "warn",
                    signal_id=signal_id,
                    confidence=confidence,
                    exploratory=False,
                )
                return

        sentiment = float(sentiments.get(symbol, 0.0))
        symbol_vol = self._symbol_volatility(quotes_df, symbol)
        reason = self.reasoner.combine(prob_value or 0.5, sentiment, symbol_vol)

        if reason.decision == "HOLD":
            self._track_outcome(
                symbol,
                signal_ts,
                "skip:neutral",
                f"ℹ️ Skipped {symbol} — reasoner suggests HOLD ({reason.reason}).",
                "info",
                signal_id=signal_id,
                confidence=confidence,
                exploratory=exploratory,
            )
            return

        if reason.decision != action:
            self._track_outcome(
                symbol,
                signal_ts,
                "skip:disagreement",
                f"⚠️ Skipped {symbol} — signal={action}, reasoner={reason.decision}.",
                "warn",
                signal_id=signal_id,
                confidence=confidence,
                exploratory=exploratory,
            )
            return

        if not self._sentiment_confirmation(action, sentiment):
            self._track_outcome(
                symbol,
                signal_ts,
                "skip:sentiment",
                f"⚠️ Skipped {symbol} — sentiment {sentiment:.2f} not aligned with {action}.",
                "warn",
                signal_id=signal_id,
                confidence=confidence,
                exploratory=exploratory,
            )
            return

        in_cooldown = await asyncio.to_thread(self.db.traded_within, symbol, self.cooldown_seconds)
        if in_cooldown:
            self._track_outcome(
                symbol,
                signal_ts,
                "skip:cooldown",
                f"⚠️ Skipped {symbol} — cooldown active ({self.cooldown_seconds}s).",
                "warn",
                signal_id=signal_id,
                confidence=confidence,
                exploratory=exploratory,
            )
            return

        open_position = await asyncio.to_thread(self.db.get_open_position, symbol)
        if action == "BUY":
            if self.max_position_per_symbol <= 0:
                self._track_outcome(
                    symbol,
                    signal_ts,
                    "skip:positions-disabled",
                    f"⚠️ Skipped {symbol} — position limit disabled by config.",
                    "warn",
                    signal_id=signal_id,
                    confidence=confidence,
                    exploratory=exploratory,
                )
                return
            if open_position:
                self._track_outcome(
                    symbol,
                    signal_ts,
                    "skip:position-open",
                    f"⚠️ Skipped {symbol} — position already open.",
                    "warn",
                    signal_id=signal_id,
                    confidence=confidence,
                    exploratory=exploratory,
                )
                return
        elif action == "SELL":
            if not open_position:
                self._track_outcome(
                    symbol,
                    signal_ts,
                    "skip:no-position",
                    f"ℹ️ Skipped {symbol} — nothing to close.",
                    "info",
                    signal_id=signal_id,
                    confidence=confidence,
                    exploratory=exploratory,
                )
                return

        rationale = (
            f"conf={confidence:.2f} prob_up={(prob_value or 0.0):.2f} "
            f"sent={sentiment:.2f} vol={symbol_vol:.3f} | {reason.reason}"
        )
        if exploratory:
            rationale += " | exploration=trial"

        await self._execute_trade(
            symbol=symbol,
            action=action,
            price=price,
            price_ts=_ensure_utc(price_ts),
            signal_ts=signal_ts,
            signal_id=signal_id,
            confidence=confidence if confidence is not None else 0.0,
            rationale=rationale,
            open_position=open_position,
            exploratory=exploratory,
        )

        conf_value = confidence if confidence is not None else 0.0
        outcome = f"executed:{action.lower()}"
        message = f"💸 AI Trade {action} {symbol} @ {price:.2f} (conf={conf_value:.2f})\n{rationale}"
        self._track_outcome(
            symbol,
            signal_ts,
            outcome,
            message,
            "warn",
            signal_id=signal_id,
            confidence=confidence,
            exploratory=exploratory,
        )
            conf_display = "n/a" if confidence is None else f"{confidence:.2f}"
            self._track_outcome(
                symbol,
                signal_ts,
                "skip:confidence",
                f"⚠️ Skipped {symbol} — confidence {conf_display} < {self.confidence_threshold:.2f}",
                "warn",
            )
            return

        sentiment = float(sentiments.get(symbol, 0.0))
        symbol_vol = self._symbol_volatility(quotes_df, symbol)
        reason = self.reasoner.combine(prob_value or 0.5, sentiment, symbol_vol)

        if reason.decision == "HOLD":
            self._track_outcome(
                symbol,
                signal_ts,
                "skip:neutral",
                f"ℹ️ Skipped {symbol} — reasoner suggests HOLD ({reason.reason}).",
                "info",
            )
            return

        if reason.decision != action:
            self._track_outcome(
                symbol,
                signal_ts,
                "skip:disagreement",
                f"⚠️ Skipped {symbol} — signal={action}, reasoner={reason.decision}.",
                "warn",
            )
            return

        if not self._sentiment_confirmation(action, sentiment):
            self._track_outcome(
                symbol,
                signal_ts,
                "skip:sentiment",
                f"⚠️ Skipped {symbol} — sentiment {sentiment:.2f} not aligned with {action}.",
                "warn",
            )
            return

        in_cooldown = await asyncio.to_thread(self.db.traded_within, symbol, self.cooldown_seconds)
        if in_cooldown:
            self._track_outcome(
                symbol,
                signal_ts,
                "skip:cooldown",
                f"⚠️ Skipped {symbol} — cooldown active ({self.cooldown_seconds}s).",
                "warn",
            )
            return

        open_position = await asyncio.to_thread(self.db.get_open_position, symbol)
        if action == "BUY":
            if self.max_position_per_symbol <= 0:
                self._track_outcome(
                    symbol,
                    signal_ts,
                    "skip:positions-disabled",
                    f"⚠️ Skipped {symbol} — position limit disabled by config.",
                    "warn",
                )
                return
            if open_position:
                self._track_outcome(
                    symbol,
                    signal_ts,
                    "skip:position-open",
                    f"⚠️ Skipped {symbol} — position already open.",
                    "warn",
                )
                return
        elif action == "SELL":
            if not open_position:
                self._track_outcome(
                    symbol,
                    signal_ts,
                    "skip:no-position",
                    f"ℹ️ Skipped {symbol} — nothing to close.",
                    "info",
                )
                return

    async def _execute_trade(
        self,
        symbol: str,
        action: str,
        price: float,
        price_ts: datetime,
        signal_ts: datetime,
        signal_id: Optional[int],
        confidence: float,
        rationale: str,
        open_position: Optional[Dict[str, Any]],
        exploratory: bool,
    ) -> None:
        side = action.lower()
        qty = self._order_quantity(price, open_position)

        success = await asyncio.to_thread(self.executor.place_order, symbol, side, qty)
        if not success:
            self.logger.warning("Trade submission failed for %s %s", action, symbol)
            self._track_outcome(
                symbol,
                signal_ts,
                "skip:order-failed",
                f"⚠️ {symbol} {action} order failed; see logs.",
                "warn",
                signal_id=signal_id,
                confidence=confidence,
                exploratory=exploratory,
            )
            return

        if action == "BUY":
            position_id = await asyncio.to_thread(
                self.db.open_position,
                symbol,
                action,
                qty,
                price,
                price_ts,
            )
        else:
            position_id = open_position["id"] if open_position else None
            if position_id:
                await asyncio.to_thread(self.db.close_position, position_id, price, price_ts)

        await asyncio.to_thread(
            self.db.record_trade,
            symbol,
            action,
            qty,
            price,
            signal_id=signal_id,
            position_id=position_id,
            pnl=0.0,
            rationale=rationale,
            ts=price_ts,
        )

    def _sentiment_confirmation(self, action: str, sentiment: float) -> bool:
        if action == "BUY":
            return sentiment >= -0.05
        return sentiment <= 0.05

    def _order_quantity(self, price: float, open_position: Optional[Dict[str, Any]]) -> float:
        if open_position and open_position.get("qty"):
            return float(open_position["qty"])
        if price <= 0:
            return 1.0
        qty = max(1, int(round(self.order_notional / price)))
        return float(qty)

    def _confidence_for_action(
        self,
        action: str,
        prob_up: Optional[float],
        details: Dict[str, Any],
    ) -> Optional[float]:
        detail_conf = details.get("confidence")
        if isinstance(detail_conf, (int, float)):
            return float(detail_conf)
        if prob_up is None:
            return None
        if action == "BUY":
            return float(prob_up)
        return 1.0 - float(prob_up)

    def _latest_quote(self, quotes_df: pd.DataFrame, symbol: str) -> Optional[Tuple[datetime, float]]:
        if quotes_df is None or quotes_df.empty:
            return None
        subset = quotes_df[quotes_df["symbol"] == symbol]
        if subset.empty:
            return None
        ordered = subset.sort_values("timestamp")
        latest = ordered.iloc[-1]
        ts = latest["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return ts, float(latest["price"])

    def _symbol_volatility(self, quotes_df: pd.DataFrame, symbol: str) -> float:
        if quotes_df is None or quotes_df.empty:
            return 0.0
        subset = quotes_df[quotes_df["symbol"] == symbol].sort_values("timestamp")
        if subset.shape[0] < 2:
            return 0.0
        returns = subset["price"].astype(float).pct_change().dropna()
        if returns.empty:
            return 0.0
        return float(returns.abs().mean())

    def _dynamic_sleep(self, quotes_df: pd.DataFrame) -> int:
        if self._is_high_vol(quotes_df):
            return self.sleep_high
        if self.sleep_low_min >= self.sleep_low_max:
            return self.sleep_low_min
        return random.randint(self.sleep_low_min, self.sleep_low_max)

    def _is_high_vol(self, quotes_df: pd.DataFrame) -> bool:
        if quotes_df is None or quotes_df.empty:
            return False
        for symbol in quotes_df["symbol"].unique():
            vol = self._symbol_volatility(quotes_df, symbol)
            if vol >= self.vol_threshold:
                return True
        return False

    def _should_explore(self, confidence: float) -> bool:
        if not isinstance(confidence, (int, float)):
            return False
        if self.exploration_probability <= 0.0:
            return False
        if confidence >= self.confidence_threshold:
            return False
        return random.random() < self.exploration_probability

    def _rebalance_state(self, symbol: str) -> None:
        if symbol in self._state_order:
            try:
                self._state_order.remove(symbol)
            except ValueError:
                pass
        self._state_order.append(symbol)
        while len(self._state_order) > self.max_active_states:
            removed = self._state_order.popleft()
            if removed != symbol:
                self._state.pop(removed, None)

    def _record_evaluation(
        self,
        symbol: str,
        signal_id: Optional[int],
        confidence: Optional[float],
        outcome: str,
        signal_ts: Optional[datetime],
        exploratory: bool,
    ) -> None:
        ts = signal_ts or datetime.now(timezone.utc)
        entry = {
            "timestamp": ts.isoformat(),
            "symbol": symbol,
            "id": signal_id,
            "confidence": None if confidence is None else float(confidence),
            "result": outcome,
            "exploratory": bool(exploratory),
        }
        self._evaluation_log.append(entry)

    def recent_evaluations(self) -> list[Dict[str, Any]]:
        return list(self._evaluation_log)

    def _track_outcome(
        self,
        symbol: str,
        signal_ts: Optional[datetime],
        outcome: str,
        message: Optional[str],
        level: Optional[str],
        *,
        signal_id: Optional[int] = None,
        confidence: Optional[float] = None,
        exploratory: bool = False,
    ) -> None:
        prev_ts, prev_outcome = self._state.get(symbol, (None, ""))
        duplicate = prev_ts == signal_ts and prev_outcome == outcome
        self._state[symbol] = (signal_ts, outcome)
        self._rebalance_state(symbol)
        self._record_evaluation(symbol, signal_id, confidence, outcome, signal_ts, exploratory)

        if duplicate:
            return

        if message and level:
            event = "trade_execution" if outcome.startswith("executed") else "trade_diagnostic"
            notify_discord(event, message, level=level, minimum_level=self.discord_threshold)
            if outcome.startswith("executed"):
                self.logger.info(message)
            elif level in {"warn", "warning", "error", "critical"}:
                self.logger.warning(message)
            else:
                self.logger.debug(message)

    @staticmethod
    def _coerce_float(value: Any, fallback: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(fallback)

    @staticmethod
    def _coerce_int(value: Any, fallback: int) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return int(fallback)
