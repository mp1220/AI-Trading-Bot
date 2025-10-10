"""Autonomous trade execution using the latest trained model."""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, List, Dict

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional joblib fallback
    import joblib
except ImportError:  # pragma: no cover
    joblib = None
import pickle

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from app.features import build_features
from app.config_loader import FOCUS_TICKERS, RISK, ALP, NOTIFY
from app.notify import discord_embed
from app.storage import DB


@dataclass
class TradeSignal:
    ticker: str
    price: float
    probability: float


class TradeExecutor:
    def __init__(self, db: DB, models_dir: Path, *, focus: Optional[Iterable[str]] = None) -> None:
        self.db = db
        self.models_dir = models_dir
        self.focus = [t for t in (focus or FOCUS_TICKERS) if t]
        self.logger = logging.getLogger("TradeExecutor")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        base_url = (
            os.getenv("ALPACA_API_BASE_URL")
            or ALP.get("rest_base_url")
            or "https://paper-api.alpaca.markets"
        )
        key = (
            ALP.get("api_key")
            or os.getenv("ALPACA_API_KEY_ID")
            or os.getenv("ALPACA_API_KEY")
        )
        secret = (
            ALP.get("secret_key")
            or os.getenv("ALPACA_API_SECRET_KEY")
            or os.getenv("ALPACA_SECRET_KEY")
        )
        paper_trading = bool(RISK.get("paper_trading", True))
        if not key or not secret:
            self.logger.warning("Alpaca credentials missing — trade executor disabled")
            self.trading_client = None
            self.enabled = False
        else:
            self.trading_client = TradingClient(key, secret, paper=paper_trading, url_override=base_url)
            self.enabled = True
        self.paper = paper_trading
        self.order_notional = 1000.0

    async def run(self) -> List[Dict[str, float]]:
        return await asyncio.to_thread(self._run_sync)

    # ------------------------------------------------------------------
    def _run_sync(self) -> List[Dict[str, float]]:
        if not self.enabled:
            self.logger.debug("Trade executor disabled; skipping run")
            return []

        model_path = self._latest_model()
        if model_path is None:
            self.logger.debug("No model available for trading")
            return []

        model = self._load_model(model_path)
        if model is None:
            return []

        frame = self._load_latest_frame()
        if frame.empty:
            self.logger.debug("No recent bars available for trading")
            return []

        signals = self._generate_signals(model, frame)
        if not signals:
            self.logger.debug("No actionable signals generated")
            return []

        executed: List[Dict[str, float]] = []
        for signal in signals:
            if self.db.traded_within(signal.ticker, 60):
                continue
            try:
                qty = max(1, int(round(self.order_notional / max(signal.price, 1.0))))
                request = MarketOrderRequest(
                    symbol=signal.ticker,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
                order = self.trading_client.submit_order(request)
                filled_price = getattr(order, "filled_avg_price", None)
                executed_price = float(filled_price) if filled_price else signal.price
                self.db.record_trade(
                    signal.ticker,
                    "buy",
                    qty,
                    executed_price,
                    rationale=f"Signal execution prob={signal.probability:.2f} model={model_path.name}",
                    ts=datetime.now(timezone.utc),
                )
                discord_embed(
                    "💸 Trade Executed",
                    f"BUY {signal.ticker} (qty={qty}, prob={signal.probability:.2f})",
                    status="warn",
                    mention_role_id=NOTIFY.get("mention_role_id"),
                )
                if self.paper:
                    self.logger.info(
                        "[Trade] BUY %s @ %.2f — paper trade executed",
                        signal.ticker,
                        executed_price,
                    )
                else:
                    self.logger.info(
                        "Trade submitted for %s (qty=%s, prob=%.2f)",
                        signal.ticker,
                        qty,
                        signal.probability,
                    )
                executed.append({
                    "ticker": signal.ticker,
                    "qty": float(qty),
                    "price": executed_price,
                    "probability": signal.probability,
                })
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Trade execution failed for %s: %s", signal.ticker, exc)
        return executed

    def _latest_model(self) -> Optional[Path]:
        candidates = list(self.models_dir.glob("model_*.pkl"))
        return max(candidates, default=None) if candidates else None

    def _load_model(self, path: Path):
        try:
            if joblib is not None:
                return joblib.load(path)
            with path.open("rb") as fh:  # pragma: no cover
                return pickle.load(fh)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to load trading model %s: %s", path.name, exc)
            return None

    def _load_latest_frame(self) -> pd.DataFrame:
        sql = """
            SELECT symbol AS ticker, timestamp AS ts, close, volume
            FROM market_bars
            WHERE symbol = ANY(%(tickers)s)
            ORDER BY ts DESC
            LIMIT 500
        """
        try:
            frame = self.db.to_df(sql, {"tickers": self.focus})
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to load latest market frame: %s", exc)
            return pd.DataFrame()
        return frame

    def _generate_signals(self, model, frame: pd.DataFrame) -> list[TradeSignal]:
        frame = build_features(frame)
        columns_needed = {"ret_1", "hl_range", "volume_z"}
        frame.dropna(subset=list(columns_needed), inplace=True)
        latest = frame.groupby("ticker").tail(1)
        if latest.empty:
            return []

        features = latest[["ret_1", "hl_range", "volume_z"]].replace([np.inf, -np.inf], 0.0)
        try:
            probabilities = (
                model.predict_proba(features.values)[:, 1]
                if hasattr(model, "predict_proba")
                else model.predict(features.values)
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Model prediction failed: %s", exc)
            return []

        signals: list[TradeSignal] = []
        for (_, row), prob in zip(latest.iterrows(), probabilities):
            if prob < 0.6:  # conservative threshold for paper trading
                continue
            signals.append(TradeSignal(ticker=row["ticker"], price=float(row["close"]), probability=float(prob)))
        return signals
