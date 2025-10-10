"""Trading signal generation layer."""
from __future__ import annotations

import asyncio
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None
import pickle
import numpy as np
import pandas as pd

from app.features import ema, realized_vol, rsi
from app.layers.base import BaseLayer
from app.trader_model import TraderModel


class TradingSignalsLayer(BaseLayer):
    """Applies the latest models to produce actionable trading signals."""

    def __init__(
        self,
        db,
        config: Dict[str, Any],
        error_manager,
        models_dir: Path,
        timezone: dt.tzinfo,
        model_prefix: str,
    ) -> None:
        super().__init__("TradingSignals", db, config, error_manager)
        self._tz = timezone
        self._models_dir = models_dir
        self._prefix = model_prefix
        self._fallback_model = TraderModel(config)
        self._tickers: List[str] = (
            config.get("universe", {}).get("tickers")
            or config.get("tickers")
            or ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
        )

    async def run(self) -> None:
        await self._update_status("🟡")
        try:
            await self._guard("SignalEngine", self._generate_signals)
            await self._update_status("🟢")
        except Exception as exc:  # noqa: BLE001
            await self._update_status("🔴")
            self.logger.error("⚠️  Signal generation failed: %s", exc, exc_info=False)

    async def _generate_signals(self) -> None:
        bars = await self._load_recent_bars()
        if bars is None:
            return

        model = self._load_latest_model()
        sentiments = await self._aggregate_sentiment()
        features = self._prepare_features(bars, sentiments)
        if features.empty:
            self._info("ℹ️  No features available for signal generation.")
            return

        accuracy_factor = await self._accuracy_factor()
        rows = []

        for _, row in features.iterrows():
            ts = row["ts"]
            ticker = row["ticker"]

            if model is not None:
                proba = float(model.predict_proba(row[["ema_fast", "ema_slow", "rsi", "vol", "volume"]].values.reshape(1, -1))[0, 1])
                details = {
                    "ema_fast": float(row["ema_fast"]),
                    "ema_slow": float(row["ema_slow"]),
                    "rsi": float(row["rsi"]),
                    "vol": float(row["vol"]),
                    "volume": float(row["volume"]),
                    "sentiment": float(row["sentiment"]),
                }
            else:
                fallback = self._fallback_model.predict_many(
                    bars[bars["ticker"] == ticker].tail(200), {ticker: row["sentiment"]}
                )[0]
                proba = float(fallback["prob_up"])
                details = fallback["details"]

            confidence = float(proba * accuracy_factor)
            action = "BUY" if proba > 0.55 else "SELL" if proba < 0.45 else "HOLD"
            rows.append(
                {
                    "ticker": ticker,
                    "ts": ts,
                    "prob_up": proba,
                    "sentiment": float(row["sentiment"]),
                    "action": action,
                    "details": json.dumps({**details, "confidence": confidence}),
                }
            )
            self._info(
                "🎯 %s signal: prob=%.2f sentiment=%.2f confidence=%.2f → %s",
                ticker,
                proba,
                row["sentiment"],
                confidence,
                action,
            )

        await asyncio.to_thread(self.db.executemany, self._insert_sql(), rows)

    async def _load_recent_bars(self) -> Optional[pd.DataFrame]:
        cutoff = dt.datetime.now(tz=self._tz) - dt.timedelta(hours=6)
        sql = """
            SELECT timestamp AS ts, symbol AS ticker, open, high, low, close, volume
            FROM market_bars
            WHERE timestamp >= %(cutoff)s
            ORDER BY symbol, timestamp
        """
        df = await asyncio.to_thread(self.db.to_df, sql, {"cutoff": cutoff})
        if df.empty:
            self._info("ℹ️  Missing market data for signal generation.")
            return None
        return df[df["ticker"].isin(self._tickers)].copy()

    def _load_latest_model(self):
        pattern_online = f"{self._prefix}-online_*.pkl"
        pattern_daily = f"{self._prefix}-daily_*.pkl"
        candidates = list((self._models_dir / "online").glob(pattern_online))
        if not candidates:
            candidates = list((self._models_dir / "daily").glob(pattern_daily))
        if not candidates:
            return None
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        try:
            if joblib:
                model = joblib.load(latest)
            else:  # pragma: no cover
                with latest.open("rb") as fh:
                    model = pickle.load(fh)
            self._info("📥 Loaded model %s", latest.name)
            return model
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("⚠️  Failed to load model %s: %s", latest, exc)
            return None

    async def _aggregate_sentiment(self) -> Dict[str, float]:
        sql = """
            SELECT ticker, sentiment
            FROM (
                SELECT ticker, sentiment,
                       row_number() OVER (PARTITION BY ticker ORDER BY published_at DESC) AS rn
                FROM news_headlines
                WHERE sentiment IS NOT NULL
            ) x
            WHERE rn <= 3
        """
        news = await asyncio.to_thread(self.db.to_df, sql)
        news_sent = news.groupby("ticker")["sentiment"].mean().to_dict() if not news.empty else {}

        social_sql = """
            SELECT ticker, score
            FROM (
                SELECT ticker, score,
                       row_number() OVER (PARTITION BY ticker ORDER BY captured_at DESC) AS rn
                FROM social_sentiment
            ) s
            WHERE rn <= 5
        """
        social = await asyncio.to_thread(self.db.to_df, social_sql)
        social_sent = social.groupby("ticker")["score"].mean().to_dict() if not social.empty else {}

        sentiments = {}
        for ticker in self._tickers:
            sentiments[ticker] = float(
                (news_sent.get(ticker, 0.0) + social_sent.get(ticker, 0.0)) / 2
            )
        return sentiments

    def _prepare_features(self, bars: pd.DataFrame, sentiments: Dict[str, float]) -> pd.DataFrame:
        rows = []
        for ticker, group in bars.groupby("ticker"):
            group = group.sort_values("ts").copy()
            group["ema_fast"] = ema(group["close"], 12)
            group["ema_slow"] = ema(group["close"], 26)
            group["rsi"] = rsi(group["close"], 14)
            group["vol"] = realized_vol(group["close"], 20)
            group["sentiment"] = sentiments.get(ticker, 0.0)
            latest = group.tail(1)
            rows.append(latest)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    async def _accuracy_factor(self) -> float:
        sql = """
            SELECT accuracy
            FROM performance_metrics
            WHERE mode='daily'
            ORDER BY recorded_at DESC LIMIT 1
        """
        df = await asyncio.to_thread(self.db.to_df, sql)
        if df.empty or not np.isfinite(df.iloc[0]["accuracy"]):
            return 0.6
        acc = float(df.iloc[0]["accuracy"])
        return max(0.3, min(1.0, acc))

    @staticmethod
    def _insert_sql() -> str:
        return """
            INSERT INTO signals (ticker, ts, prob_up, sentiment, action, details)
            VALUES (%(ticker)s, %(ts)s, %(prob_up)s, %(sentiment)s, %(action)s, CAST(%(details)s AS JSONB))
            ON CONFLICT (ticker, ts) DO UPDATE
            SET prob_up=EXCLUDED.prob_up,
                sentiment=EXCLUDED.sentiment,
                action=EXCLUDED.action,
                details=EXCLUDED.details
        """
