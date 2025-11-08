"""Technical indicator layer using stored market data."""
from __future__ import annotations

import asyncio
import datetime as dt
from typing import Any, Dict, List

import pandas as pd

from app.features import ema, realized_vol, rsi
from app.layers.base import BaseLayer


class TechnicalsLayer(BaseLayer):
    """Computes rolling technical factors from stored price data."""

    def __init__(self, db, config: Dict[str, Any], error_manager) -> None:
        super().__init__("Technicals", db, config, error_manager)
        self._tz = dt.timezone.utc
        self._window_hours = int(config.get("technicals", {}).get("window_hours", 6))
        self._tickers: List[str] = (
            config.get("universe", {}).get("tickers")
            or config.get("tickers")
            or ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
        )

    async def run(self) -> None:
        await self._update_status("🟡")
        self._info("📐 Technicals pulse started")
        try:
            await self._guard("Technicals", self._compute_factors)
            await self._update_status("🟢")
        except Exception as exc:  # noqa: BLE001
            await self._update_status("🔴")
            self.logger.error("⚠️  Technicals pulse failed: %s", exc, exc_info=False)

    async def _compute_factors(self) -> None:
        cutoff = dt.datetime.now(tz=self._tz) - dt.timedelta(hours=self._window_hours)
        sql = """
            SELECT timestamp AS ts, symbol AS ticker, open, high, low, close, volume
            FROM market_bars
            WHERE timestamp >= %(cutoff)s
            ORDER BY timestamp
        """
        df = await asyncio.to_thread(
            self.db.to_df,
            sql,
            {"cutoff": cutoff},
        )
        if df.empty:
            self._info("ℹ️  No recent bars for technical computation.")
            return

        frames: list[pd.DataFrame] = []
        for ticker, group in df[df["ticker"].isin(self._tickers)].groupby("ticker"):
            group = group.sort_values("ts").copy()
            group["ema_fast"] = ema(group["close"], 12)
            group["ema_slow"] = ema(group["close"], 26)
            group["rsi14"] = rsi(group["close"], 14)
            group["rsi2"] = rsi(group["close"], 2)
            group["volatility"] = realized_vol(group["close"], 20)
            group["momentum"] = group["close"].pct_change(periods=5)
            frames.append(group)

        features = pd.concat(frames, ignore_index=True)
        features["computed_at"] = dt.datetime.now(tz=self._tz)
        latest = features.groupby("ticker").tail(1).reset_index(drop=True)

        if not self.validate_frame(latest, ["ts", "ticker", "close", "ema_fast", "rsi14"]):
            return

        await asyncio.to_thread(
            self._store_dataframe,
            "technical_factors",
            latest[
                [
                    "ts",
                    "ticker",
                    "close",
                    "ema_fast",
                    "ema_slow",
                    "rsi14",
                    "rsi2",
                    "volatility",
                    "momentum",
                    "computed_at",
                ]
            ],
        )
