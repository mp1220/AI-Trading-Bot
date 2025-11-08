"""Market regime detection layer."""
from __future__ import annotations

import asyncio
import datetime as dt
import numpy as np
import pandas as pd
from typing import Any, Dict

from app.layers.base import BaseLayer


class RegimeDetectorLayer(BaseLayer):
    """Classifies the market as bull/bear/sideways using volatility and momentum."""

    def __init__(self, db, config: Dict[str, Any], error_manager) -> None:
        super().__init__("RegimeDetector", db, config, error_manager)
        self._tz = dt.timezone.utc
        self._benchmark = config.get("regime", {}).get("benchmark", "SPY")
        self._lookback_days = int(config.get("regime", {}).get("lookback_days", 20))
        self._vol_threshold = float(config.get("regime", {}).get("vol_threshold", 0.02))
        self._momentum_threshold = float(config.get("regime", {}).get("momentum_threshold", 0.01))

    async def run(self) -> None:
        await self._update_status("🟡")
        try:
            await self._guard("RegimeDetector", self._classify_regime)
            await self._update_status("🟢")
        except Exception as exc:  # noqa: BLE001
            await self._update_status("🔴")
            self.logger.error("⚠️  Regime detection failed: %s", exc, exc_info=False)

    async def _classify_regime(self) -> None:
        since = dt.datetime.now(tz=self._tz) - dt.timedelta(days=self._lookback_days + 5)
        sql = """
            SELECT ts, close FROM daily_bars
            WHERE ticker=%(ticker)s AND ts >= %(since)s
            ORDER BY ts
        """
        df = await asyncio.to_thread(self.db.to_df, sql, {"ticker": self._benchmark, "since": since})
        if df.empty:
            self.logger.warning("⚠️  No benchmark data for regime detection.")
            return

        df["return"] = df["close"].pct_change()
        recent = df.tail(self._lookback_days).copy()
        momentum = (recent["close"].iloc[-1] / recent["close"].iloc[0]) - 1
        volatility = float(np.sqrt(252) * recent["return"].std())

        if momentum > self._momentum_threshold and volatility < self._vol_threshold:
            regime = "Bull"
        elif momentum < -self._momentum_threshold and volatility > self._vol_threshold:
            regime = "Bear"
        else:
            regime = "Sideways"

        payload = {
            "classified_at": dt.datetime.now(tz=self._tz),
            "benchmark": self._benchmark,
            "regime": regime,
            "momentum": float(momentum),
            "volatility": float(volatility),
        }
        await asyncio.to_thread(self.db.executemany, self._insert_sql(), [payload])
        self._info(
            "🧭 Market regime: %s (momentum=%.3f volatility=%.3f)",
            regime,
            momentum,
            volatility,
        )

    @staticmethod
    def _insert_sql() -> str:
        return """
            INSERT INTO market_regime (classified_at, benchmark, regime, momentum, volatility)
            VALUES (%(classified_at)s, %(benchmark)s, %(regime)s, %(momentum)s, %(volatility)s)
        """
