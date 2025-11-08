"""Catalyst detection layer combining macro and news signals."""
from __future__ import annotations

import asyncio
import datetime as dt
import json
from typing import Any, Dict, List

import pandas as pd

from app.layers.base import BaseLayer


class CatalystAILayer(BaseLayer):
    """Derives sector catalysts from news, macro data, and historical context."""

    THEMES: Dict[str, Dict[str, Any]] = {
        "ai": {"keywords": ["artificial intelligence", "ai", "machine learning"], "sector": "Technology"},
        "oil": {"keywords": ["oil", "crude", "opec", "energy"], "sector": "Energy"},
        "semiconductors": {"keywords": ["semiconductor", "chip", "wafer", "fabs"], "sector": "Technology"},
        "rates": {"keywords": ["fed", "rate hike", "rate cut", "treasury", "yields"], "sector": "Financials"},
    }

    def __init__(self, db, config: Dict[str, Any], error_manager) -> None:
        super().__init__("CatalystAI", db, config, error_manager)
        self._tz = dt.timezone.utc
        self._lookback_hours = int(config.get("catalysts", {}).get("lookback_hours", 12))

    async def run(self) -> None:
        await self._update_status("🟡")
        self._info("🧠 Catalyst AI pulse started")
        try:
            await self._guard("CatalystAI", self._detect_catalysts)
            await self._update_status("🟢")
        except Exception as exc:  # noqa: BLE001
            await self._update_status("🔴")
            self.logger.error("⚠️  Catalyst AI pulse failed: %s", exc, exc_info=False)

    async def _detect_catalysts(self) -> None:
        since = dt.datetime.now(tz=self._tz) - dt.timedelta(hours=self._lookback_hours)
        sql = """
            SELECT ticker, headline, sentiment, published_at
            FROM news_headlines
            WHERE published_at >= %(since)s
        """
        news = await asyncio.to_thread(self.db.to_df, sql, {"since": since})
        if news.empty:
            self._info("ℹ️  No recent headlines for catalyst detection.")
            return

        catalysts: list[dict[str, Any]] = []
        for theme, meta in self.THEMES.items():
            mask = news["headline"].str.contains(
                "|".join(meta["keywords"]),
                case=False,
                na=False,
            )
            subset = news[mask]
            if subset.empty:
                continue
            avg_sent = float(subset["sentiment"].fillna(0.0).mean())
            magnitude = float(min(1.0, max(0.0, subset.shape[0] / 10)))
            tickers = sorted(set(subset["ticker"].dropna().tolist()))
            catalysts.append(
                {
                    "detected_at": dt.datetime.now(tz=self._tz),
                    "theme": theme,
                    "sector": meta["sector"],
                    "sentiment": avg_sent,
                    "magnitude": magnitude,
                    "tickers": json.dumps(tickers),
                }
            )

        if not catalysts:
            self._info("ℹ️  Catalyst AI found no active themes.")
            return

        await asyncio.to_thread(self.db.executemany, self._insert_sql(), catalysts)
        for row in catalysts:
            self._info(
                "🚀 Catalyst detected: %s (%s) sent=%.2f mag=%.2f tickers=%s",
                row["theme"],
                row["sector"],
                row["sentiment"],
                row["magnitude"],
                row["tickers"],
            )

    @staticmethod
    def _insert_sql() -> str:
        return """
            INSERT INTO catalysts (detected_at, theme, sector, sentiment, magnitude, tickers)
            VALUES (%(detected_at)s, %(theme)s, %(sector)s, %(sentiment)s, %(magnitude)s, CAST(%(tickers)s AS JSONB))
            ON CONFLICT (theme, detected_at) DO UPDATE
            SET sentiment=EXCLUDED.sentiment,
                magnitude=EXCLUDED.magnitude,
                tickers=EXCLUDED.tickers
        """
