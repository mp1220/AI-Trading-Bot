"""News and sentiment processing layer."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from app.config import data_source_enabled
from app.layers.base import BaseLayer
from app.news_fetch import NewsFetch
from app.sentiment import Sentiment


class NewsSentimentLayer(BaseLayer):
    """Fetches headlines and scores sentiment asynchronously."""

    def __init__(self, db, config: Dict[str, Any], error_manager) -> None:
        super().__init__("NewsSentiment", db, config, error_manager)
        self._news = NewsFetch(config)
        self._sentiment = Sentiment(config)
        self._tickers: List[str] = (
            config.get("universe", {}).get("tickers")
            or config.get("tickers")
            or ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
        )
        self._lookback_days = int(config.get("news", {}).get("lookback_days", 3))
        self._news_enabled = any(
            data_source_enabled(config, source)
            for source in ("gdelt", "newsapi", "finnhub")
        )

    async def run(self) -> None:
        if not self._news_enabled:
            await self._update_status("⚪️")
            return

        await self._update_status("🟡")
        try:
            added = await self._guard(
                "NewsFetch",
                asyncio.to_thread,
                self._news.fetch_latest,
                self.db,
                self._tickers,
                self._lookback_days * 24,
            )
            rescored = await self._guard(
                "Sentiment",
                asyncio.to_thread,
                self._sentiment.update_unscored,
                self.db,
            )
            self._info("📰 News layer stored %s articles, rescored %s", added, rescored)
            await self._update_status("🟢")
        except Exception as exc:  # noqa: BLE001
            await self._update_status("🔴")
            self.logger.error("⚠️  NewsSentiment pulse failed: %s", exc, exc_info=False)
