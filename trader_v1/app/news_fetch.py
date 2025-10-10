from __future__ import annotations

import datetime as dt
import hashlib
import os
import re
from typing import Dict, List, Tuple

from app.config import data_source_enabled
from app.utils import cooldown_enforce, request_json
from app.utils.logging import get_logger


class NewsFetch:
    """Fetches headlines from GDELT with fallbacks."""

    GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
    NEWSAPI_URL = "https://newsapi.org/v2/everything"
    FINNHUB_URL = "https://finnhub.io/api/v1/company-news"

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.logger = get_logger("News")
        self._quiet = bool(cfg.get("runtime", {}).get("quiet_mode", False))
        self._gdelt_enabled = data_source_enabled(cfg, "gdelt")
        self._newsapi_enabled = data_source_enabled(cfg, "newsapi")
        self._finnhub_enabled = data_source_enabled(cfg, "finnhub")
        self._newsapi_key = os.getenv("NEWS_API_KEY")
        self._finnhub_key = os.getenv("FINNHUB_API_KEY")

    def fetch_latest(self, db, tickers: List[str], lookback_hours: int = 72) -> int:
        total_inserted = 0
        total_skipped = 0
        for ticker in tickers:
            inserted, skipped = self._fetch_for_ticker(db, ticker, lookback_hours)
            total_inserted += inserted
            total_skipped += skipped
        if total_inserted or total_skipped:
            self.logger.info(
                "💾 Stored %d new headlines, skipped %d duplicates",
                total_inserted,
                total_skipped,
            )
        return total_inserted

    def _fetch_for_ticker(self, db, ticker: str, lookback_hours: int) -> Tuple[int, int]:
        providers = [
            ("GDELT", self._gdelt_enabled, self._fetch_gdelt),
            ("NewsAPI", self._newsapi_enabled and bool(self._newsapi_key), self._fetch_newsapi),
            ("Finnhub", self._finnhub_enabled and bool(self._finnhub_key), self._fetch_finnhub),
        ]

        for name, enabled, fetcher in providers:
            if not enabled:
                continue
            articles = fetcher(ticker, lookback_hours)
            if not articles:
                continue
            to_insert = self._prepare_rows(db, ticker, articles)
            inserted, skipped = db.insert_news_rows(to_insert)
            self.logger.info(
                "📰 [%s] Fetching headlines for %s... %d new (skipped %d)",
                name,
                ticker,
                inserted,
                skipped,
            )
            if inserted:
                return inserted, skipped
        return 0, 0

    def _prepare_rows(self, db, ticker: str, articles: List[Dict]) -> List[Dict]:
        seen = set()
        rows: List[Dict] = []
        for article in articles:
            headline = article.get("headline") or article.get("title")
            if not headline:
                continue
            url = article.get("url") or ""
            key = hashlib.sha1(f"{headline}{url}".encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            published = self._parse_datetime(
                article.get("published_at")
                or article.get("seendate")
                or article.get("datetime")
                or article.get("publishedAt")
            )
            rows.append(
                {
                    "id": key,
                    "ticker": ticker,
                    "headline": headline,
                    "source": article.get("source") or article.get("sourceName"),
                    "url": article.get("url"),
                    "published_at": published,
                    "sentiment": article.get("sentiment"),
                }
            )
        return rows

    def _fetch_gdelt(self, ticker: str, lookback_hours: int) -> List[Dict]:
        cooldown_enforce("gdelt", 5)
        params = {
            "query": ticker,
            "maxrecords": 250,
            "timespan": f"{lookback_hours}h",
            "format": "json",
        }
        data = request_json(
            "GDELT",
            self.GDELT_URL,
            params=params,
            backoff_schedule=[0, 2, 4, 8, 16],
        )
        articles = data.get("articles", []) if isinstance(data, dict) else []
        formatted = []
        for article in articles:
            formatted.append(
                {
                    "headline": article.get("title"),
                    "source": article.get("sourcecommonname"),
                    "url": article.get("url"),
                    "published_at": article.get("seendate"),
                    "id": article.get("id"),
                }
            )
        return formatted

    def _fetch_newsapi(self, ticker: str, lookback_hours: int) -> List[Dict]:
        if not self._newsapi_key:
            return []
        since = dt.datetime.utcnow() - dt.timedelta(hours=lookback_hours)
        params = {
            "q": ticker,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 100,
            "from": since.isoformat(timespec="seconds") + "Z",
            "apiKey": self._newsapi_key,
        }
        data = request_json("NewsAPI", self.NEWSAPI_URL, params=params, backoff_schedule=[0, 2, 4])
        if not data:
            return []
        articles = data.get("articles", [])
        formatted = []
        for article in articles:
            formatted.append(
                {
                    "headline": article.get("title"),
                    "source": (article.get("source") or {}).get("name"),
                    "url": article.get("url"),
                    "published_at": article.get("publishedAt"),
                }
            )
        return formatted

    def _fetch_finnhub(self, ticker: str, lookback_hours: int) -> List[Dict]:
        if not self._finnhub_key:
            return []
        end = dt.datetime.utcnow()
        start = end - dt.timedelta(hours=lookback_hours)
        params = {
            "symbol": ticker,
            "from": start.strftime("%Y-%m-%d"),
            "to": end.strftime("%Y-%m-%d"),
            "token": self._finnhub_key,
        }
        data = request_json("Finnhub", self.FINNHUB_URL, params=params, backoff_schedule=[0, 2, 4])
        if isinstance(data, list):
            articles = data
        else:
            articles = data.get("news", []) if isinstance(data, dict) else []
        formatted = []
        for article in articles:
            formatted.append(
                {
                    "headline": article.get("headline"),
                    "source": article.get("source"),
                    "url": article.get("url"),
                    "published_at": article.get("datetime"),
                }
            )
        return formatted

    @staticmethod
    def _parse_datetime(value) -> dt.datetime:
        if isinstance(value, dt.datetime):
            return value.replace(tzinfo=None)
        if isinstance(value, (int, float)):
            return dt.datetime.utcfromtimestamp(value)
        if isinstance(value, str):
            for fmt in ("%Y%m%d%H%M%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
                try:
                    return dt.datetime.strptime(value.replace("Z", ""), fmt)
                except ValueError:
                    continue
            try:
                return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                pass
        return dt.datetime.utcnow()
