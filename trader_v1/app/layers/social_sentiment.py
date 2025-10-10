"""Social sentiment ingestion from Reddit."""
from __future__ import annotations

import asyncio
import datetime as dt
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from app.layers.base import BaseLayer

try:
    import praw
except ImportError:  # pragma: no cover - dependency optional in runtime
    praw = None


class SocialSentimentLayer(BaseLayer):
    """Collects Reddit discussions and scores ticker sentiment."""

    POSITIVE_WORDS = {"buy", "bull", "moon", "surge", "beat", "upgrade", "strong"}
    NEGATIVE_WORDS = {"sell", "bear", "dump", "downgrade", "miss", "weak"}

    def __init__(self, db, config: Dict[str, Any], error_manager) -> None:
        super().__init__("SocialSentiment", db, config, error_manager)
        self._tz = dt.timezone.utc
        reddit_cfg = config.get("social", {})
        self._subreddits = reddit_cfg.get("subreddits", ["stocks", "wallstreetbets"])
        self._limit = int(reddit_cfg.get("limit", 100))
        self._reddit = self._init_reddit()
        self._tickers: List[str] = (
            config.get("universe", {}).get("tickers")
            or config.get("tickers")
            or ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
        )

    def _init_reddit(self):
        if praw is None:
            self._info("ℹ️  PRAW not installed; social sentiment disabled.")
            return None
        client_id = os.getenv("REDDIT_CLIENT_ID")
        secret = os.getenv("REDDIT_CLIENT_SECRET")
        agent = os.getenv("REDDIT_USER_AGENT", "TraderModel/1.0")
        if not client_id or not secret:
            self._info("ℹ️  Reddit credentials missing; social sentiment disabled.")
            return None
        return praw.Reddit(
            client_id=client_id,
            client_secret=secret,
            user_agent=agent,
        )

    async def run(self) -> None:
        await self._update_status("🟡")
        if self._reddit is None:
            await self._update_status("⚪️")
            return

        self._info("💬 Social sentiment pulse started")
        try:
            await self._guard("Reddit", self._collect_and_store)
            await self._update_status("🟢")
        except Exception as exc:  # noqa: BLE001
            await self._update_status("🔴")
            self.logger.error("⚠️  Social sentiment failed: %s", exc, exc_info=False)

    async def _collect_and_store(self) -> None:
        submissions = await asyncio.to_thread(self._fetch_posts)
        if not submissions:
            self._info("ℹ️  No Reddit submissions collected.")
            return

        sentiment_rows = self._score_sentiment(submissions)
        if not sentiment_rows:
            self._info("ℹ️  No ticker mentions detected.")
            return

        await asyncio.to_thread(self.db.executemany, self._insert_sql(), sentiment_rows)
        self._info("💬 Stored %d social sentiment rows", len(sentiment_rows))

    def _fetch_posts(self) -> List[dict]:
        data: List[dict] = []
        pattern = re.compile(r"\b[A-Z]{2,5}\b")
        for sub in self._subreddits:
            subreddit = self._reddit.subreddit(sub)
            for submission in subreddit.new(limit=self._limit):
                text = f"{submission.title}\n{submission.selftext or ''}"
                tickers = [m for m in pattern.findall(text) if m in self._tickers]
                if not tickers:
                    continue
                data.append(
                    {
                        "subreddit": sub,
                        "created_utc": dt.datetime.utcfromtimestamp(submission.created_utc),
                        "title": submission.title,
                        "text": submission.selftext or "",
                        "tickers": tickers,
                    }
                )
        return data

    def _score_sentiment(self, posts: List[dict]) -> List[dict]:
        rows: List[dict] = []
        for post in posts:
            full_text = f"{post['title']} {post['text']}".lower()
            pos_hits = sum(word in full_text for word in self.POSITIVE_WORDS)
            neg_hits = sum(word in full_text for word in self.NEGATIVE_WORDS)
            score = 0.0
            if pos_hits or neg_hits:
                score = (pos_hits - neg_hits) / max(1, pos_hits + neg_hits)
            for ticker in post["tickers"]:
                rows.append(
                    {
                        "detected_at": dt.datetime.now(tz=self._tz),
                        "ticker": ticker,
                        "subreddit": post["subreddit"],
                        "score": score,
                        "mentions": len(post["tickers"]),
                        "captured_at": post["created_utc"],
                    }
                )
        return rows

    @staticmethod
    def _insert_sql() -> str:
        return """
            INSERT INTO social_sentiment (detected_at, ticker, subreddit, score, mentions, captured_at)
            VALUES (%(detected_at)s, %(ticker)s, %(subreddit)s, %(score)s, %(mentions)s, %(captured_at)s)
        """
