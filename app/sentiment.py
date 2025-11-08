# app/sentiment.py
from __future__ import annotations

import logging
import os
import re

from openai import OpenAI

logger = logging.getLogger("Layer.Sentiment")


def classify_sentiment(headline_text: str) -> str:
    """Classify sentiment for a given headline while ensuring the client closes."""
    try:
        with OpenAI() as client:
            reply = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Classify the following headline sentiment as Positive, Negative, or Neutral.",
                    },
                    {
                        "role": "user",
                        "content": headline_text,
                    },
                ],
            )
            sentiment_text = reply.choices[0].message.content.strip()
            return sentiment_text
    except Exception as exc:  # noqa: BLE001
        logger.warning("⚠️ Sentiment classification failed: %s", exc)
        return "Neutral"


class Sentiment:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = cfg.get("sentiment_model", "gpt-4o-mini")
        self._quiet = bool(cfg.get("runtime", {}).get("quiet_mode", False))
        self._logger = logging.getLogger("Sentiment")
        key = os.getenv("OPENAI_API_KEY")
        if not key and not self._quiet:
            self._logger.warning("⚠️  Sentiment scoring disabled — OPENAI_API_KEY not set.")
        elif key and not self._quiet:
            self._logger.info("🧠 Sentiment initialized using OpenAI (%s)", self.model)
        self._enabled = bool(key)

    def score_texts(self, ticker: str, headlines: list[str]) -> float:
        if not headlines or not self._enabled:
            return 0.0

        joined = "\n".join(headlines)
        text = classify_sentiment(joined)
        lower = text.lower()
        if "positive" in lower:
            val = 0.7
        elif "negative" in lower:
            val = -0.7
        elif "neutral" in lower:
            val = 0.0
        else:
            m = re.search(r"[+-]?\d*\.?\d+", text)
            val = float(m.group()) if m else 0.0
        return max(-1.0, min(1.0, val))

    def update_unscored(self, db) -> int:
        rows = db.to_df("SELECT id, ticker, headline FROM news_headlines WHERE sentiment IS NULL")
        if rows.empty:
            return 0
        updated = 0
        for _, r in rows.iterrows():
            s = self.score_texts(r["ticker"], [r["headline"]])
            db.execute(
                "UPDATE news_headlines SET sentiment=%(s)s WHERE id=%(i)s",
                {"s": s, "i": int(r["id"])}
            )
            updated += 1
        return updated
