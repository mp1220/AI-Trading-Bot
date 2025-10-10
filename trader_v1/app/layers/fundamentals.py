"""Fundamentals ingestion layer."""
from __future__ import annotations

import asyncio
import datetime as dt
import os
from typing import Any, Dict, List

from app.config import data_source_enabled
from app.layers.base import BaseLayer
from app.utils import cooldown_enforce, request_json


class FundamentalsLayer(BaseLayer):
    """Fetches company profiles from FinancialModelingPrep."""

    FMP_URL = "https://financialmodelingprep.com/api/v3/profile/{ticker}"

    def __init__(self, db, config: Dict[str, Any], error_manager) -> None:
        super().__init__("Fundamentals", db, config, error_manager)
        self._tickers: List[str] = (
            config.get("runtime", {}).get("benchmark_tickers")
            or config.get("universe", {}).get("tickers")
            or config.get("tickers")
            or ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
        )
        self._enabled = data_source_enabled(config, "fmp")
        self._api_key = os.getenv("FMP_API_KEY")
        self._tz = dt.timezone.utc

    async def run(self) -> None:
        if not self._enabled or not self._api_key:
            await self._update_status("⚪️")
            return

        await self._update_status("🟡")
        rows: List[Dict[str, Any]] = []
        try:
            for ticker in self._tickers:
                cooldown_enforce(f"fmp:{ticker}", 300)
                row = await self._guard(
                    "FMP",
                    asyncio.to_thread,
                    self._fetch_profile,
                    ticker,
                )
                if row:
                    rows.append(row)
            if rows:
                await asyncio.to_thread(self._insert_rows, rows)
            await self._update_status("🟢")
        except Exception as exc:  # noqa: BLE001
            await self._update_status("🔴")
            self.logger.error("⚠️  Fundamentals pulse failed: %s", exc, exc_info=False)

    def _fetch_profile(self, ticker: str) -> Dict[str, Any] | None:
        params = {"apikey": self._api_key}
        data = request_json(
            "FMP",
            self.FMP_URL.format(ticker=ticker),
            params=params,
            backoff_schedule=[0, 2, 4, 8, 16],
        )
        if not data:
            return None
        record = data[0] if isinstance(data, list) else data
        if not record:
            return None
        pe = record.get("pe")
        eps = record.get("eps")
        self._info(
            "📊 [FMP] Retrieved fundamentals for %s (PE=%s, EPS=%s)",
            ticker,
            f"{pe:.2f}" if isinstance(pe, (int, float)) else "n/a",
            f"{eps:.2f}" if isinstance(eps, (int, float)) else "n/a",
        )
        return {
            "ticker": ticker,
            "company_name": record.get("companyName"),
            "market_cap": record.get("mktCap"),
            "price": record.get("price"),
            "pe": pe,
            "eps": eps,
            "beta": record.get("beta"),
            "fetched_at": dt.datetime.now(tz=self._tz).replace(tzinfo=None),
        }

    def _insert_rows(self, rows: List[Dict[str, Any]]) -> None:
        sql = """
            INSERT INTO fundamentals (ticker, company_name, market_cap, price, pe, eps, beta, fetched_at)
            VALUES (%(ticker)s, %(company_name)s, %(market_cap)s, %(price)s, %(pe)s, %(eps)s, %(beta)s, %(fetched_at)s)
            ON CONFLICT (ticker) DO UPDATE SET
                company_name=EXCLUDED.company_name,
                market_cap=EXCLUDED.market_cap,
                price=EXCLUDED.price,
                pe=EXCLUDED.pe,
                eps=EXCLUDED.eps,
                beta=EXCLUDED.beta,
                fetched_at=EXCLUDED.fetched_at
        """
        self.db.executemany(sql, rows)
