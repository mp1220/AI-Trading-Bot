"""Market data collection layer."""
from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List

from app.data_ingest import DataIngest
from app.layers.base import BaseLayer


class MarketDataLayer(BaseLayer):
    """Fetches market data from multiple providers without blocking other layers."""

    def __init__(self, db, config: Dict[str, Any], error_manager) -> None:
        super().__init__("MarketData", db, config, error_manager)
        self._tickers: List[str] = (
            config.get("universe", {}).get("tickers")
            or config.get("tickers")
            or ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
        )
        self._ingest = DataIngest(db, config)

    async def run(self) -> None:
        await self._update_status("🟡")
        self._info("📡 MarketData pulse started")

        try:
            providers = [
                ("finnhub", self._ingest.pulse_finnhub_quotes),
                ("alpaca", self._ingest.update_from_alpaca),
                ("yfinance", self._ingest.pulse_yfinance_bars),
                ("polygon", getattr(self._ingest, "pulse_polygon_news", None)),
            ]
            pulse_tasks = []
            for name, func in providers:
                maybe = self._maybe(name, func)
                if maybe is not None:
                    pulse_tasks.append((name, asyncio.create_task(maybe)))
            if pulse_tasks:
                results = await asyncio.gather(
                    *(task for _, task in pulse_tasks), return_exceptions=True
                )
                for (provider, _), result in zip(pulse_tasks, results):
                    if isinstance(result, Exception):
                        if provider == "polygon":
                            self._info("ℹ️ Polygon pulse unavailable, continuing with other sources.")
                        else:
                            self._info("ℹ️  %s subtask error: %s", provider, result)
            await self._update_status("🟢")
            self._info("📡 MarketData pulse complete")
        except Exception as exc:  # noqa: BLE001
            await self._update_status("🔴")
            self.logger.error("⚠️  MarketData pulse failed: %s", exc, exc_info=False)

    def ensure_initial_history(self, interval: str = "15m") -> None:
        self._ingest.ensure_initial_history(interval=interval)

    def _maybe(self, provider: str, func):
        token_env = {
            "finnhub": "FINNHUB_API_KEY",
            "alpaca": "ALPACA_API_KEY_ID",
            "polygon": "POLYGON_API_KEY",
            "yfinance": None,
        }
        token_name = token_env.get(provider)
        if provider == "alpaca":
            key = (
                os.getenv("ALPACA_API_KEY_ID")
                or os.getenv("ALPACA_API_KEY")
            )
            secret = (
                os.getenv("ALPACA_API_SECRET_KEY")
                or os.getenv("ALPACA_SECRET_KEY")
            )
            if not key or not secret:
                async def fallback():
                    self._info("ℹ️  Alpaca credentials missing; running simulated pulse.")
                return fallback()
        elif token_name and not os.getenv(token_name):
            self._info("ℹ️  %s credentials missing; skipping.", provider)
            return None
        if func is None:
            self._info("ℹ️  %s pulse unavailable.", provider)
            return None
        provider_name = provider.capitalize()

        def runner():
            if provider == "yfinance":
                func(self._tickers, interval="1h")
            elif provider == "finnhub":
                func(self._tickers)
            else:
                func()

        async def wrapped():
            try:
                await self._guard(provider_name, asyncio.to_thread, runner)
            except Exception as exc:
                if provider == "polygon":
                    self._info("ℹ️ Polygon pulse unavailable, continuing with other sources.")
                else:
                    raise exc

        return wrapped()
