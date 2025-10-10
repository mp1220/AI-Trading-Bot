"""Threaded Alpaca live data streaming."""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Iterable, Optional

from app.config import normalize_alpaca_env

normalize_alpaca_env()

try:
    from alpaca.data.enums import DataFeed
    from alpaca.data.live import StockDataStream
except ImportError:  # pragma: no cover
    StockDataStream = None
    DataFeed = None

logger = logging.getLogger("Stream")


class StreamRunner:
    """Run the Alpaca streaming client in a background thread."""

    def __init__(self, tickers: Iterable[str]):
        if StockDataStream is None or DataFeed is None:
            raise RuntimeError("alpaca-py streaming client not installed.")
        api_key = (
            os.getenv("ALPACA_API_KEY_ID")
            or os.getenv("ALPACA_API_KEY")
        )
        secret_key = (
            os.getenv("ALPACA_API_SECRET_KEY")
            or os.getenv("ALPACA_SECRET_KEY")
        )
        if not api_key or not secret_key:
            raise RuntimeError("ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY must be set for streaming.")

        feed_name = (
            os.getenv("ALPACA_FEED")
            or os.getenv("ALPACA_DATA_FEED")
            or "iex"
        ).upper()
        feed_enum = DataFeed.IEX if feed_name != "SIP" else DataFeed.SIP

        self.tickers = list(tickers)
        self.stream = StockDataStream(api_key, secret_key, feed=feed_enum)
        self._thread: Optional[threading.Thread] = None
        self._running = False

    async def _on_trade(self, trade) -> None:
        from app.storage import store_live_quote

        store_live_quote(trade.symbol, float(trade.price), trade.timestamp)
        logger.info("%s: %s", trade.symbol, trade.price)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.stream.subscribe_trades(self._on_trade, *self.tickers)
        self._thread = threading.Thread(target=self._run_blocking, daemon=True)
        self._thread.start()
        logger.info("📡 Alpaca stream thread started.")

    def _run_blocking(self) -> None:
        try:
            self.stream.run()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Stream exited with: %s", exc)
        finally:
            logger.info("🛑 Alpaca stream thread ended.")

    def stop(self, join: bool = True) -> None:
        if not self._running:
            return
        self._running = False
        try:
            self.stream.stop()
        except Exception as exc:  # noqa: BLE001
            logger.debug("stream.stop() guard: %s", exc)
        if join and self._thread and self._thread.is_alive():
            for _ in range(10):
                if not self._thread.is_alive():
                    break
                time.sleep(0.1)
        logger.info("✅ Alpaca stream stopped gracefully.")


_runner: Optional[StreamRunner] = None


def start_stream(tickers: Iterable[str]) -> None:
    global _runner
    if StockDataStream is None or DataFeed is None:
        logger.info("Alpaca streaming client unavailable; skipping live stream.")
        return
    try:
        _runner = StreamRunner(tickers)
        _runner.start()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unable to start Alpaca stream: %s", exc)
        _runner = None


def stop_stream() -> None:
    global _runner
    if _runner is not None:
        _runner.stop()
        _runner = None


class LiveStream:
    """Async wrapper around the streaming runner for orchestrator integration."""

    def __init__(self, api_key: Optional[str], secret_key: Optional[str], tickers: Iterable[str] | None = None):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
        self.tickers = [t for t in (tickers or []) if t]
        self._runner: Optional[StreamRunner] = None

    async def start(self) -> None:
        if StockDataStream is None or DataFeed is None:
            raise RuntimeError("alpaca-py streaming client not installed.")
        if not self.api_key or not self.secret_key:
            raise RuntimeError("ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY must be provided.")
        if not self.tickers:
            raise RuntimeError("No tickers supplied for live stream.")

        os.environ["ALPACA_API_KEY_ID"] = self.api_key
        os.environ["ALPACA_API_SECRET_KEY"] = self.secret_key

        if self._runner and self._runner._running:
            return

        loop = asyncio.get_running_loop()
        self._runner = StreamRunner(self.tickers)
        await loop.run_in_executor(None, self._runner.start)

    async def stop(self) -> None:
        if not self._runner:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._runner.stop)
        self._runner = None

    def is_running(self) -> bool:
        return bool(self._runner and self._runner._running)
