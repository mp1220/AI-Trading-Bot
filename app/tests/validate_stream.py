"""Quick check that Alpaca streaming is operational."""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import suppress

from alpaca.data.enums import DataFeed
from alpaca.data.live import StockDataStream

from app.config import normalize_alpaca_env


logger = logging.getLogger("StreamTest")


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    normalize_alpaca_env()

    key = (
        os.getenv("ALPACA_API_KEY_ID")
        or os.getenv("ALPACA_API_KEY")
    )
    secret = (
        os.getenv("ALPACA_API_SECRET_KEY")
        or os.getenv("ALPACA_SECRET_KEY")
    )

    if not key or not secret:
        logger.error("❌ Missing Alpaca credentials (ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY)")
        raise SystemExit(1)

    base_url = "wss://stream.data.alpaca.markets/v2/iex"
    stream = StockDataStream(key, secret, feed=DataFeed.IEX)

    received = asyncio.Event()

    async def _on_trade(data):
        logger.info("✅ Received trade update: %s @ %s", getattr(data, "symbol", "?"), getattr(data, "price", "?"))
        received.set()
        with suppress(Exception):
            await stream.stop()

    stream.subscribe_trades(_on_trade, "AAPL")

    try:
        logger.info("🔌 Connecting to %s", base_url)
        runner = asyncio.create_task(stream.run(), name="alpaca_stream_run")
        await asyncio.wait_for(received.wait(), timeout=15)
    except asyncio.TimeoutError:
        logger.error("❌ Stream connected but no data received within 15s")
        with suppress(Exception):
            await stream.stop()
        raise SystemExit(1)
    except Exception as exc:  # noqa: BLE001
        logger.error("❌ WebSocket failure: %s", exc)
        with suppress(Exception):
            await stream.stop()
        raise SystemExit(1)
    else:
        logger.info("✅ WebSocket connected and streaming OK")
        raise SystemExit(0)
    finally:
        current = asyncio.current_task()
        for task in asyncio.all_tasks():
            if task is current:
                continue
            if task.get_name() == "alpaca_stream_run" and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stream test interrupted by user")
