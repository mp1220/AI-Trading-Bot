"""Alpaca trading execution layer."""
from __future__ import annotations

import logging
import os
from typing import Any

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest
except ImportError:  # pragma: no cover
    TradingClient = None
    MarketOrderRequest = None
    OrderSide = None
    TimeInForce = None

from app.config import get_alpaca_settings

logger = logging.getLogger("Trading")


class TradeExecutor:
    """Submit market orders to Alpaca (paper or live)."""

    def __init__(self) -> None:
        self.settings = get_alpaca_settings()
        self.mode = self.settings["mode"]
        self.enabled = bool(self.settings["trade_enabled"])
        api_key = (
            os.getenv("ALPACA_API_KEY_ID")
            or os.getenv("ALPACA_API_KEY")
        )
        secret_key = (
            os.getenv("ALPACA_API_SECRET_KEY")
            or os.getenv("ALPACA_SECRET_KEY")
        )
        if TradingClient is None or not api_key or not secret_key:
            logger.info("🚫 TradingClient unavailable or credentials missing; trading disabled.")
            self.client = None
            self.enabled = False
            return
        self.client = TradingClient(
            api_key,
            secret_key,
            paper=self.mode == "paper",
        )
        logger.info("📈 TradingClient initialized (%s mode)", self.mode)

    def place_order(self, symbol: str, side: str, qty: float) -> bool:
        if not self.enabled:
            logger.info("🚫 Trade skipped (trading disabled): %s %s %s", side, qty, symbol)
            return False
        if self.client is None or MarketOrderRequest is None:
            logger.info("🚫 Trade skipped (client unavailable).")
            return False

        try:
            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            resp = self.client.submit_order(order)
            status = getattr(resp, "status", "submitted")
            logger.info("✅ %s order executed: %s %s @ %s", side.upper(), qty, symbol, status)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("⚠️  Order failed: %s", exc)
            return False
