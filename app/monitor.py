import time
import asyncio
import logging
import os
from typing import Dict, Optional

from alpaca.trading.client import TradingClient

from app.config_loader import RISK, SCHED, NOTIFY, THRESHOLDS, ALP
from app.notify import discord_embed

logger = logging.getLogger("Monitor")


class SystemMonitor:
    def __init__(self, key, secret):
        key = (
            key
            or ALP.get("api_key")
            or os.getenv("ALPACA_API_KEY_ID")
            or os.getenv("ALPACA_API_KEY")
        )
        secret = (
            secret
            or ALP.get("secret_key")
            or os.getenv("ALPACA_API_SECRET_KEY")
            or os.getenv("ALPACA_SECRET_KEY")
        )
        if not key or not secret:
            raise RuntimeError("Missing Alpaca credentials for SystemMonitor")
        base_url = (
            ALP.get("rest_base_url")
            or os.getenv("ALPACA_API_BASE_URL")
            or "https://paper-api.alpaca.markets"
        )
        paper_mode = True if base_url and "paper" in base_url.lower() else bool(RISK.get("paper_trading", True))
        self.trading_client = TradingClient(key, secret, paper=paper_mode, url_override=base_url)
        self.last_equity = None
        self.last_heartbeat = 0
        self.connection_failures = 0
        self.halted = False
        self.halt_reason: Optional[str] = None
        self.halt_timestamp: Optional[float] = None
        self.latest_metrics: Dict[str, float] = {}
        self.auto_restart_seconds = int(THRESHOLDS.get("auto_restart_after_minutes", 0) * 60)

    async def heartbeat_loop(self):
        while True:
            try:
                await self.heartbeat()
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")
            await asyncio.sleep(SCHED["heartbeat_seconds"])

    async def heartbeat(self) -> None:
        if time.time() - self.last_heartbeat < THRESHOLDS.get("min_heartbeat_interval", 600):
            return
        await asyncio.to_thread(self._heartbeat_once)

    def _heartbeat_once(self) -> None:
        account = self.trading_client.get_account()
        eq, cash = float(account.equity), float(account.cash)
        pos_count = len(self.trading_client.get_all_positions())
        self._send_heartbeat(eq, cash, pos_count)
        self.last_equity, self.last_heartbeat = eq, time.time()
        self.check_thresholds()

    def _send_heartbeat(self, equity, cash, pos_count):
        lines = [
            f"Equity: ${equity:,.2f}",
            f"Cash: ${cash:,.2f}",
            f"Positions: {pos_count}",
            f"ConnFails: {self.connection_failures}",
        ]
        if self.latest_metrics:
            for key, value in self.latest_metrics.items():
                lines.append(f"{key}: {value}")
        description = "\n".join(lines)
        color = 0x00FF00 if not self.halted else 0xE74C3C
        discord_embed(
            "🫀 System Heartbeat",
            description=description,
            color=color,
            mention_role_id=NOTIFY.get("mention_role_id"),
            channel="market-updates",
        )

    def check_thresholds(self):
        if not THRESHOLDS.get("enabled"):
            return False
        account = self.trading_client.get_account()
        eq = float(account.equity)
        if self.last_equity:
            loss_pct = (self.last_equity - eq) / max(self.last_equity, 1)
            if loss_pct >= THRESHOLDS["daily_loss_limit"]:
                self.halt_bot(f"Daily loss {loss_pct:.2%} exceeded limit")
                return True
        if self.connection_failures >= THRESHOLDS["max_connection_failures"]:
            self.halt_bot(f"{self.connection_failures} connection failures exceeded limit")
            return True
        return False

    def record_connection_failure(self):
        self.connection_failures += 1
        self.check_thresholds()

    def halt_bot(self, reason):
        if self.halted:
            return
        self.halted = True
        self.halt_reason = reason
        self.halt_timestamp = time.time()
        discord_embed(
            "🚨 Threshold Triggered — Bot Halted",
            reason,
            status="error",
            mention_role_id=NOTIFY.get("mention_role_id"),
            channel="system-backend",
        )
        logger.error(f"Bot halted: {reason}")

    def update_metrics(self, **metrics: float) -> None:
        formatted = {self._format_metric_name(k): self._format_metric_value(v) for k, v in metrics.items() if v is not None}
        self.latest_metrics.update(formatted)

    @staticmethod
    def _format_metric_name(name: str) -> str:
        return name.replace("_", " ").title()

    @staticmethod
    def _format_metric_value(value) -> str:
        if isinstance(value, (int, float)):
            return f"{value:.3f}" if abs(value) < 100 else f"{value:,.2f}"
        return str(value)

    def try_auto_restart(self) -> bool:
        if not self.halted:
            return False
        if not THRESHOLDS.get("enabled") or self.auto_restart_seconds <= 0:
            return False
        if self.halt_timestamp is None:
            return False
        if time.time() - self.halt_timestamp < self.auto_restart_seconds:
            return False

        self.halted = False
        self.halt_reason = None
        self.halt_timestamp = None
        self.connection_failures = 0
        self.last_heartbeat = 0
        discord_embed(
            "♻️ Auto-Restart",
            "Threshold cooldown elapsed — resuming trading.",
            status="info",
            mention_role_id=NOTIFY.get("mention_role_id"),
            channel="system-backend",
        )
        logger.info("Bot auto-restart executed after cooldown.")
        return True
