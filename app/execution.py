# app/execution.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional


@dataclass
class RiskConfig:
    mode: str = "paper"
    max_positions: int = 10
    per_trade_usd: float = 5000.0
    position_size_pct: float = 0.05
    max_gross_exposure_pct: float = 0.5
    fee_bps: float = 5.0
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    daily_max_drawdown_pct: float = 0.05
    borrow_cost_pct: float = 0.0

    @classmethod
    def from_config(cls, cfg: dict) -> "RiskConfig":
        risk_cfg = cfg.get("risk") or {}
        defaults = cls()
        return cls(
            mode=risk_cfg.get("mode", defaults.mode),
            max_positions=int(risk_cfg.get("max_positions", defaults.max_positions)),
            per_trade_usd=float(risk_cfg.get("per_trade_usd", defaults.per_trade_usd)),
            position_size_pct=float(risk_cfg.get("position_size_pct", defaults.position_size_pct)),
            max_gross_exposure_pct=float(risk_cfg.get("max_gross_exposure_pct", defaults.max_gross_exposure_pct)),
            fee_bps=float(risk_cfg.get("fee_bps", defaults.fee_bps)),
            stop_loss_pct=float(risk_cfg.get("stop_loss_pct", defaults.stop_loss_pct)),
            take_profit_pct=float(risk_cfg.get("take_profit_pct", defaults.take_profit_pct)),
            daily_max_drawdown_pct=float(
                risk_cfg.get("daily_max_drawdown_pct", defaults.daily_max_drawdown_pct)
            ),
            borrow_cost_pct=float(risk_cfg.get("borrow_cost_pct", defaults.borrow_cost_pct)),
        )

class Executor:
    def __init__(self, db, cfg: RiskConfig, runtime: Optional[dict] = None):
        self.db = db
        self.cfg = cfg
        runtime = runtime or {}
        self._paper_mode = bool(runtime.get("paper_trading", cfg.mode == "paper"))
        self._max_positions = int(runtime.get("max_position_size", cfg.max_positions))
        self._quiet = bool(runtime.get("quiet_mode", False))
        self.logger = logging.getLogger("Executor")

    def place_order(self, ticker: str, price: float, side: str, ts) -> None:
        if self._max_positions > 0 and self._open_positions() >= self._max_positions:
            self.logger.warning("Position cap reached, skipping trade")
            return

        if self._paper_mode:
            self.simulate_order(ticker, price, side, ts)
        else:
            self.place_alpaca_order(ticker, price, side, ts)

    def simulate_order(self, ticker: str, price: float, side: str, ts) -> None:
        qty = max(1.0, self.cfg.per_trade_usd / max(1.0, price))
        rationale = "Simulated paper trade"
        self.db.record_trade(
            ticker,
            side,
            qty,
            price,
            rationale=rationale,
            ts=ts,
        )
        if not self._quiet:
            self.logger.info(
                "Simulated trade executed: %s %s qty=%.2f price=%.2f", side, ticker, qty, price
            )

    def place_paper_order(self, ticker: str, price: float, side: str, ts):  # backward compatibility
        self.simulate_order(ticker, price, side, ts)

    def place_alpaca_order(self, ticker: str, price: float, side: str, ts) -> None:  # pragma: no cover
        if not self._quiet:
            self.logger.info("Alpaca order placed: %s %s at %.2f", side, ticker, price)

    def _open_positions(self) -> int:
        df = self.db.to_df("SELECT COUNT(*) AS c FROM positions WHERE closed_at IS NULL")
        if df.empty:
            return 0
        return int(df.iloc[0]["c"])
