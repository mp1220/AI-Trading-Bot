"""Reporting utilities for Trader_Model."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict


def market_pulse(signals: list[dict]) -> str:
    lines = ["Market Pulse"]
    for s in signals:
        decision = s.get("decision") or s.get("action") or "HOLD"
        lines.append(
            f"{s['ticker']}: prob_up={s['prob_up']:.2f} sent={s['sentiment']:.2f} dec={decision}"
        )
    return "\n".join(lines)


def write_daily_report(report_dir: Path, summary: Dict[str, str | int | float]) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d")
    path = report_dir / f"daily_report_{stamp}.txt"
    lines = [
        "Trader_Model Daily Report",
        "-------------------------",
        f"Generated at: {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC",
        "",
        f"Trades executed: {summary.get('trades_executed', 0)}",
        f"Articles ingested: {summary.get('articles_ingested', 0)}",
        f"Model accuracy: {summary.get('model_accuracy', 'n/a')}",
        f"Retraining count: {summary.get('retraining_count', 0)}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
