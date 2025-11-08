"""Basic system healthcheck for Trader_Model."""
from __future__ import annotations

import os

import requests
from sqlalchemy import text

from app.db_utils import get_engine


def run_healthcheck() -> None:
    print("🩺 Trader System Healthcheck")

    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Database connection OK")
    except Exception as exc:  # noqa: BLE001
        print("❌ DB error:", exc)

    env_keys = [
        "ALPACA_API_KEY_ID",
        "ALPACA_API_SECRET_KEY",
        "ALPACA_API_BASE_URL",
        "FMP_API_KEY",
        "FINNHUB_API_KEY",
        "OPENAI_API_KEY",
    ]
    for key in env_keys:
        status = "set" if os.getenv(key) else "missing"
        print(f"🔑 {key}: {status}")

    webhook = os.getenv("DISCORD_WEBHOOK")
    if webhook:
        try:
            response = requests.post(webhook, json={"content": "✅ Healthcheck passed"}, timeout=10)
            response.raise_for_status()
            print("✅ Discord reachable")
        except Exception as exc:  # noqa: BLE001
            print("⚠️ Discord issue:", exc)
    else:
        print("⚠️ Discord webhook missing")


if __name__ == "__main__":
    run_healthcheck()
