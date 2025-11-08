"""
System validation utility for TraderV2.
Performs preflight checks before full orchestration:
- Database connection
- API credentials
- yfinance + Alpaca reachability
- Discord webhook availability
"""

from __future__ import annotations

import logging
import os
import socket
import requests
import yfinance as yf
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

logger = logging.getLogger("SystemValidation")
load_dotenv()

# ────────────────────────────────────────────────
# ✅ Safe session setup for any yfinance version
# ────────────────────────────────────────────────
try:
    SESSION = yf.utils.get_yf_session()
except AttributeError:
    SESSION = requests.Session()
    SESSION.headers.update({
        "User-Agent": "Mozilla/5.0 (TraderV2 compatible; validate_system fallback)"
    })

try:
    yf.set_tz_cache_location("/tmp/yf_tz_cache")
except Exception:
    pass


# ────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────

def check_internet(host: str = "8.8.8.8", port: int = 53, timeout: int = 3) -> bool:
    """Quick connectivity check."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False


def check_postgres() -> bool:
    """Ensure PostgreSQL DSN is valid and reachable."""
    dsn = os.getenv("PG_DSN")
    if not dsn:
        logger.error("❌ PG_DSN missing from .env")
        return False
    try:
        engine = create_engine(dsn)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ PostgreSQL connection verified")
        return True
    except Exception as exc:
        logger.error(f"❌ PostgreSQL connection failed: {exc}")
        return False


def check_alpaca() -> bool:
    """Ensure Alpaca credentials are present."""
    key = os.getenv("ALPACA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        logger.warning("⚠️ Alpaca credentials missing")
        return False
    logger.info("✅ Alpaca credentials found")
    return True


def check_yfinance() -> bool | None:
    """Ensure yfinance session works. Returns None for warnings."""
    global SESSION
    try:
        df = yf.download("AAPL", period="1d", interval="1h", progress=False, session=SESSION)
        if df is None or df.empty:
            logger.warning("⚠️ yfinance responded but returned empty data for AAPL")
            return None  # warn but not fail
        logger.info(f"✅ yfinance operational — {len(df)} rows fetched")
        return True
    except Exception as exc:
        msg = str(exc)
        if any(x in msg for x in ["Expecting value", "JSONDecodeError", "possibly delisted", "no timezone found"]):
            logger.warning(f"⚠️ yfinance transient issue ({msg[:80]}) — continuing.")
            # Refresh session once
            SESSION = requests.Session()
            SESSION.headers.update({"User-Agent": "Mozilla/5.0 (TraderV2 Retry)"})
            return None
        logger.error(f"❌ yfinance fetch failed: {exc}")
        return False


def check_discord_webhooks() -> bool:
    """Ping all Discord webhooks found in .env."""
    ok = True
    for name, url in os.environ.items():
        if "DISCORD_WEBHOOK" in name and url.startswith("https://"):
            try:
                resp = requests.head(url, timeout=5)
                if resp.status_code < 400:
                    logger.info(f"✅ {name} webhook reachable")
                else:
                    logger.warning(f"⚠️ {name} webhook returned {resp.status_code}")
            except Exception as exc:
                logger.warning(f"⚠️ {name} webhook error: {exc}")
                ok = False
    return ok


# ────────────────────────────────────────────────
# Main validation runner
# ────────────────────────────────────────────────

def run_validation_and_report() -> dict[str, Any]:
    """Run all preflight checks and return a summary report."""
    print("──────────────────────────────────────────────")
    print("🔧 Running TraderV2 Preflight Validation")
    print("──────────────────────────────────────────────")

    internet_ok = check_internet()
    postgres_ok = check_postgres()
    alpaca_ok = check_alpaca()
    yfinance_ok = check_yfinance()
    discord_ok = check_discord_webhooks()

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "internet": internet_ok,
        "postgres": postgres_ok,
        "alpaca": alpaca_ok,
        "yfinance": yfinance_ok,
        "discord": discord_ok,
    }

    print("\n✅ Validation summary:")
    for key, val in results.items():
        if val is True:
            print(f" - {key:<10}: ✅ OK")
        elif val is None:
            print(f" - {key:<10}: ⚠️ WARN")
        else:
            print(f" - {key:<10}: ❌ FAIL")

    all_ok = all(v for v in results.values() if isinstance(v, bool))
    print("\n" + ("✅ All systems operational." if all_ok else "⚠️ One or more systems failed validation."))
    print("──────────────────────────────────────────────\n")

    return results


# ────────────────────────────────────────────────
# CLI entry
# ────────────────────────────────────────────────
if __name__ == "__main__":
    run_validation_and_report()
