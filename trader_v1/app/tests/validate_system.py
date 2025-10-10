import os
import time
import threading

import websocket
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()


def log_ok(label, msg="OK"):
    print(f"[Preflight] {label}: ✅ {msg}")


def log_fail(label, msg):
    print(f"[Preflight] {label}: ❌ FAIL — {msg}")


def run_validation_and_report():
    # -------------------- DATABASE CHECK --------------------
    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        log_fail("Database", "PG_DSN not set in .env")
        return False
    try:
        engine = create_engine(pg_dsn)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        log_ok("Database", "Connection verified")
        db_ok = True
    except Exception as e:
        log_fail("Database", f"DB unexpected error: {e}")
        db_ok = False

    # -------------------- ALPACA REST CHECK --------------------
    rest_ok = False
    api_key = os.getenv("ALPACA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not api_secret:
        log_fail("REST", "Missing ALPACA_API_KEY_ID or ALPACA_API_SECRET_KEY in .env")
        rest_ok = False
    else:
        print("[Preflight] Using ALPACA API keys from .env")
        try:
            from alpaca.trading.client import TradingClient

            base_url = os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets")
            paper_mode = True if base_url and "paper" in base_url.lower() else False
            client = TradingClient(api_key, api_secret, paper=paper_mode, url_override=base_url)
            account = client.get_account()
            mode = "PAPER" if paper_mode else "LIVE"
            log_ok("REST", f"(alpaca-py) — Account {account.id} ({mode})")
            rest_ok = True
        except Exception as e2:
            log_fail("REST", f"{e2}")
            rest_ok = False

    # -------------------- WEBSOCKET CHECK --------------------
    ws_ok = False
    try:
        def on_open(ws):
            log_ok("WebSocket", "Connected")
            nonlocal ws_ok
            ws_ok = True
            try:
                if getattr(ws, "sock", None):
                    ws.close()
            except Exception:
                pass

        def on_error(ws, error):
            log_fail("WebSocket", error)

        wss_url = "wss://stream.data.alpaca.markets/v2/iex"
        ws = websocket.WebSocketApp(wss_url, on_open=on_open, on_error=on_error)
        thread = threading.Thread(target=ws.run_forever, daemon=True)
        thread.start()
        time.sleep(3)
        if ws_ok:
            log_ok("WebSocket", "Verified")
        else:
            log_fail("WebSocket", "No response from stream")
    except Exception as e:
        log_fail("WebSocket", e)

    # -------------------- SUMMARY --------------------
    if db_ok and rest_ok and ws_ok:
        print("\n✅ All preflight checks passed. System ready for orchestration.\n")
        return True
    else:
        print("\n❌ Preflight failed — see logs for details.\n")
        return False


if __name__ == "__main__":
    run_validation_and_report()
