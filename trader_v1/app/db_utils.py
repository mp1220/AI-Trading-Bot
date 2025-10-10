"""Shared database utilities for Trader_Model."""
from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

from app.config import normalize_alpaca_env


ROOT_DIR = Path(__file__).resolve().parent.parent
DOTENV_PATH = ROOT_DIR / ".env"


def get_engine(echo: bool = False):
    """Return a SQLAlchemy engine with automatic .env loading and fallback."""
    load_dotenv(dotenv_path=DOTENV_PATH)
    normalize_alpaca_env()
    print(f"✅ Loaded .env from: {DOTENV_PATH}")

    dsn = os.getenv("PG_DSN")
    if dsn:
        print("✅ Config loaded (PG_DSN present)")
        print(f"✅ PG_DSN: {dsn}")
    else:
        print("⚠️  PG_DSN not found — using SQLite memory database.")
        return create_engine("sqlite:///:memory:", echo=echo)

    try:
        engine = create_engine(dsn, pool_pre_ping=True, echo=echo)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Connected to PostgreSQL using SQLAlchemy engine.")
        return engine
    except OperationalError as exc:
        print(f"⚠️  PostgreSQL connection failed: {exc}")
        print("➡️  Using SQLite in-memory fallback.")
        return create_engine("sqlite:///:memory:", echo=echo)


def get_connection():
    """Return an open SQLAlchemy connection."""
    engine = get_engine()
    return engine.connect()
