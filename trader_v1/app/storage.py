"""Database access layer built on SQLAlchemy."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional, Tuple

import pandas as pd
from sqlalchemy import text
from app.db_utils import get_engine

LOGGER = logging.getLogger("Storage")


class DB:
    def __init__(self, engine=None):
        self.engine = engine or get_engine()
        self.ensure_schema()

    def close(self) -> None:
        try:
            self.engine.dispose()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------
    def ensure_schema(self) -> None:
        """Ensure all required database tables and constraints exist."""
        statements = [
            """
            CREATE TABLE IF NOT EXISTS market_bars (
                id SERIAL PRIMARY KEY,
                symbol TEXT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                volume DOUBLE PRECISION,
                UNIQUE (symbol, timestamp)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS news_headlines (
                id TEXT,
                ticker TEXT,
                headline TEXT NOT NULL,
                source TEXT,
                url TEXT,
                published_at TIMESTAMPTZ,
                sentiment DOUBLE PRECISION,
                inserted_at TIMESTAMPTZ DEFAULT NOW(),
                headline_hash TEXT
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS social_sentiment (
                id SERIAL PRIMARY KEY,
                detected_at TIMESTAMPTZ NOT NULL,
                ticker TEXT NOT NULL,
                subreddit TEXT,
                score DOUBLE PRECISION,
                mentions INT,
                captured_at TIMESTAMPTZ
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS live_quotes (
                symbol TEXT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                price DOUBLE PRECISION,
                PRIMARY KEY (symbol, timestamp)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS signals (
                id SERIAL PRIMARY KEY,
                ticker TEXT NOT NULL,
                ts TIMESTAMPTZ NOT NULL,
                prob_up DOUBLE PRECISION,
                sentiment DOUBLE PRECISION,
                action TEXT,
                details JSONB,
                UNIQUE (ticker, ts)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS positions (
                id SERIAL PRIMARY KEY,
                ticker TEXT NOT NULL,
                side TEXT,
                opened_at TIMESTAMPTZ NOT NULL,
                closed_at TIMESTAMPTZ,
                qty DOUBLE PRECISION,
                entry_price DOUBLE PRECISION,
                exit_price DOUBLE PRECISION,
                pnl DOUBLE PRECISION,
                status TEXT,
                UNIQUE (ticker, opened_at)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                ticker TEXT NOT NULL,
                ts TIMESTAMPTZ NOT NULL,
                side TEXT,
                qty DOUBLE PRECISION,
                price DOUBLE PRECISION,
                fees DOUBLE PRECISION,
                pnl DOUBLE PRECISION,
                rationale TEXT,
                signal_id INT,
                position_id INT,
                UNIQUE (ticker, ts, side)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS sync_log (
                name TEXT PRIMARY KEY,
                last_sync TIMESTAMPTZ
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id SERIAL PRIMARY KEY,
                recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                mode TEXT NOT NULL,
                accuracy DOUBLE PRECISION,
                precision DOUBLE PRECISION,
                recall DOUBLE PRECISION,
                f1 DOUBLE PRECISION,
                notes JSONB
            );
            """,
        ]

        # ✅ Use engine.begin() instead of nested connection.begin()
        with self.engine.begin() as conn:
            for stmt in statements:
                conn.exec_driver_sql(stmt)

            def _column_exists(table: str, column: str) -> bool:
                result = conn.execute(
                    text(
                        """
                        SELECT COUNT(*) FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = :table
                          AND column_name = :column
                        """
                    ),
                    {"table": table, "column": column},
                ).scalar()
                return bool(result)

            def _add_column_if_missing(table: str, column: str, definition: str) -> None:
                if _column_exists(table, column):
                    LOGGER.info("[Schema] Skipped existing migration %s.%s column already exists.", table, column)
                    return
                try:
                    conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
                    LOGGER.info("[Schema] ✅ Added column %s.%s.", table, column)
                except Exception as exc:
                    LOGGER.warning("[Schema] ⚠️ Could not add column %s.%s: %s", table, column, exc)

            def _primary_key_exists(table: str) -> bool:
                result = conn.execute(
                    text(
                        """
                        SELECT COUNT(*) FROM information_schema.table_constraints
                        WHERE table_schema = 'public'
                          AND table_name = :table
                          AND constraint_type = 'PRIMARY KEY'
                        """
                    ),
                    {"table": table},
                ).scalar()
                return bool(result)

            def _ensure_primary_key(table: str, columns: str) -> None:
                if _primary_key_exists(table):
                    LOGGER.info("[Schema] Skipped existing migration primary key on %s.", table)
                    return
                try:
                    conn.exec_driver_sql(f"ALTER TABLE {table} ADD PRIMARY KEY ({columns})")
                    LOGGER.info("[Schema] ✅ Added primary key on %s.", table)
                except Exception as exc:
                    LOGGER.warning("[Schema] ⚠️ Could not add primary key on %s: %s", table, exc)

            def _constraint_exists(name: str) -> bool:
                result = conn.execute(
                    text("SELECT COUNT(*) FROM pg_constraint WHERE conname = :name"),
                    {"name": name},
                ).scalar()
                return bool(result)

            # Run migrations safely (no nested transactions)
            _add_column_if_missing("news_headlines", "ticker", "TEXT")
            _add_column_if_missing("news_headlines", "headline_hash", "TEXT")

            conn.exec_driver_sql(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_news_unique "
                "ON news_headlines (ticker, source, published_at, headline_hash)"
            )

            _add_column_if_missing("sync_log", "name", "TEXT")
            _add_column_if_missing("sync_log", "last_sync", "TIMESTAMPTZ")
            _ensure_primary_key("sync_log", "name")

            _add_column_if_missing("positions", "side", "TEXT")

            # --- Ensure trades table has all required columns ---
            _add_column_if_missing("trades", "signal_id", "INT")
            _add_column_if_missing("trades", "position_id", "INT")
            _add_column_if_missing("trades", "pnl", "DOUBLE PRECISION")
            _add_column_if_missing("trades", "rationale", "TEXT")

            live_quotes_constraint = "live_quotes_symbol_timestamp_key"
            if not _constraint_exists(live_quotes_constraint):
                try:
                    conn.exec_driver_sql(
                        "ALTER TABLE live_quotes "
                        "ADD CONSTRAINT live_quotes_symbol_timestamp_key UNIQUE (symbol, timestamp)"
                    )
                    LOGGER.info("[Schema] ✅ Added live_quotes unique constraint.")
                except Exception as exc:
                    LOGGER.warning("[Schema] ⚠️ Could not add constraint: %s", exc)
            else:
                LOGGER.info("[Schema] Skipped: live_quotes unique constraint already exists.")

        LOGGER.info("✅ Schema verified/created (including live_quotes unique constraint).")
        # NOTE: Always define new trading schema columns here for auto-migration.

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def execute(self, sql: str, params: Optional[dict] = None) -> None:
        with self.engine.begin() as conn:
            conn.exec_driver_sql(sql, params or {})

    def executemany(self, sql: str, rows: Iterable[dict]) -> None:
        rows = list(rows)
        if not rows:
            return
        with self.engine.begin() as conn:
            conn.exec_driver_sql(sql, rows)

    def to_df(self, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
        return pd.read_sql(sql, self.engine, params=params)

    def traded_within(self, ticker: str, minutes: int) -> bool:
        """Return True when ticker traded within the last `minutes` minutes."""
        if minutes <= 0:
            return False
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(
                        """
                        SELECT COUNT(*) FROM trades
                        WHERE ticker = :ticker AND ts >= :cutoff
                        """
                    ),
                    {"ticker": ticker, "cutoff": cutoff},
                ).scalar()
            return bool(result and result > 0)
        except Exception as exc:
            LOGGER.error("[DB] traded_within() failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Sync tracking
    # ------------------------------------------------------------------
    def get_last_sync(self, key: str) -> Optional[datetime]:
        with self.engine.connect() as conn:
            result = conn.exec_driver_sql(
                "SELECT last_sync FROM sync_log WHERE name=%(name)s",
                {"name": key},
            ).fetchone()
        return result[0] if result else None

    def set_last_sync(self, key: str, value: datetime) -> None:
        with self.engine.begin() as conn:
            conn.exec_driver_sql(
                """
                INSERT INTO sync_log (name, last_sync)
                VALUES (%(name)s, %(ts)s)
                ON CONFLICT (name)
                DO UPDATE SET last_sync = EXCLUDED.last_sync
                """,
                {"name": key, "ts": value},
            )

    # ------------------------------------------------------------------
    # Market bars
    # ------------------------------------------------------------------
    def latest_bar_timestamp(self, symbol: str) -> Optional[datetime]:
        with self.engine.connect() as conn:
            result = conn.exec_driver_sql(
                "SELECT MAX(timestamp) FROM market_bars WHERE symbol=%(symbol)s",
                {"symbol": symbol},
            ).fetchone()
        return result[0] if result else None

    def upsert_bars_multi(self, df: pd.DataFrame, update: bool = False) -> Tuple[int, int]:
        if df is None or df.empty:
            return 0, 0
        working = df.copy()
        working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
        working = working.dropna(subset=["timestamp", "symbol"])
        if working.empty:
            return 0, 0

        records = [
            {
                "symbol": row.symbol,
                "timestamp": row.timestamp,
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": row.volume,
            }
            for row in working.itertuples(index=False)
        ]
        conflict_clause = (
            "DO UPDATE SET open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, "
            "close=EXCLUDED.close, volume=EXCLUDED.volume"
            if update
            else "DO NOTHING"
        )
        sql = (
            "INSERT INTO market_bars (symbol, timestamp, open, high, low, close, volume) "
            "VALUES (%(symbol)s, %(timestamp)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s) "
            "ON CONFLICT (symbol, timestamp) "
            f"{conflict_clause}"
        )
        with self.engine.begin() as conn:
            result = conn.exec_driver_sql(sql, records)
        inserted = result.rowcount if result.rowcount is not None else 0
        skipped = len(records) - inserted if not update else 0
        return inserted, skipped

    # ------------------------------------------------------------------
    # Live quotes
    # ------------------------------------------------------------------
    def store_live_quote(self, symbol: str, price: float, timestamp) -> None:
        with self.engine.begin() as conn:
            conn.exec_driver_sql(
                """
                INSERT INTO live_quotes (symbol, timestamp, price)
                VALUES (%(symbol)s, %(timestamp)s, %(price)s)
                ON CONFLICT (symbol, timestamp)
                DO UPDATE SET price = EXCLUDED.price
                """,
                {"symbol": symbol, "timestamp": timestamp, "price": price},
            )

    def record_trade(
        self,
        ticker: str,
        side: str,
        qty: float,
        price: float,
        signal_id: Optional[int] = None,
        position_id: Optional[int] = None,
        pnl: Optional[float] = None,
        rationale: Optional[str] = None,
        ts: Optional[datetime] = None,
    ) -> None:
        """Log executed trades into the database."""
        ts = ts or datetime.now(timezone.utc)
        side_value = (side or "").upper()
        try:
            with self.engine.begin() as conn:
                conn.exec_driver_sql(
                    """
                    INSERT INTO trades (ticker, ts, side, qty, price, pnl, rationale, signal_id, position_id)
                    VALUES (%(ticker)s, %(ts)s, %(side)s, %(qty)s, %(price)s, %(pnl)s, %(rationale)s, %(signal_id)s, %(position_id)s)
                    ON CONFLICT (ticker, ts, side)
                    DO UPDATE SET
                        qty = EXCLUDED.qty,
                        price = EXCLUDED.price,
                        pnl = EXCLUDED.pnl,
                        rationale = EXCLUDED.rationale
                    """,
                    {
                        "ticker": ticker,
                        "ts": ts,
                        "side": side_value,
                        "qty": qty,
                        "price": price,
                        "pnl": pnl,
                        "rationale": rationale,
                        "signal_id": signal_id,
                        "position_id": position_id,
                    },
                )
            LOGGER.info("[DB] ✅ Recorded trade: %s %s @ %.2f", side_value, ticker, price)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("[DB] ⚠️ Failed to record trade for %s: %s", ticker, exc)


def get_db() -> DB:
    return DB()


_DEFAULT_DB: Optional[DB] = None


def _get_default_db() -> DB:
    global _DEFAULT_DB
    if _DEFAULT_DB is None:
        _DEFAULT_DB = DB()
    return _DEFAULT_DB


def store_live_quote(symbol: str, price: float, timestamp) -> None:
    db = _get_default_db()
    db.store_live_quote(symbol, price, timestamp)
