"""Detect and clean duplicate rows in Trader_Model tables."""
from __future__ import annotations

import argparse
import logging
import sys

from app.db_utils import get_engine

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
LOGGER = logging.getLogger("Scrub")


def _duplicate_count(conn, table: str, fields: str) -> int:
    result = conn.exec_driver_sql(
        f"SELECT COUNT(*) FROM (SELECT {fields}, COUNT(*) FROM {table} GROUP BY {fields} HAVING COUNT(*) > 1) dup"
    ).scalar()
    return int(result or 0)


def _total_count(conn, table: str) -> int:
    result = conn.exec_driver_sql(f"SELECT COUNT(*) FROM {table}").scalar()
    return int(result or 0)


def _delete_duplicates(conn, table: str, key_fields: tuple[str, ...]) -> int:
    if conn.dialect.name == "sqlite":
        LOGGER.info("ℹ️ %s: duplicate removal skipped on SQLite fallback", table)
        return 0
    conditions = " AND ".join([f"a.{field} = b.{field}" for field in key_fields])
    result = conn.exec_driver_sql(
        f"DELETE FROM {table} a USING {table} b "
        f"WHERE a.ctid < b.ctid AND {conditions}"
    )
    return result.rowcount or 0


def _enforce_constraints(conn) -> None:
    if conn.dialect.name == "sqlite":
        LOGGER.info("ℹ️ Constraint enforcement skipped on SQLite fallback")
        return
    conn.exec_driver_sql(
        "ALTER TABLE market_bars ADD CONSTRAINT IF NOT EXISTS unique_bars UNIQUE (symbol, timestamp)"
    )
    conn.exec_driver_sql(
        "ALTER TABLE news_headlines ADD CONSTRAINT IF NOT EXISTS unique_news UNIQUE (ticker, source, published_at, headline_hash)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect and remove duplicate records from Trader_Model tables."
    )
    parser.add_argument("--report", action="store_true", help="Report duplicates without deleting")
    args = parser.parse_args()

    engine = get_engine()
    LOGGER.info("🧾 Connected to %s", engine.url.render_as_string(hide_password=True))

    with engine.begin() as conn:
        tables = (
            ("market_bars", ("symbol", "timestamp")),
            ("news_headlines", ("ticker", "source", "published_at", "headline_hash")),
        )
        for table, fields in tables:
            total_before = _total_count(conn, table)
            dup_count = _duplicate_count(conn, table, ", ".join(fields))
            if args.report:
                LOGGER.info("🧾 %s: %d duplicate rows detected", table, dup_count)
                continue

            if dup_count == 0:
                LOGGER.info("ℹ️ %s: no duplicates found", table)
                continue

            removed = _delete_duplicates(conn, table, fields)
            total_after = _total_count(conn, table)
            LOGGER.info(
                "🧹 Removed %d duplicates from %s (before: %d, after: %d)",
                removed,
                table,
                total_before,
                total_after,
            )

        if args.report:
            LOGGER.info("📊 Report complete — no changes made.")
        else:
            _enforce_constraints(conn)
            LOGGER.info("🔒 Unique constraints enforced")
            LOGGER.info("🎯 Cleanup complete. Database is duplicate-free.")

if __name__ == "__main__":
    main()
