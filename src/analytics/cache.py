"""Materialised analytics tables for low-latency conversational queries."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .mtpl2_queries import DRIVER_AGE_SQL, FUEL_SQL, KPI_SQL, REGIONAL_SQL, run_query

CACHE_TABLES = {
    "kpis": "analytics_kpis",
    "regional": "analytics_regional_summary",
    "fuel": "analytics_fuel_summary",
    "age": "analytics_driver_age_summary",
}


def refresh_analytics_cache(db_path: Path) -> dict[str, int]:
    """Materialise small aggregate tables so chat tools avoid full-table scans."""
    db_path = Path(db_path)

    kpis_df = run_query(db_path, KPI_SQL)
    regional_df = run_query(db_path, REGIONAL_SQL)
    fuel_df = run_query(db_path, FUEL_SQL)
    age_df = run_query(db_path, DRIVER_AGE_SQL)

    with sqlite3.connect(db_path) as conn:
        pd.DataFrame([kpis_df.iloc[0].to_dict()]).to_sql(
            CACHE_TABLES["kpis"], conn, if_exists="replace", index=False
        )
        regional_df.to_sql(
            CACHE_TABLES["regional"], conn, if_exists="replace", index=False
        )
        fuel_df.to_sql(
            CACHE_TABLES["fuel"], conn, if_exists="replace", index=False
        )
        age_df.to_sql(
            CACHE_TABLES["age"], conn, if_exists="replace", index=False
        )

        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{CACHE_TABLES['regional']}_rank "
            f"ON {CACHE_TABLES['regional']}(frequency_rank)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{CACHE_TABLES['regional']}_region "
            f"ON {CACHE_TABLES['regional']}(region)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{CACHE_TABLES['fuel']}_fuel "
            f"ON {CACHE_TABLES['fuel']}(fuel_type)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{CACHE_TABLES['age']}_band "
            f"ON {CACHE_TABLES['age']}(driver_age_band)"
        )
        conn.commit()

    return {
        "kpis": len(kpis_df),
        "regional": len(regional_df),
        "fuel": len(fuel_df),
        "age": len(age_df),
    }
