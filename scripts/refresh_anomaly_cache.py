"""Refresh anomaly cache tables in an existing ClaimsIQ real-data database."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml.mtpl2_anomaly import materialize_anomaly_tables
from src.analytics.cache import refresh_analytics_cache


def refresh(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "claim_anomalies" not in tables:
            raise RuntimeError("claim_anomalies table not found. Run bootstrap_real.py first.")
        import pandas as pd
        df = pd.read_sql_query("SELECT * FROM claim_anomalies", conn)

    if "anomaly_percentile_pct" not in df.columns and "anomaly_score" in df.columns:
        df["anomaly_percentile_pct"] = df["anomaly_score"] * 100.0

    required = {"anomaly_score", "is_anomaly", "Region"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"claim_anomalies is missing required columns: {sorted(missing)}")

    materialize_anomaly_tables(db_path, df)
    analytics = refresh_analytics_cache(db_path)
    print(f"Refreshed anomaly cache for {len(df):,} rows.")
    print(f"Refreshed analytics cache: {analytics}")
    print("Anomaly, KPI, regional, fuel and age tools now use indexed/materialised tables.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "claims_real.db")
    args = parser.parse_args()
    refresh(args.db)
