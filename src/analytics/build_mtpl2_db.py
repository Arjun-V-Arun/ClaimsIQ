"""Create a SQLite analytics database from prepared MTPL2 policy data."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from .cache import refresh_analytics_cache


def build_database(csv_path: Path, db_path: Path) -> None:
    df = pd.read_csv(csv_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        df.to_sql("policies", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_policies_region ON policies(Region)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_policies_id ON policies(IDpol)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_policies_claims ON policies(ClaimNb)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_policies_age ON policies(DrivAge)")
        conn.commit()

    cache = refresh_analytics_cache(db_path)
    print(f"Created {db_path} with {len(df):,} policies")
    print(f"Analytics cache: {cache}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    args = parser.parse_args()
    build_database(args.input, args.db)
