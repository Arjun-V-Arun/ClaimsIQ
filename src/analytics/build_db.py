"""Load cleaned claims into SQLite."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


def build_database(csv_path: Path, db_path: Path) -> None:
    df = pd.read_csv(csv_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql("claims", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_claims_region ON claims(region)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_claims_customer ON claims(customer_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_claims_date ON claims(claim_date)")
        conn.commit()
    print(f"Created {db_path} with {len(df):,} claims")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    args = parser.parse_args()
    build_database(args.input, args.db)


if __name__ == "__main__":
    main()
