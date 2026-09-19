"""Create a local demo ClaimsIQ environment in one command."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.build_db import build_database
from src.data.generate import generate_claims
from src.data.preprocess import clean_claims
from src.ml.anomaly_detection import detect_anomalies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    raw = args.data_dir / "claims.csv"
    clean = args.data_dir / "claims_clean.csv"
    db = args.data_dir / "claims.db"

    generate_claims(args.rows, args.seed).to_csv(raw, index=False)
    cleaned = clean_claims(generate_claims(args.rows, args.seed))
    cleaned.to_csv(clean, index=False)
    build_database(clean, db)
    result = detect_anomalies(db)

    print("\\nClaimsIQ demo environment ready.")
    print(f"Claims: {len(result):,}")
    print(f"Anomalies flagged: {int(result['is_anomaly'].sum()):,}")
    print("Run: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
