"""Bootstrap ClaimsIQ on the public real-world French MTPL2 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.build_mtpl2_db import build_database
from src.data.mtpl2 import fetch_mtpl2, prepare_mtpl2
from src.ml.mtpl2_anomaly import detect_anomalies
from src.rag.index_policy import index_policies


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ClaimsIQ real-data environment")
    parser.add_argument("--contamination", type=float, default=0.02)
    parser.add_argument("--skip-rag", action="store_true")
    args = parser.parse_args()

    raw_dir = ROOT / "data" / "raw"
    prepared_path = ROOT / "data" / "mtpl2_policy.csv"
    db_path = ROOT / "data" / "claims_real.db"
    chroma_dir = ROOT / "data" / "chroma"

    freq, sev = fetch_mtpl2(raw_dir)
    prepared = prepare_mtpl2(freq, sev)
    prepared.to_csv(prepared_path, index=False)
    build_database(prepared_path, db_path)
    result = detect_anomalies(db_path, contamination=args.contamination)

    indexed = 0
    if not args.skip_rag:
        indexed = index_policies(chroma_dir)

    print("\nClaimsIQ real-data environment ready.")
    print(f"Policies: {len(result):,}")
    print(f"Policies with claims: {int(result['has_claim'].sum()):,}")
    print(f"Claims: {int(result['ClaimNb'].sum()):,}")
    print(f"Statistical review signals: {int(result['is_anomaly'].sum()):,}")
    print(f"Demonstration policy docs indexed: {indexed:,}")
    print("Run: python -m streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
