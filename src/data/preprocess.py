"""Clean claims data and derive analytical/model features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "claim_id",
    "customer_id",
    "device_type",
    "device_age_months",
    "damage_type",
    "claim_amount",
    "repair_cost",
    "claim_status",
    "claim_date",
    "resolution_date",
    "region",
    "channel",
    "fraud_signal",
}


def clean_claims(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["claim_date"] = pd.to_datetime(out["claim_date"], errors="coerce")
    out["resolution_date"] = pd.to_datetime(out["resolution_date"], errors="coerce")

    if out["claim_id"].duplicated().any():
        raise ValueError("claim_id must be unique")

    out = out.dropna(subset=["claim_date", "resolution_date"])
    out = out[out["claim_amount"] >= 0]
    out = out[out["repair_cost"] >= 0]
    out["resolution_days"] = (
        out["resolution_date"] - out["claim_date"]
    ).dt.total_seconds() / 86400
    out["repair_ratio"] = np.divide(
        out["repair_cost"],
        out["claim_amount"],
        out=np.zeros(len(out), dtype=float),
        where=out["claim_amount"].to_numpy() != 0,
    )
    out["claim_month"] = out["claim_date"].dt.to_period("M").astype(str)
    out["is_closed"] = out["claim_status"].isin(["Approved", "Rejected", "Repaired"])

    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cleaned = clean_claims(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output, index=False)
    print(f"Wrote {len(cleaned):,} cleaned claims to {args.output}")


if __name__ == "__main__":
    main()
