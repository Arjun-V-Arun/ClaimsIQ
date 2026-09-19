"""Generate reproducible synthetic insurance claims."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REGIONS = ["Northeast", "Midwest", "South", "West"]
CHANNELS = ["Web", "Mobile", "Retail", "Call Center"]
DEVICE_TYPES = ["Phone", "Tablet", "Laptop", "Watch"]
DAMAGE_TYPES = ["Screen Damage", "Water Damage", "Theft", "Battery", "Other"]
STATUSES = ["Approved", "Rejected", "In Review", "Repaired"]


def generate_claims(rows: int, seed: int = 42) -> pd.DataFrame:
    if rows <= 0:
        raise ValueError("rows must be positive")

    rng = np.random.default_rng(seed)
    claim_dates = pd.Timestamp("2026-01-01") + pd.to_timedelta(
        rng.integers(0, 273, size=rows), unit="D"
    )
    customer_ids = rng.integers(10000, 22000, size=rows)
    device_age = rng.integers(1, 61, size=rows)
    claim_amount = np.round(rng.lognormal(mean=7.0, sigma=0.55, size=rows), 2)

    damage = rng.choice(DAMAGE_TYPES, size=rows, p=[0.38, 0.14, 0.10, 0.18, 0.20])
    device = rng.choice(DEVICE_TYPES, size=rows, p=[0.55, 0.14, 0.16, 0.15])
    region = rng.choice(REGIONS, size=rows)
    channel = rng.choice(CHANNELS, size=rows)

    # Add a small, reproducible anomalous population.
    anomaly_mask = rng.random(rows) < 0.025
    claim_amount[anomaly_mask] *= rng.uniform(2.5, 5.5, anomaly_mask.sum())
    device_age[anomaly_mask] = rng.integers(1, 8, anomaly_mask.sum())

    repair_cost = np.round(
        claim_amount * rng.uniform(0.45, 0.9, size=rows), 2
    )
    fraud_signal = (
        (claim_amount > np.quantile(claim_amount, 0.9)).astype(int)
        + (device_age < 6).astype(int)
        + (rng.random(rows) < 0.06).astype(int)
    )

    status_probs = np.full((rows, len(STATUSES)), 0.0)
    status_probs[:, 0] = 0.52
    status_probs[:, 1] = 0.14 + 0.07 * (fraud_signal >= 2)
    status_probs[:, 2] = 0.10
    status_probs[:, 3] = 1.0 - status_probs[:, 0] - status_probs[:, 1] - status_probs[:, 2]
    status = np.array([rng.choice(STATUSES, p=p / p.sum()) for p in status_probs])

    resolution_days = np.clip(
        rng.normal(loc=4.8, scale=2.2, size=rows)
        + 0.8 * (status == "In Review")
        + 1.3 * (damage == "Theft"),
        0.5,
        30,
    )
    resolution_dates = claim_dates + pd.to_timedelta(np.round(resolution_days, 1), unit="D")

    return pd.DataFrame(
        {
            "claim_id": [f"CLM-{i:07d}" for i in range(1, rows + 1)],
            "customer_id": customer_ids,
            "device_type": device,
            "device_age_months": device_age,
            "damage_type": damage,
            "claim_amount": claim_amount,
            "repair_cost": repair_cost,
            "claim_status": status,
            "claim_date": claim_dates,
            "resolution_date": resolution_dates,
            "region": region,
            "channel": channel,
            "fraud_signal": fraud_signal,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50000)
    parser.add_argument("--output", type=Path, default=Path("data/claims.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_claims(args.rows, args.seed)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df):,} synthetic claims to {args.output}")


if __name__ == "__main__":
    main()
