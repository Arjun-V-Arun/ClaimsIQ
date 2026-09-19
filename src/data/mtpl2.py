"""Download and prepare the public French MTPL2 insurance datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

FREQ_OPENML_ID = 41214
SEV_OPENML_ID = 41215


def fetch_mtpl2(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download freMTPL2 frequency + severity datasets from OpenML."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    freq_path = raw_dir / "freMTPL2freq.csv"
    sev_path = raw_dir / "freMTPL2sev.csv"

    if freq_path.exists() and sev_path.exists():
        return pd.read_csv(freq_path), pd.read_csv(sev_path)

    freq = fetch_openml(data_id=FREQ_OPENML_ID, as_frame=True).frame
    sev = fetch_openml(data_id=SEV_OPENML_ID, as_frame=True).frame

    freq.to_csv(freq_path, index=False)
    sev.to_csv(sev_path, index=False)
    return freq, sev


def prepare_mtpl2(freq: pd.DataFrame, sev: pd.DataFrame) -> pd.DataFrame:
    """Join policy-level risk data with claim severity aggregates."""
    freq = freq.copy()
    sev = sev.copy()

    freq["IDpol"] = pd.to_numeric(freq["IDpol"], errors="raise").astype("int64")
    freq["ClaimNb"] = pd.to_numeric(freq["ClaimNb"], errors="raise")
    freq["Exposure"] = pd.to_numeric(freq["Exposure"], errors="raise")
    sev["IDpol"] = pd.to_numeric(sev["IDpol"], errors="raise").astype("int64")
    sev["ClaimAmount"] = pd.to_numeric(sev["ClaimAmount"], errors="raise")

    if freq["IDpol"].duplicated().any():
        raise ValueError("IDpol must be unique in freMTPL2freq")
    if freq["Exposure"].le(0).any():
        raise ValueError("freMTPL2freq contains non-positive Exposure values")
    if sev["ClaimAmount"].lt(0).any():
        raise ValueError("freMTPL2sev contains negative ClaimAmount values")

    claim_agg = (
        sev.groupby("IDpol", as_index=False)
        .agg(
            claim_record_count=("ClaimAmount", "size"),
            total_claim_amount=("ClaimAmount", "sum"),
            avg_claim_amount=("ClaimAmount", "mean"),
            max_claim_amount=("ClaimAmount", "max"),
        )
    )

    out = freq.merge(claim_agg, on="IDpol", how="left", validate="one_to_one")
    out["claim_record_count"] = out["claim_record_count"].fillna(0).astype(int)
    out["total_claim_amount"] = out["total_claim_amount"].fillna(0.0)
    out["avg_claim_amount"] = out["avg_claim_amount"].fillna(0.0)
    out["max_claim_amount"] = out["max_claim_amount"].fillna(0.0)
    out["has_claim"] = out["ClaimNb"].gt(0).astype(int)
    out["claim_frequency_per_year"] = out["ClaimNb"] / out["Exposure"]
    out["loss_cost_per_exposure_year"] = out["total_claim_amount"] / out["Exposure"]
    out["claim_severity_per_claim"] = out["total_claim_amount"].div(
        out["ClaimNb"].where(out["ClaimNb"].gt(0), pd.NA)
    ).fillna(0.0)

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/mtpl2_policy.csv"))
    args = parser.parse_args()

    freq, sev = fetch_mtpl2(args.raw_dir)
    prepared = prepare_mtpl2(freq, sev)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(args.output, index=False)

    print(f"Prepared {len(prepared):,} real motor-insurance policies")
    print(f"Policies with claims: {int(prepared['has_claim'].sum()):,}")
    print(f"Total claims: {int(prepared['ClaimNb'].sum()):,}")
    print(f"Total claim amount: €{prepared['total_claim_amount'].sum():,.2f}")


if __name__ == "__main__":
    main()
