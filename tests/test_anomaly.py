from pathlib import Path
import sqlite3

import pandas as pd

from src.analytics.build_mtpl2_db import build_database
from src.ml.mtpl2_anomaly import detect_anomalies


def _tiny_mtpl2(n: int = 120) -> pd.DataFrame:
    rows = []
    for i in range(n):
        has_claim = int(i % 5 == 0)
        claim = float(has_claim * (300 + i))
        rows.append(
            {
                "IDpol": i + 1,
                "ClaimNb": has_claim,
                "Exposure": 0.5 + (i % 10) / 10,
                "VehPower": 5 + (i % 8),
                "VehAge": 1 + (i % 15),
                "DrivAge": 24 + (i % 50),
                "BonusMalus": 50 + (i % 70),
                "VehBrand": f"B{i % 4}",
                "VehGas": "Diesel" if i % 2 else "Regular",
                "Area": f"A{i % 3}",
                "Density": 100 + i * 5,
                "Region": f"R{i % 5}",
                "claim_record_count": has_claim,
                "total_claim_amount": claim,
                "avg_claim_amount": claim,
                "max_claim_amount": claim,
                "has_claim": has_claim,
                "claim_frequency_per_year": has_claim,
                "loss_cost_per_exposure_year": has_claim * 200,
                "claim_severity_per_claim": claim,
            }
        )
    return pd.DataFrame(rows)


def test_anomaly_pipeline_creates_review_table(tmp_path: Path):
    source = tmp_path / "prepared.csv"
    db = tmp_path / "claims.db"
    _tiny_mtpl2().to_csv(source, index=False)
    build_database(source, db)

    result = detect_anomalies(db, contamination=0.05)

    assert "anomaly_score" in result.columns
    assert result["anomaly_score"].between(0, 1).all()
    assert result["review_band"].isin(["Normal", "Watch", "High"]).all()

    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM claim_anomalies").fetchone()[0] == len(result)
        assert conn.execute("SELECT COUNT(*) FROM anomaly_region_summary").fetchone()[0] > 0
        assert "anomaly_percentile_pct" in result.columns
