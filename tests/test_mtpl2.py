import pandas as pd

from src.analytics.build_mtpl2_db import build_database
from src.analytics.mtpl2_queries import regional_summary
from src.data.mtpl2 import prepare_mtpl2


def test_prepare_mtpl2_joins_claim_severity():
    freq = pd.DataFrame(
        {
            "IDpol": [1, 2],
            "ClaimNb": [2, 0],
            "Exposure": [1.0, 0.5],
            "VehPower": [6, 8],
            "VehAge": [3, 7],
            "DrivAge": [35, 52],
            "BonusMalus": [70, 90],
            "VehBrand": ["B1", "B2"],
            "VehGas": ["Regular", "Diesel"],
            "Area": ["A", "F"],
            "Density": [100, 1000],
            "Region": ["R1", "R2"],
        }
    )
    sev = pd.DataFrame({"IDpol": [1, 1], "ClaimAmount": [100.0, 300.0]})

    out = prepare_mtpl2(freq, sev)

    assert out.loc[out["IDpol"] == 1, "total_claim_amount"].iloc[0] == 400.0
    assert out.loc[out["IDpol"] == 1, "avg_claim_amount"].iloc[0] == 200.0
    assert out.loc[out["IDpol"] == 2, "total_claim_amount"].iloc[0] == 0.0
    assert int(out["has_claim"].sum()) == 1


def test_regional_summary_contains_window_rank(tmp_path):
    freq = pd.DataFrame(
        {
            "IDpol": [1, 2, 3],
            "ClaimNb": [2, 1, 0],
            "Exposure": [1.0, 1.0, 1.0],
            "VehPower": [6, 8, 7],
            "VehAge": [3, 7, 4],
            "DrivAge": [35, 52, 40],
            "BonusMalus": [70, 90, 60],
            "VehBrand": ["B1", "B2", "B3"],
            "VehGas": ["Regular", "Diesel", "Regular"],
            "Area": ["A", "F", "A"],
            "Density": [100, 1000, 200],
            "Region": ["R1", "R2", "R2"],
        }
    )
    sev = pd.DataFrame({"IDpol": [1, 2], "ClaimAmount": [100.0, 200.0]})
    prepared = prepare_mtpl2(freq, sev)
    csv_path = tmp_path / "prepared.csv"
    db_path = tmp_path / "claims.db"
    prepared.to_csv(csv_path, index=False)
    build_database(csv_path, db_path)

    out = regional_summary(db_path)
    assert "frequency_rank" in out.columns
    assert int(out.iloc[0]["frequency_rank"]) == 1
    assert out.iloc[0]["region"] == "R1"
