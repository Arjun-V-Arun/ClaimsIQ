from pathlib import Path
import sqlite3

import pandas as pd

from src.analytics.cache import refresh_analytics_cache


def test_refresh_analytics_cache_materialises_small_tables(tmp_path: Path):
    db = tmp_path / "real_like.db"
    df = pd.DataFrame(
        [
            [1, "R1", 1, 1.0, "Regular", 30, 0.1, 100.0, 100.0, 1],
            [2, "R2", 0, 1.0, "Diesel", 45, 0.2, 0.0, 0.0, 0],
            [3, "R1", 2, 0.5, "Regular", 70, 0.3, 600.0, 300.0, 1],
        ],
        columns=[
            "IDpol", "Region", "ClaimNb", "Exposure", "VehGas", "DrivAge",
            "Density", "total_claim_amount", "claim_severity_per_claim", "has_claim",
        ],
    )
    with sqlite3.connect(db) as conn:
        df.to_sql("policies", conn, index=False)

    result = refresh_analytics_cache(db)

    assert result == {"kpis": 1, "regional": 2, "fuel": 2, "age": 3}
    with sqlite3.connect(db) as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
    assert "analytics_regional_summary" in tables
    assert "analytics_fuel_summary" in tables
    assert "analytics_driver_age_summary" in tables
