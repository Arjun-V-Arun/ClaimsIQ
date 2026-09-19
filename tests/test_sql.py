from pathlib import Path

from src.analytics.build_db import build_database
from src.analytics.queries import kpis, regional_summary
from src.data.generate import generate_claims
from src.data.preprocess import clean_claims


def test_sql_analytics(tmp_path: Path):
    csv = tmp_path / "claims.csv"
    cleaned_csv = tmp_path / "clean.csv"
    db = tmp_path / "claims.db"

    generate_claims(500, seed=42).to_csv(csv, index=False)
    clean_claims(generate_claims(500, seed=42)).to_csv(cleaned_csv, index=False)
    build_database(cleaned_csv, db)

    kpi = kpis(db).iloc[0]
    regions = regional_summary(db)

    assert int(kpi["total_claims"]) == 500
    assert len(regions) == 4
    assert set(regions["region"]) == {"Northeast", "Midwest", "South", "West"}
