from pathlib import Path

from src.agent.tools import ClaimsTools
from src.data.generate import generate_claims
from src.data.preprocess import clean_claims
from src.analytics.build_db import build_database


def test_find_claim(tmp_path: Path):
    raw = tmp_path / "raw.csv"
    db = tmp_path / "claims.db"
    clean = clean_claims(generate_claims(50, seed=7))
    clean.to_csv(raw, index=False)
    build_database(raw, db)

    # The anomaly table is optional in the MVP, so this tests that the database
    # tool layer exposes a well-defined not-found response until the ML stage is run.
    tools = ClaimsTools(db, tmp_path / "chroma")
    assert tools.find_claim("CLM-9999999") is None
