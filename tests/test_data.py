import pandas as pd

from src.data.generate import generate_claims
from src.data.preprocess import clean_claims


def test_generation_is_reproducible():
    a = generate_claims(100, seed=42)
    b = generate_claims(100, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_clean_claims_has_required_features():
    cleaned = clean_claims(generate_claims(100, seed=42))
    assert cleaned["claim_id"].is_unique
    assert cleaned["resolution_days"].ge(0).all()
    assert cleaned["repair_ratio"].ge(0).all()
    assert cleaned["claim_month"].notna().all()
