from src.rag.hybrid import accept_hybrid, content_tokens


def test_hybrid_acceptance_rejects_generic_false_positive():
    assert accept_hybrid(0.70, 0.66) is True
    assert accept_hybrid(1.27, 1.00) is True
    assert accept_hybrid(1.18, 0.33) is False
    assert accept_hybrid(1.30, 0.50) is False


def test_token_normalisation_handles_inflections():
    tokens = content_tokens("accidentally dropped devices")
    assert "accidental" in tokens
    assert "drop" in tokens
    assert "device" in tokens
