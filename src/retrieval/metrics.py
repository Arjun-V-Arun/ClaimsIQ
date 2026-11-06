"""Ranking metrics and their uncertainty.

Every metric is first computed per question, so intervals and paired
comparisons resample questions rather than recomputing anything.

Many questions have more than one correct answer, so two families are kept
apart: success@k asks whether any correct answer is in the top k, and recall@k
asks what share of the correct answers are.
"""
from __future__ import annotations

import numpy as np

SUCCESS_AT = (1, 5, 20, 100)
RECALL_AT = (20, 100)
HEADLINE = ("success@5", "mrr@10", "ndcg@10")


def per_question(ranked: np.ndarray, gold: list[tuple[int, ...]]) -> dict[str, np.ndarray]:
    n = len(gold)
    hits = np.zeros(ranked.shape, dtype=bool)
    for i, answers in enumerate(gold):
        hits[i] = np.isin(ranked[i], answers)
    sizes = np.array([len(a) for a in gold], dtype=float)

    out: dict[str, np.ndarray] = {}
    for k in SUCCESS_AT:
        if k <= ranked.shape[1]:
            out[f"success@{k}"] = hits[:, :k].any(axis=1).astype(float)
    for k in RECALL_AT:
        if k <= ranked.shape[1]:
            out[f"recall@{k}"] = hits[:, :k].sum(axis=1) / sizes

    top10 = hits[:, :10]
    first = np.where(top10.any(axis=1), top10.argmax(axis=1), -1)
    out["mrr@10"] = np.divide(1.0, first + 1, out=np.zeros(n), where=first >= 0)

    discounts = 1.0 / np.log2(np.arange(2, 12))
    dcg = (top10 * discounts[: top10.shape[1]]).sum(axis=1)
    ideal = np.array([discounts[: int(min(s, 10))].sum() for s in sizes])
    out["ndcg@10"] = dcg / ideal
    assert all(len(v) == n for v in out.values())
    return out


def summarize(scores: dict[str, np.ndarray], resamples: int = 1000,
              seed: int = 0) -> dict[str, dict]:
    rng = np.random.default_rng(seed)
    n = len(next(iter(scores.values())))
    draws = rng.integers(0, n, size=(resamples, n))
    out = {}
    for name, values in scores.items():
        means = values[draws].mean(axis=1)
        low, high = np.quantile(means, [0.025, 0.975])
        out[name] = {"estimate": round(float(values.mean()), 4),
                     "low": round(float(low), 4), "high": round(float(high), 4)}
    return out


def paired(base: np.ndarray, candidate: np.ndarray, resamples: int = 2000,
           seed: int = 0) -> dict:
    """Candidate minus base on the same questions, with a 95% interval."""
    rng = np.random.default_rng(seed)
    difference = candidate - base
    draws = rng.integers(0, len(difference), size=(resamples, len(difference)))
    means = difference[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"estimate": round(float(difference.mean()), 4),
            "low": round(float(low), 4), "high": round(float(high), 4),
            "better": int((difference > 0).sum()), "worse": int((difference < 0).sum())}
