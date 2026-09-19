"""Offline retrieval benchmark for the demonstration policy corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.rag.retrieve import PolicyRetriever

ROOT = Path(__file__).resolve().parents[2]


def load_cases(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(retriever: "PolicyRetriever", cases: list[dict], k: int = 3) -> dict:
    if not cases:
        return {"recall_at_1": 0.0, "recall_at_3": 0.0, "negative_rejection_rate": 0.0, "cases": 0}

    positive = [c for c in cases if c.get("expected_policy_ids")]
    negative = [c for c in cases if not c.get("expected_policy_ids")]

    positive_diagnostics = []

    def positive_recall(case_list: list[dict], kk: int) -> float:
        if not case_list:
            return 0.0
        hits = 0
        for case in case_list:
            results = retriever.search(case["query"], k=kk)
            returned = {item.get("policy_id") for item in results}
            if returned.intersection(case["expected_policy_ids"]):
                hits += 1
            if kk == 1:
                positive_diagnostics.append({
                    "id": case.get("id"),
                    "nearest_distance": results[0].get("distance") if results else None,
                    "nearest_policy_id": results[0].get("policy_id") if results else None,
                    "expected_policy_ids": case.get("expected_policy_ids", []),
                })
        return hits / len(case_list)

    negative_rejections = 0
    negative_diagnostics = []
    for case in negative:
        threshold = float(case.get("max_distance", 1.25))
        if hasattr(retriever, "diagnose"):
            candidates = retriever.diagnose(case["query"], k=k)
        else:
            # Backward-compatible path for simple test doubles that only expose search().
            candidates = [
                {
                    **item,
                    "accepted": isinstance(item.get("distance"), (int, float))
                    and float(item["distance"]) <= threshold,
                }
                for item in retriever.search(case["query"], k=k)
            ]
        nearest = candidates[0] if candidates else None
        rejected = not any(item.get("accepted") for item in candidates)
        negative_rejections += int(rejected)
        negative_diagnostics.append({
            "id": case.get("id"),
            "rejected": rejected,
            "nearest_distance": nearest.get("distance") if nearest else None,
            "threshold": threshold,
            "nearest_lexical_coverage": nearest.get("lexical_coverage") if nearest else None,
            "nearest_hybrid_score": nearest.get("hybrid_score") if nearest else None,
            "nearest_policy_id": nearest.get("policy_id") if nearest else None,
        })

    positive_distances = [
        d["nearest_distance"] for d in positive_diagnostics
        if isinstance(d.get("nearest_distance"), (int, float))
    ]
    negative_distances = [
        d["nearest_distance"] for d in negative_diagnostics
        if isinstance(d.get("nearest_distance"), (int, float))
    ]

    return {
        "recall_at_1": positive_recall(positive, 1),
        "recall_at_3": positive_recall(positive, min(3, k)),
        "negative_rejection_rate": negative_rejections / len(negative) if negative else 1.0,
        "positive_cases": len(positive),
        "negative_cases": len(negative),
        "positive_nearest_distance_max": max(positive_distances) if positive_distances else None,
        "negative_nearest_distance_min": min(negative_distances) if negative_distances else None,
        "positive_diagnostics": positive_diagnostics,
        "negative_diagnostics": negative_diagnostics,
        "cases": len(cases),
    }


def recall_at_k(retriever: "PolicyRetriever", cases: list[dict], k: int = 3) -> float:
    positive = [c for c in cases if c.get("expected_policy_ids")]
    if not positive:
        return 0.0
    hits = 0
    for case in positive:
        results = retriever.search(case["query"], k=k)
        returned = {item.get("policy_id") for item in results}
        if returned.intersection(case["expected_policy_ids"]):
            hits += 1
    return hits / len(positive)


def main() -> None:
    import argparse
    from src.rag.retrieve import PolicyRetriever

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-dir", type=Path, default=ROOT / "data" / "chroma")
    parser.add_argument("--cases", type=Path, default=ROOT / "eval" / "rag_benchmark.json")
    args = parser.parse_args()

    retriever = PolicyRetriever(args.policy_dir)
    cases = load_cases(args.cases)
    print(json.dumps(evaluate(retriever, cases, 3), indent=2))


if __name__ == "__main__":
    main()
