from src.evaluation.rag import recall_at_k


class FakeRetriever:
    def __init__(self, results):
        self.results = results

    def search(self, query, k=3):
        return self.results.get(query, [])[:k]


def test_recall_at_k():
    cases = [
        {"query": "a", "expected_policy_ids": ["P-001"]},
        {"query": "b", "expected_policy_ids": ["P-002"]},
    ]
    retriever = FakeRetriever({
        "a": [{"policy_id": "P-001"}],
        "b": [{"policy_id": "P-999"}, {"policy_id": "P-002"}],
    })
    assert recall_at_k(retriever, cases, 2) == 1.0


def test_negative_retrieval_is_rejected():
    class DistanceRetriever:
        def search(self, query, k=3):
            return [{"policy_id": "P-001", "distance": 1.8}]

    from src.evaluation.rag import evaluate

    out = evaluate(DistanceRetriever(), [{"query": "unrelated", "expected_policy_ids": [], "max_distance": 1.25}])
    assert out["negative_rejection_rate"] == 1.0


def test_negative_retrieval_uses_hybrid_acceptance():
    class HybridRetriever:
        def diagnose(self, query, k=3):
            return [
                {
                    "policy_id": "P-001",
                    "distance": 1.18,
                    "lexical_coverage": 0.33,
                    "hybrid_score": 0.57,
                    "accepted": False,
                }
            ]

        def search(self, query, k=3):
            return []

    from src.evaluation.rag import evaluate

    out = evaluate(HybridRetriever(), [{"query": "health insurance deductibles", "expected_policy_ids": []}])
    assert out["negative_rejection_rate"] == 1.0
    assert out["negative_nearest_distance_min"] == 1.18
