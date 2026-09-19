"""Hybrid semantic + lexical retrieval for the demonstration policy corpus."""

from __future__ import annotations

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from .hybrid import (
    DEFAULT_LEXICAL_COVERAGE_MIN,
    DEFAULT_SEMANTIC_DISTANCE_MAX,
    accept_hybrid,
    content_tokens,
)
from .index_policy import COLLECTION_NAME, MODEL_NAME
from .policy_docs import POLICIES

class PolicyRetriever:
    def __init__(
        self,
        persist_dir: Path = Path("data/chroma"),
        semantic_distance_max: float = DEFAULT_SEMANTIC_DISTANCE_MAX,
        lexical_coverage_min: float = DEFAULT_LEXICAL_COVERAGE_MIN,
    ) -> None:
        self.model = SentenceTransformer(MODEL_NAME)
        client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = client.get_or_create_collection(COLLECTION_NAME)
        self.semantic_distance_max = semantic_distance_max
        self.lexical_coverage_min = lexical_coverage_min
        self._doc_tokens = {p["policy_id"]: content_tokens(p["text"]) for p in POLICIES}

    def _lexical_coverage(self, query: str, policy_id: str | None) -> float:
        query_tokens = content_tokens(query)
        if not query_tokens or policy_id is None:
            return 0.0
        doc_tokens = self._doc_tokens.get(policy_id, set())
        return len(query_tokens & doc_tokens) / len(query_tokens)

    def _accept(self, distance: float, lexical_coverage: float) -> bool:
        return accept_hybrid(
            distance,
            lexical_coverage,
            semantic_distance_max=self.semantic_distance_max,
            lexical_coverage_min=self.lexical_coverage_min,
        )

    def diagnose(self, query: str, k: int = 3) -> list[dict]:
        """Return ranked candidates including rejected ones for evaluation/debugging."""
        if not query.strip() or self.collection.count() == 0:
            return []

        candidate_k = min(self.collection.count(), max(int(k) * 3, int(k)))
        embedding = self.model.encode([query], normalize_embeddings=True).tolist()
        result = self.collection.query(query_embeddings=embedding, n_results=candidate_k)
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        candidates = []
        for text, metadata, distance in zip(docs, metadatas, distances):
            policy_id = metadata.get("policy_id")
            lexical_coverage = self._lexical_coverage(query, policy_id)
            accepted = self._accept(float(distance), lexical_coverage)
            semantic_similarity = 1.0 - min(float(distance) / 2.0, 1.0)
            hybrid_score = round(0.70 * semantic_similarity + 0.30 * lexical_coverage, 4)
            candidates.append(
                {
                    "text": text,
                    "title": metadata.get("title", "Unknown policy"),
                    "policy_id": policy_id,
                    "distance": float(distance),
                    "lexical_coverage": round(lexical_coverage, 4),
                    "hybrid_score": hybrid_score,
                    "accepted": accepted,
                }
            )

        candidates.sort(key=lambda item: item["hybrid_score"], reverse=True)
        return candidates

    def search(self, query: str, k: int = 3) -> list[dict]:
        candidates = self.diagnose(query, k=k)
        accepted = [item for item in candidates if item["accepted"]]
        return accepted[: max(1, min(int(k), len(accepted)))]
