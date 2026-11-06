"""Embedding retrieval, reciprocal rank fusion, and cross-encoder reranking.

Models are loaded only when used, so the keyword baseline, the tests, and CI
never download anything. Each model is wrapped behind a small interface, which
also lets the tests run the full pipeline with deterministic stand-ins.
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

BATCH = 512

# Each family was trained with its own prefixes. Leaving them out quietly costs
# accuracy, so they are part of the model's definition here.
PREFIXES = {
    "sentence-transformers/all-MiniLM-L6-v2": ("", ""),
    "BAAI/bge-small-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "intfloat/e5-small-v2": ("query: ", "passage: "),
}
DENSE_MODELS = tuple(PREFIXES)
CURRENT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # what the copilot used
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Encoder(Protocol):
    name: str

    def encode(self, texts: list[str], kind: str) -> np.ndarray: ...


class Scorer(Protocol):
    name: str

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray: ...


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def device_name() -> str:
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"


class SentenceEncoder:
    def __init__(self, name: str):
        from sentence_transformers import SentenceTransformer
        self.name = name
        self.query_prefix, self.passage_prefix = PREFIXES[name]
        self.model = SentenceTransformer(name, device=_device())

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        prefix = self.query_prefix if kind == "query" else self.passage_prefix
        return self.model.encode([prefix + t for t in texts], batch_size=128,
                                 normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=len(texts) > 5000)


class CrossEncoderScorer:
    def __init__(self, name: str = RERANKER):
        from sentence_transformers import CrossEncoder
        self.name = name
        self.model = CrossEncoder(name, device=_device(), max_length=384)

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        return np.asarray(self.model.predict(pairs, batch_size=128,
                                             show_progress_bar=len(pairs) > 20000))


class DenseIndex:
    def __init__(self, encoder: Encoder, cache_dir: Path | None = None):
        self.encoder = encoder
        self.name = f"dense:{encoder.name}"
        self.cache_dir = cache_dir

    def fit(self, ids: list[int], texts: list[str]) -> "DenseIndex":
        order = np.argsort(np.asarray(ids), kind="stable")
        self.ids = np.asarray(ids)[order]
        texts = [texts[i] for i in order]
        key = hashlib.sha256(("\x00".join(texts) + self.encoder.name).encode()).hexdigest()[:16]
        cached = (self.cache_dir / f"{re.sub(r'[^A-Za-z0-9]+', '_', self.encoder.name)}"
                  f"_{key}.npy") if self.cache_dir else None
        if cached and cached.exists():
            self.vectors = np.load(cached)
        else:
            started = time.perf_counter()
            self.vectors = self.encoder.encode(texts, "passage").astype(np.float32)
            self.encode_seconds = time.perf_counter() - started
            if cached:
                cached.parent.mkdir(parents=True, exist_ok=True)
                np.save(cached, self.vectors)
        return self

    def search(self, queries: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        q = self.encoder.encode(queries, "query").astype(np.float32)
        kept = min(k, len(self.ids))
        out_ids = np.full((len(queries), k), -1, dtype=np.int64)
        out_scores = np.full((len(queries), k), -np.inf)
        for start in range(0, len(queries), BATCH):
            block = q[start:start + BATCH] @ self.vectors.T
            order = np.argsort(-block, axis=1, kind="stable")[:, :kept]
            out_ids[start:start + BATCH, :kept] = self.ids[order]
            out_scores[start:start + BATCH, :kept] = np.take_along_axis(block, order, axis=1)
        return out_ids, out_scores


def rrf(rankings: list[np.ndarray], k: int, depth: int) -> np.ndarray:
    """Reciprocal rank fusion of several ranked id lists per query.

    Fusing ranks rather than scores avoids calibrating BM25 scores against
    cosine similarities, which live on unrelated scales.
    """
    fused = np.empty((rankings[0].shape[0], depth), dtype=np.int64)
    for row in range(fused.shape[0]):
        totals: dict[int, float] = {}
        for ranking in rankings:
            for rank, answer in enumerate(ranking[row]):
                if answer < 0:
                    continue
                totals[int(answer)] = totals.get(int(answer), 0.0) + 1.0 / (k + rank + 1)
        ordered = sorted(totals, key=lambda a: (-totals[a], a))[:depth]
        fused[row, :len(ordered)] = ordered
        fused[row, len(ordered):] = -1
    return fused


def rerank(scorer: Scorer, queries: list[str], candidates: np.ndarray,
           answers: dict[int, str], depth: int) -> tuple[np.ndarray, np.ndarray]:
    """Re-order the first `depth` candidates of each query by cross-encoder score."""
    top = candidates[:, :depth]
    valid = top >= 0
    pairs = [(queries[i], answers[int(a)]) for i in range(len(queries)) for a in top[i] if a >= 0]
    scores = np.full(top.shape, -np.inf)
    scores[valid] = scorer.score(pairs) if pairs else []
    order = np.argsort(-scores, axis=1, kind="stable")
    return np.take_along_axis(top, order, axis=1), np.take_along_axis(scores, order, axis=1)


# --- Deterministic stand-ins for tests and smoke runs ---------------------------

class HashingEncoder:
    """Bag-of-words hashed into a fixed space. Not a real model."""

    def __init__(self, name: str = "hashing-stand-in", dims: int = 256):
        self.name = name
        self.dims = dims

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        out = np.zeros((len(texts), self.dims), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                out[i, int(hashlib.md5(word.encode()).hexdigest()[:8], 16) % self.dims] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class OverlapScorer:
    """Scores a pair by shared words. Not a real model."""
    name = "overlap-stand-in"

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        words = lambda t: set(re.findall(r"[a-z]+", t.lower()))
        return np.array([len(words(q) & words(a)) for q, a in pairs], dtype=float)


@dataclass
class Timer:
    seconds: list[float]

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds.append(time.perf_counter() - self._start)
