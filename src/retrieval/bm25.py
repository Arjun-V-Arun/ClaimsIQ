"""BM25 over a sparse matrix, so all test questions are scored in one product.

A keyword baseline is the bar every learned retriever has to clear, and it is
the only retriever here that runs anywhere without a model download, which is
why CI uses it as the regression check.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from src.rag.hybrid import normalise_token

_WORD = re.compile(r"[a-z0-9]+")
BATCH = 256


def analyzer(stem: bool):
    def analyze(text: str) -> list[str]:
        words = [w for w in _WORD.findall(text.lower())
                 if w not in ENGLISH_STOP_WORDS and len(w) > 1]
        return [normalise_token(w) for w in words] if stem else words
    return analyze


@dataclass(frozen=True)
class Params:
    k1: float = 1.2
    b: float = 0.75
    stem: bool = True


class BM25:
    name = "bm25"

    def __init__(self, params: Params = Params()):
        self.params = params

    def fit(self, ids: list[int], texts: list[str]) -> "BM25":
        # Sorting by id makes "ties go to the lower id" true for any input order.
        order = np.argsort(np.asarray(ids), kind="stable")
        self.ids = np.asarray(ids)[order]
        texts = [texts[i] for i in order]
        self.vectorizer = CountVectorizer(analyzer=analyzer(self.params.stem))
        self._tf = self.vectorizer.fit_transform(texts).astype(np.float64).tocsr()
        self.reweight(self.params.k1, self.params.b)
        return self

    def reweight(self, k1: float, b: float) -> None:
        """Change k1 and b without re-tokenizing the corpus."""
        self.params = Params(k1, b, self.params.stem)
        tf = self._tf.copy()
        documents = tf.shape[0]
        frequency = np.bincount(tf.indices, minlength=tf.shape[1])
        # Lucene's idf, which stays positive for very common terms.
        idf = np.log(1 + (documents - frequency + 0.5) / (frequency + 0.5))
        lengths = np.asarray(tf.sum(axis=1)).ravel()
        rows = np.repeat(np.arange(documents), np.diff(tf.indptr))
        norm = k1 * (1 - b + b * lengths[rows] / lengths.mean())
        tf.data = idf[tf.indices] * tf.data * (k1 + 1) / (tf.data + norm)
        self._weights = tf.T.tocsr()
        self.idf = idf

    def scores(self, queries: list[str]) -> np.ndarray:
        """Dense (queries x corpus) scores. Each query term counts once."""
        q = self.vectorizer.transform(queries)
        q.data[:] = 1.0
        return (q @ self._weights).toarray()

    def search(self, queries: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-k answer ids and scores per query. Ties go to the lower answer id.

        A corpus smaller than k pads with id -1, which never matches an answer.
        """
        kept = min(k, len(self.ids))
        out_ids = np.full((len(queries), k), -1, dtype=np.int64)
        out_scores = np.full((len(queries), k), -np.inf)
        for start in range(0, len(queries), BATCH):
            block = self.scores(queries[start:start + BATCH])
            order = np.argsort(-block, axis=1, kind="stable")[:, :kept]
            out_ids[start:start + BATCH, :kept] = self.ids[order]
            out_scores[start:start + BATCH, :kept] = np.take_along_axis(block, order, axis=1)
        return out_ids, out_scores

    def query_weight(self, queries: list[str]) -> np.ndarray:
        """Total idf of each query's known terms, used to make scores comparable."""
        q = self.vectorizer.transform(queries)
        q.data[:] = 1.0
        return np.asarray(q @ self.idf).ravel()


def matrix_fingerprint(model: BM25) -> tuple[int, int]:
    weights: sparse.csr_matrix = model._weights
    return weights.shape[1], weights.nnz
