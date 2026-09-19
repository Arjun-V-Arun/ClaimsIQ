"""Dependency-light helpers for hybrid policy retrieval."""

from __future__ import annotations

import re


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "does", "do", "for",
    "from", "how", "if", "in", "is", "it", "may", "of", "on", "or", "should", "that", "the",
    "this", "to", "what", "when", "with", "without", "your", "policy", "policies",
    "corpus", "discuss", "question", "doesn", "i", "we", "you",
}


def normalise_token(token: str) -> str:
    token = token.lower()
    if token.endswith("ingly") and len(token) > 6:
        token = token[:-5]
    elif token.endswith("edly") and len(token) > 5:
        token = token[:-4]
    elif token.endswith("ly") and len(token) > 4:
        token = token[:-2]
    elif token.endswith("ing") and len(token) > 5:
        token = token[:-3]
    elif token.endswith("ed") and len(token) > 4:
        token = token[:-2]
        if len(token) >= 2 and token[-1] == token[-2]:
            token = token[:-1]
    elif token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith("s") and len(token) > 3:
        token = token[:-1]
    return token


def content_tokens(text: str) -> set[str]:
    raw = re.findall(r"[a-z]+", text.lower())
    return {
        normalise_token(tok)
        for tok in raw
        if tok not in STOPWORDS and len(tok) >= 3
    }

DEFAULT_SEMANTIC_DISTANCE_MAX = 1.35
DEFAULT_LEXICAL_COVERAGE_MIN = 0.50
STRICT_SEMANTIC_DISTANCE = 0.75


def accept_hybrid(
    distance: float,
    lexical_coverage: float,
    semantic_distance_max: float = DEFAULT_SEMANTIC_DISTANCE_MAX,
    lexical_coverage_min: float = DEFAULT_LEXICAL_COVERAGE_MIN,
) -> bool:
    """Accept only when semantic relevance is supported by lexical evidence."""
    return (
        lexical_coverage > lexical_coverage_min
        or (distance <= STRICT_SEMANTIC_DISTANCE and lexical_coverage >= 0.25)
    ) and distance <= semantic_distance_max
