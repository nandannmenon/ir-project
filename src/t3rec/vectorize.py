"""Sparse TF-IDF document rows with inspectable DF, IDF, and weights."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log, sqrt
from typing import Iterable

from .models import Article
from .text import TextConfig, preprocess_text

SparseVector = dict[str, float]


def _normalize(vector: SparseVector) -> SparseVector:
    norm = sqrt(sum(value * value for value in vector.values()))
    return {term: value / norm for term, value in vector.items()} if norm else {}


@dataclass(slots=True)
class ItemVectors:
    """Rows are sparse term->weight maps: an inspectable row-sparse matrix."""

    vectors: dict[str, SparseVector]
    document_frequency: dict[str, int]
    idf: dict[str, float]
    vocabulary: tuple[str, ...]

    def vector(self, article_id: str) -> SparseVector:
        return self.vectors.get(article_id, {})


def build_item_vectors(
    articles: Iterable[Article], config: TextConfig | None = None
) -> ItemVectors:
    """Build log-TF * smoothed-IDF vectors and L2-normalize each article row."""
    docs = list(articles)
    tokenized = {a.article_id: preprocess_text(a.document_text, config) for a in docs}
    df: Counter[str] = Counter()
    for tokens in tokenized.values():
        df.update(set(tokens))
    n_docs = len(docs)
    idf = {term: log((1 + n_docs) / (1 + count)) + 1 for term, count in df.items()}
    vectors: dict[str, SparseVector] = {}
    for article_id, tokens in tokenized.items():
        tf = Counter(tokens)
        vectors[article_id] = _normalize(
            {term: (1 + log(count)) * idf[term] for term, count in tf.items()}
        )
    return ItemVectors(vectors, dict(df), idf, tuple(sorted(df)))


def cosine_similarity(left: SparseVector, right: SparseVector) -> float:
    """Cosine for sparse vectors; works for normalized or unnormalized rows."""
    if len(left) > len(right):
        left, right = right, left
    dot = sum(value * right.get(term, 0.0) for term, value in left.items())
    if not dot:
        return 0.0
    left_norm = sqrt(sum(v * v for v in left.values()))
    right_norm = sqrt(sum(v * v for v in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0
