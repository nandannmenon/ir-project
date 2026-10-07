"""Small data models; raw dataset rows are not retained by the recommender."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping


@dataclass(frozen=True, slots=True)
class Article:
    article_id: str
    title: str
    abstract: str = ""
    category: str = ""
    subcategory: str = ""
    timestamp: datetime | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    @property
    def document_text(self) -> str:
        return f"{self.title or ''} {self.abstract or ''}".strip()


@dataclass(frozen=True, slots=True)
class User:
    user_id: str
    history: tuple[str, ...]
    timestamps: tuple[datetime | None, ...] = ()


@dataclass(frozen=True, slots=True)
class Recommendation:
    article: Article
    relevance: float
    jaccard: float
    category_match: float
    recency: float
    popularity: float
    net_score: float
    final_score: float
    explanation: str
    term_contributions: tuple[tuple[str, float], ...] = ()
    rank: int = 0
    candidate_count: int = 0
    score_weights: Mapping[str, float] = field(default_factory=dict)
    mode: str = "personalized"
    original_rank: int = 0
    reranked_rank: int = 0
    redundancy_score: float = 0.0
    redundancy_penalty: float = 0.0
    # Cosine and Jaccard divided by their maximum over the scored candidate set;
    # these, not the raw values, enter ``net_score``.
    relevance_normalized: float = 0.0
    jaccard_normalized: float = 0.0
    # Label of the matching interest when multi-interest profiles are used.
    interest: str = ""

    @property
    def article_id(self) -> str:
        return self.article.article_id

    @property
    def title(self) -> str:
        return self.article.title

    @property
    def category(self) -> str:
        return self.article.category

    @property
    def cosine_score(self) -> float:
        return self.relevance

    @property
    def jaccard_score(self) -> float:
        return self.jaccard

    @property
    def recency_score(self) -> float:
        return self.recency

    @property
    def popularity_score(self) -> float:
        return self.popularity

    @property
    def relevance_score(self) -> float:
        """Net score entering the diversity stage."""
        return self.net_score

    @property
    def diversity_adjusted_score(self) -> float:
        return self.final_score

