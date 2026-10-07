"""Modular relevance, set similarity, quality signals, and score fusion."""

from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from math import exp, log1p
from typing import Mapping

from .models import Article
from .profiles import UserProfile
from .vectorize import ItemVectors, SparseVector, cosine_similarity


@dataclass(frozen=True, slots=True)
class ScoreWeights:
    """Transparent fusion weights; cosine remains primary, other signals secondary."""

    alpha: float = 0.75  # lexical profile relevance
    beta: float = 0.10   # categorical + term-set overlap
    gamma: float = 0.05 # article timestamp recency
    delta: float = 0.10 # historical click popularity

    def __post_init__(self) -> None:
        values = (self.alpha, self.beta, self.gamma, self.delta)
        if any(value < 0 for value in values) or sum(values) <= 0:
            raise ValueError("score weights must be nonnegative and at least one must be positive")

    def as_mapping(self) -> dict[str, float]:
        return {"relevance": self.alpha, "jaccard": self.beta,
                "recency": self.gamma, "popularity": self.delta}


def cosine_relevance(profile: SparseVector, item: SparseVector) -> float:
    return cosine_similarity(profile, item)


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def recency_score(article: Article, as_of: datetime, half_life_days: float = 14) -> float:
    """Exponential freshness in [0,1]; one half-life old is 0.5."""
    if not article.timestamp or half_life_days <= 0:
        return 0.0
    age = max(0.0, (as_of - article.timestamp).total_seconds() / 86400)
    return exp(-0.6931471805599453 * age / half_life_days)


def popularity_score(article_id: str, click_counts: Mapping[str, int], *,
                     max_log_clicks: float | None = None) -> float:
    """Log-compressed click count divided by corpus maximum, yielding [0,1]."""
    if not click_counts:
        return 0.0
    max_log = (max((log1p(max(0, n)) for n in click_counts.values()), default=0.0)
               if max_log_clicks is None else max_log_clicks)
    value = log1p(max(0, click_counts.get(article_id, 0)))
    return value / max_log if max_log else 0.0


def calculate_net_score(
    relevance: float,
    *,
    jaccard: float = 0.0,
    recency: float = 0.0,
    popularity: float = 0.0,
    weights: Mapping[str, float] | ScoreWeights | None = None,
) -> float:
    """Normalize each bounded component to [0,1], then compute a weighted mean.

    The default allocates 75% to cosine and only 25% to secondary signals;
    weights are configuration, not fitted parameters, and are shown in results.
    """
    if isinstance(weights, ScoreWeights):
        w = weights.as_mapping()
    else:
        w = weights or ScoreWeights().as_mapping()
    values = {"relevance": relevance, "jaccard": jaccard,
              "recency": recency, "popularity": popularity}
    total_weight = sum(max(0.0, w.get(key, 0.0)) for key in values)
    if not total_weight:
        return 0.0
    return sum(max(0.0, w.get(key, 0.0)) * values[key] for key in values) / total_weight


def term_contributions(profile: SparseVector, item: SparseVector) -> tuple[tuple[str, float], ...]:
    contributions = [(term, profile[term] * item[term]) for term in profile.keys() & item.keys()]
    return tuple(sorted(contributions, key=lambda pair: (-pair[1], pair[0])))


def category_match_score(profile: UserProfile, article: Article) -> float:
    return profile.category_weights.get(article.category, 0.0)


def article_feature_set(article: Article, item: SparseVector) -> set[str]:
    """Typed set for Jaccard: title/abstract terms plus separate category zones."""
    features = {f"term:{term}" for term in item}
    if article.category:
        features.add(f"category:{article.category}")
    if article.subcategory:
        features.add(f"subcategory:{article.subcategory}")
    return features
