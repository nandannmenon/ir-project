"""User query vectors built from previously consumed article vectors."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from math import exp, sqrt
from typing import Sequence

from .models import Article, User
from .vectorize import ItemVectors, SparseVector


@dataclass(slots=True)
class UserProfile:
    vector: SparseVector
    consumed: set[str]
    category_weights: dict[str, float]
    feature_set: set[str]


def build_user_profile(
    user: User,
    item_vectors: ItemVectors,
    articles: dict[str, Article],
    *,
    as_of: datetime | None = None,
    half_life_days: float | None = None,
    interaction_weights: Sequence[float] | None = None,
) -> UserProfile:
    """Build query q from clicks: q=normalize(sum_i w_i*d_i).

    Each d_i is the normalized TF-IDF row of a clicked article. Its default
    interaction weight is 1.0; callers may pass explicit nonnegative weights.
    If enabled and timestamps exist, multiply by exp(-ln(2)*age/half_life),
    so an interaction one half-life old contributes half as much. The final
    query is L2-normalized before cosine scoring.
    """
    if half_life_days is not None and half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    sums: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    subcategories: Counter[str] = Counter()
    history_times = user.timestamps
    for position, article_id in enumerate(user.history):
        vector = item_vectors.vector(article_id)
        timestamp = history_times[position] if position < len(history_times) else None
        weight = interaction_weights[position] if interaction_weights and position < len(interaction_weights) else 1.0
        if weight < 0:
            raise ValueError("interaction weights must be nonnegative")
        if half_life_days and as_of and timestamp:
            age = max(0.0, (as_of - timestamp).total_seconds() / 86400)
            weight *= exp(-0.6931471805599453 * age / half_life_days)
        for term, value in vector.items():
            sums[term] += weight * value
        article = articles.get(article_id)
        if article and article.category:
            categories[article.category] += weight
        if article and article.subcategory:
            subcategories[article.subcategory] += weight
    norm = sqrt(sum(value * value for value in sums.values()))
    profile = {term: value / norm for term, value in sums.items()} if norm else {}
    total = sum(categories.values())
    category_weights = {key: value / total for key, value in categories.items()} if total else {}
    top_terms = sorted(profile, key=lambda term: (-profile[term], term))[:20]
    features = {f"term:{term}" for term in top_terms}
    features.update(f"category:{value}" for value in categories)
    features.update(f"subcategory:{value}" for value in subcategories)
    return UserProfile(profile, set(user.history), category_weights, features)


@dataclass(slots=True)
class Interest:
    """One cluster of a user's reading history, used as its own query vector."""

    vector: SparseVector
    members: tuple[str, ...]
    label: str
    category: str


def build_interest_profiles(
    user: User,
    item_vectors: ItemVectors,
    articles: dict[str, Article],
    *,
    threshold: float = 0.1,
    max_interests: int = 3,
) -> list[Interest]:
    """Split the history into interests with single-pass leader clustering.

    Articles are visited in reading order. Each joins the existing interest
    whose centroid it is most similar to when that cosine is at least
    ``threshold``; otherwise it becomes the leader of a new interest. Once
    ``max_interests`` exist, an article joins its nearest interest (the
    largest one if it shares no term with any). This is the leader/follower
    idea of cluster pruning applied to one user's history, so a single
    off-topic article forms a small interest instead of reshaping one
    averaged profile.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if max_interests < 1:
        raise ValueError("max_interests must be at least 1")
    sums: list[dict[str, float]] = []
    squared_norms: list[float] = []
    members: list[list[str]] = []
    for article_id in dict.fromkeys(user.history):
        vector = item_vectors.vector(article_id)
        if not vector:
            continue
        vector_norm = sqrt(sum(value * value for value in vector.values()))
        best, best_similarity = None, 0.0
        for index, centroid in enumerate(sums):
            dot = sum(value * centroid.get(term, 0.0) for term, value in vector.items())
            similarity = dot / (vector_norm * sqrt(squared_norms[index])) if dot else 0.0
            if similarity > best_similarity:
                best, best_similarity = index, similarity
        if best is None and len(sums) >= max_interests:
            best = max(range(len(sums)), key=lambda index: (len(members[index]), -index))
        if best is not None and (best_similarity >= threshold or len(sums) >= max_interests):
            centroid = sums[best]
            for term, value in vector.items():
                old = centroid.get(term, 0.0)
                centroid[term] = old + value
                squared_norms[best] += (old + value) ** 2 - old * old
            members[best].append(article_id)
        else:
            sums.append(dict(vector))
            squared_norms.append(vector_norm * vector_norm)
            members.append([article_id])

    interests: list[Interest] = []
    for centroid, squared_norm, ids in zip(sums, squared_norms, members):
        norm = sqrt(squared_norm)
        vector = {term: value / norm for term, value in centroid.items()}
        top_terms = sorted(vector, key=lambda term: (-vector[term], term))[:3]
        categories = Counter(articles[a].category for a in ids if a in articles and articles[a].category)
        category = min(categories, key=lambda name: (-categories[name], name)) if categories else ""
        interests.append(Interest(vector, tuple(ids), " · ".join(top_terms), category))
    return interests


def proportional_quotas(sizes: Sequence[int], slots: int) -> list[int]:
    """Split ``slots`` across interests in proportion to their sizes.

    Uses the largest-remainder (Hamilton) method; ties go to the larger
    interest, then the earlier one.
    """
    total = sum(sizes)
    if slots <= 0 or total <= 0:
        return [0 for _ in sizes]
    exact = [slots * size / total for size in sizes]
    quotas = [int(value) for value in exact]
    order = sorted(range(len(sizes)), key=lambda index: (-(exact[index] - quotas[index]),
                                                          -sizes[index], index))
    for index in order[:slots - sum(quotas)]:
        quotas[index] += 1
    return quotas
