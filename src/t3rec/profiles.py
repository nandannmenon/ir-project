"""User query vectors built from previously consumed article vectors."""

from __future__ import annotations

from array import array
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from math import exp, sqrt
from typing import Sequence

from .entities import EntityIndex, EntityVector, dot
from .models import Article, User
from .vectorize import ItemVectors, SparseVector, cosine_similarity


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
    # Normalized sum of the members' centred entity vectors, when known.
    entity_vector: EntityVector | None = None


def blend_similarity(text: float, entity: float | None, entity_weight: float) -> float:
    """(1 - w) * text cosine + w * entity cosine; text alone when either side has no entities.

    Negative centred entity cosines (unrelated entities) are clipped to 0 so
    that missing evidence never outweighs shared words.
    """
    if entity is None or entity_weight <= 0:
        return text
    return (1.0 - entity_weight) * text + entity_weight * max(0.0, entity)


def interest_similarity(interest: Interest, item_vector: SparseVector,
                        item_entity: EntityVector | None, entity_weight: float = 0.0) -> float:
    """Similarity of one article to one interest: word cosine blended with entity cosine."""
    text = cosine_similarity(interest.vector, item_vector)
    entity = (dot(interest.entity_vector, item_entity)
              if interest.entity_vector is not None and item_entity is not None else None)
    return blend_similarity(text, entity, entity_weight)


def build_interest_profiles(
    user: User,
    item_vectors: ItemVectors,
    articles: dict[str, Article],
    *,
    threshold: float = 0.1,
    max_interests: int = 3,
    entity_index: EntityIndex | None = None,
    entity_weight: float = 0.0,
) -> list[Interest]:
    """Split the history into interests with single-pass leader clustering.

    Articles are visited in reading order. Each joins the existing interest
    it is most similar to when that similarity is at least ``threshold``;
    otherwise it becomes the leader of a new interest. Once ``max_interests``
    exist, an article joins its nearest interest (the largest one if it is
    similar to none). This is the leader/follower idea of cluster pruning
    applied to one user's history, so a single off-topic article forms a small
    interest instead of reshaping one averaged profile.

    Similarity is the TF-IDF cosine to the interest's centroid, blended with
    the cosine of their centred entity vectors when ``entity_index`` is given
    and ``entity_weight`` > 0: two stories about Gordon Hayward and the Boston
    Celtics can then join even when their headlines share no word. With an
    entity index, labels are the interest's most frequent entity names.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if max_interests < 1:
        raise ValueError("max_interests must be at least 1")
    if not 0.0 <= entity_weight <= 1.0:
        raise ValueError("entity_weight must be between 0 and 1")
    use_entities = entity_index is not None and entity_weight > 0
    sums: list[dict[str, float]] = []
    squared_norms: list[float] = []
    entity_sums: list[list[float] | None] = []
    members: list[list[str]] = []
    for article_id in dict.fromkeys(user.history):
        vector = item_vectors.vector(article_id)
        if not vector:
            continue
        vector_norm = sqrt(sum(value * value for value in vector.values()))
        entity = entity_index.vector(article_id) if use_entities else None
        best, best_similarity = None, 0.0
        for index, centroid in enumerate(sums):
            dot_text = sum(value * centroid.get(term, 0.0) for term, value in vector.items())
            text = dot_text / (vector_norm * sqrt(squared_norms[index])) if dot_text else 0.0
            group_entity = entity_sums[index]
            entity_cosine = None
            if entity is not None and group_entity is not None:
                group_norm = sqrt(sum(value * value for value in group_entity))
                if group_norm:
                    entity_cosine = sum(a * b for a, b in zip(entity, group_entity)) / group_norm
            similarity = blend_similarity(text, entity_cosine, entity_weight)
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
            if entity is not None:
                if entity_sums[best] is None:
                    entity_sums[best] = list(entity)
                else:
                    entity_sums[best] = [a + b for a, b in zip(entity_sums[best], entity)]
            members[best].append(article_id)
        else:
            sums.append(dict(vector))
            squared_norms.append(vector_norm * vector_norm)
            entity_sums.append(list(entity) if entity is not None else None)
            members.append([article_id])

    interests: list[Interest] = []
    for centroid, squared_norm, group_entity, ids in zip(sums, squared_norms, entity_sums, members):
        norm = sqrt(squared_norm)
        vector = {term: value / norm for term, value in centroid.items()}
        top_terms = sorted(vector, key=lambda term: (-vector[term], term))[:3]
        categories = Counter(articles[a].category for a in ids if a in articles and articles[a].category)
        category = min(categories, key=lambda name: (-categories[name], name)) if categories else ""
        label = _entity_label(ids, entity_index) if entity_index is not None else ""
        entity_vector = None
        if group_entity is not None:
            entity_norm = sqrt(sum(value * value for value in group_entity))
            if entity_norm:
                entity_vector = array("d", (value / entity_norm for value in group_entity))
        interests.append(Interest(vector, tuple(ids), label or " · ".join(top_terms), category,
                                  entity_vector))
    return interests


def _entity_label(members: Sequence[str], entity_index: EntityIndex) -> str:
    """The entity names mentioned by the most members (then by total weight).

    A second name is added only when at least two members mention it (or the
    interest is a single article), so one article's side entity does not
    become half of a three-article interest's label.
    """
    weight: Counter[str] = Counter()
    mentions: Counter[str] = Counter()
    for article_id in members:
        for label, value in entity_index.labels.get(article_id, ()):
            weight[label] += value
            mentions[label] += 1
    ranked = sorted(weight, key=lambda label: (-mentions[label], -weight[label], label))
    chosen = ranked[:1] + [label for label in ranked[1:2] if mentions[label] >= min(2, len(members))]
    return " · ".join(chosen)


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
