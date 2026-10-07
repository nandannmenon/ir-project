"""Candidate generation with consumed-item filtering and champion lists."""

from __future__ import annotations

import heapq
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import ceil, sqrt

from .models import Article
from .vectorize import ItemVectors, cosine_similarity


@dataclass(frozen=True, slots=True)
class ClusterIndex:
    """Deterministic leader/follower index for approximate retrieval.

    With ``followers_per_item`` (b1) above 1, an article is listed under each
    of its b1 nearest leaders, so ``followers`` lists can overlap.
    """

    leaders: tuple[str, ...]
    followers: Mapping[str, tuple[str, ...]]
    followers_per_item: int = 1


def build_cluster_index(item_vectors: ItemVectors, leader_count: int | None = None,
                        followers_per_item: int = 2) -> ClusterIndex:
    """Choose leaders by deterministic farthest-first cosine and attach each
    item to its b1 nearest leaders.

    Start with the lexicographically first article, then repeatedly add the
    article least similar to its nearest selected leader. Sparse dot products
    use an inverted index rather than comparing every pair of article rows.
    """
    if followers_per_item < 1:
        raise ValueError("followers_per_item must be at least 1")
    article_ids = sorted(item_vectors.vectors)
    if not article_ids:
        return ClusterIndex((), {}, followers_per_item)
    count = leader_count if leader_count is not None else ceil(sqrt(len(article_ids)))
    if count < 1:
        raise ValueError("leader_count must be at least 1")
    count = min(count, len(article_ids))
    article_postings: dict[str, list[tuple[str, float]]] = defaultdict(list)
    article_norms: dict[str, float] = {}
    for article_id in article_ids:
        vector = item_vectors.vector(article_id)
        article_norms[article_id] = sqrt(sum(weight * weight for weight in vector.values()))
        for term, weight in vector.items():
            article_postings[term].append((article_id, weight))

    leaders = [article_ids[0]]
    leader_set = set(leaders)
    nearest_similarity = {article_id: 0.0 for article_id in article_ids}
    nearest_leader = {article_id: leaders[0] for article_id in article_ids}

    def update_nearest(leader: str) -> None:
        leader_vector = item_vectors.vector(leader)
        leader_norm = article_norms[leader]
        dots: dict[str, float] = defaultdict(float)
        for term, weight in leader_vector.items():
            for article_id, article_weight in article_postings.get(term, ()):
                dots[article_id] += weight * article_weight
        for article_id in article_ids:
            norm = article_norms[article_id]
            similarity = dots.get(article_id, 0.0) / (norm * leader_norm) if norm and leader_norm else 0.0
            if (similarity > nearest_similarity[article_id] or
                    (similarity == nearest_similarity[article_id] and
                     leader > nearest_leader[article_id])):
                nearest_similarity[article_id] = similarity
                nearest_leader[article_id] = leader

    update_nearest(leaders[0])
    while len(leaders) < count:
        next_leader = min(
            (article_id for article_id in article_ids if article_id not in leader_set),
            key=lambda article_id: (nearest_similarity[article_id], article_id),
        )
        leaders.append(next_leader)
        leader_set.add(next_leader)
        update_nearest(next_leader)
    leaders = tuple(sorted(leaders))
    leader_norms = [sqrt(sum(weight * weight for weight in item_vectors.vector(leader).values()))
                    for leader in leaders]
    leader_postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for position, leader in enumerate(leaders):
        for term, weight in item_vectors.vector(leader).items():
            leader_postings[term].append((position, weight))
    clusters: dict[str, list[str]] = {leader: [leader] for leader in leaders}
    for article_id in article_ids:
        if article_id in leader_set:
            continue
        vector = item_vectors.vector(article_id)
        norm = sqrt(sum(weight * weight for weight in vector.values()))
        dots: dict[int, float] = defaultdict(float)
        for term, weight in vector.items():
            for position, leader_weight in leader_postings.get(term, ()):
                dots[position] += weight * leader_weight
        similarities = {position: dot / (norm * leader_norms[position])
                        for position, dot in dots.items()
                        if dot and norm and leader_norms[position]}
        if not similarities:
            # No term in common with any leader; ties go to the highest leader ID.
            clusters[leaders[-1]].append(article_id)
            continue
        nearest = heapq.nlargest(followers_per_item, similarities.items(),
                                 key=lambda pair: (pair[1], leaders[pair[0]]))
        for position, _ in nearest:
            clusters[leaders[position]].append(article_id)
    return ClusterIndex(leaders, {
        leader: tuple(sorted(members)) for leader, members in clusters.items()
    }, followers_per_item)


def generate_cluster_candidates(
    articles: Mapping[str, Article], consumed: set[str], *,
    profile: dict[str, float], cluster_index: ClusterIndex,
    item_vectors: ItemVectors, limit: int, probe_count: int = 20,
    available_ids: set[str] | None = None,
    candidate_ids: set[str] | None = None,
) -> list[Article]:
    """Probe the b2 leaders nearest the profile, expand their followers, then
    keep the ``limit`` followers with the highest exact cosine.

    If the probed clusters hold fewer than ``limit`` eligible articles, further
    leaders are probed in rank order until the shortlist is full.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if probe_count < 1:
        raise ValueError("probe_count must be at least 1")
    eligible = set(articles) if available_ids is None else set(available_ids)
    if candidate_ids is not None:
        eligible.intersection_update(candidate_ids)
    if not profile or not cluster_index.leaders:
        return []
    ranked_leaders = sorted(
        cluster_index.leaders,
        key=lambda leader: (-cosine_similarity(profile, item_vectors.vector(leader)), leader),
    )
    selected: set[str] = set()
    for probed, leader in enumerate(ranked_leaders):
        if probed >= probe_count and len(selected) >= limit:
            break
        selected.update(article_id for article_id in cluster_index.followers.get(leader, ())
                        if article_id in eligible and article_id not in consumed)
    candidates = [articles[article_id] for article_id in selected if article_id in articles]
    candidates.sort(key=lambda article: (
        -cosine_similarity(profile, item_vectors.vector(article.article_id)), article.article_id
    ))
    return candidates[:limit]


def build_champion_lists(item_vectors: ItemVectors, limit: int = 25) -> dict[str, tuple[str, ...]]:
    champions: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for article_id, vector in item_vectors.vectors.items():
        for term, weight in vector.items():
            champions[term].append((article_id, weight))
    return {term: tuple(aid for aid, _ in sorted(rows, key=lambda x: x[1], reverse=True)[:limit])
            for term, rows in champions.items()}


def generate_candidates(
    articles: Iterable[Article] | Mapping[str, Article], consumed: set[str], *,
    profile: dict[str, float] | None = None,
    champions: dict[str, tuple[str, ...]] | None = None,
    item_vectors: ItemVectors | None = None,
    limit: int | None = None,
    interest_limit: int = 20,
    per_interest_limit: int = 50,
    available_ids: set[str] | None = None,
    candidate_ids: set[str] | None = None,
) -> list[Article]:
    article_map = articles if isinstance(articles, Mapping) else {a.article_id: a for a in articles}
    eligible = article_map.keys() if available_ids is None else available_ids
    if candidate_ids is not None:
        eligible = set(eligible) & candidate_ids
    if profile is None or champions is None:
        ids = [article_id for article_id in eligible if article_id not in consumed]
    else:
        selected: set[str] = set()
        interests = sorted(profile.items(), key=lambda row: (-row[1], row[0]))[:interest_limit]
        for term, _ in interests:
            selected.update(champions.get(term, ())[:per_interest_limit])
        ids = [article_id for article_id in selected if article_id in eligible and article_id not in consumed]
    candidates = [article_map[article_id] for article_id in ids]
    if profile and item_vectors:
        # Exact cosine is calculated only within the champion-generated shortlist.
        candidates.sort(key=lambda article: (-cosine_similarity(
            profile, item_vectors.vector(article.article_id)), article.article_id))
    else:
        candidates.sort(key=lambda article: article.article_id)
    return candidates[:limit] if limit is not None else candidates
