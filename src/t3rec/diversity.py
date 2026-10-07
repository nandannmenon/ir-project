"""Greedy, inspectable diversity reranking and top-K coverage measurements."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from .models import Recommendation
from .vectorize import ItemVectors, cosine_similarity


@dataclass(frozen=True, slots=True)
class DiversityMetrics:
    item_count: int
    unique_category_count: int
    category_coverage: float
    average_pairwise_similarity: float


@dataclass(frozen=True, slots=True)
class DiversityComparison:
    without_diversity: tuple[Recommendation, ...]
    with_diversity: tuple[Recommendation, ...]
    baseline_metrics: DiversityMetrics
    diversified_metrics: DiversityMetrics
    relevance_weight: float


def diversity_metrics(
    recommendations: Sequence[Recommendation], item_vectors: ItemVectors,
) -> DiversityMetrics:
    """Measure category coverage and mean pairwise sparse-vector cosine."""
    categories = {row.article.category for row in recommendations if row.article.category}
    pairs: list[float] = []
    for left_index, left in enumerate(recommendations):
        for right in recommendations[left_index + 1:]:
            pairs.append(cosine_similarity(item_vectors.vector(left.article_id),
                                           item_vectors.vector(right.article_id)))
    count = len(recommendations)
    return DiversityMetrics(
        item_count=count,
        unique_category_count=len(categories),
        category_coverage=len(categories) / count if count else 0.0,
        average_pairwise_similarity=sum(pairs) / len(pairs) if pairs else 0.0,
    )


def diversity_rerank(
    recommendations: Sequence[Recommendation], item_vectors: ItemVectors,
    *, limit: int = 10, relevance_weight: float = 0.75,
    category_only: bool = False,
    diversity_weight: float | None = None,
) -> list[Recommendation]:
    """Greedily maximize λ*net_score - (1-λ)*redundancy for each next item.

    Redundancy is the maximum cosine similarity to any already selected item.
    With ``category_only=True`` it is 1 for a repeated nonempty category and 0
    otherwise (used by content-independent cold start). ``diversity_weight`` is
    retained as a compatibility alias for the old API and means ``1-λ``.
    The redundancy penalty stored on each result is (1-λ)*redundancy.
    """
    if diversity_weight is not None:
        relevance_weight = 1.0 - diversity_weight
    if not 0.0 <= relevance_weight <= 1.0:
        raise ValueError("relevance_weight must be between 0 and 1")
    if limit < 0:
        raise ValueError("limit cannot be negative")
    remaining = list(recommendations)
    selected: list[Recommendation] = []
    source_ranks = {
        id(row): row.original_rank if row.original_rank > 0 else index
        for index, row in enumerate(remaining, start=1)
    }
    while remaining and len(selected) < limit:
        def parts(candidate: Recommendation) -> tuple[float, float, float]:
            if category_only:
                redundancy = max((1.0 if candidate.article.category and
                                  candidate.article.category == chosen.article.category else 0.0
                                  for chosen in selected), default=0.0)
            else:
                redundancy = max((cosine_similarity(item_vectors.vector(candidate.article_id),
                                                    item_vectors.vector(chosen.article_id))
                                  for chosen in selected), default=0.0)
            penalty = (1.0 - relevance_weight) * redundancy
            adjusted = relevance_weight * candidate.net_score - penalty
            return redundancy, penalty, adjusted

        choice = max(remaining, key=lambda row: (parts(row)[2], -source_ranks[id(row)]))
        remaining.remove(choice)
        redundancy, penalty, adjusted = parts(choice)
        rerank_position = len(selected) + 1
        message = (f" Diversity reranking: original rank {source_ranks[id(choice)]}, "
                   f"redundancy {redundancy:.3f}, penalty {penalty:.3f}, "
                   f"adjusted score {adjusted:.3f}.")
        selected.append(replace(
            choice,
            rank=rerank_position,
            reranked_rank=rerank_position,
            original_rank=source_ranks[id(choice)],
            redundancy_score=redundancy,
            redundancy_penalty=penalty,
            final_score=adjusted,
            explanation=choice.explanation + message,
        ))
    return selected


def compare_diversity(
    top_n_candidates: Sequence[Recommendation], item_vectors: ItemVectors,
    *, top_k: int = 5, relevance_weight: float = 0.75,
    category_only: bool = False,
) -> DiversityComparison:
    """Compare ordinary net-score top K with greedy diversity reranking."""
    if top_k < 0:
        raise ValueError("top_k cannot be negative")
    baseline = tuple(
        replace(row, rank=index, reranked_rank=index,
                original_rank=row.original_rank if row.original_rank > 0 else index,
                redundancy_score=0.0, redundancy_penalty=0.0,
                final_score=row.net_score)
        for index, row in enumerate(top_n_candidates[:top_k], start=1)
    )
    reranked = tuple(diversity_rerank(top_n_candidates, item_vectors, limit=top_k,
                                      relevance_weight=relevance_weight,
                                      category_only=category_only))
    return DiversityComparison(
        without_diversity=baseline,
        with_diversity=reranked,
        baseline_metrics=diversity_metrics(baseline, item_vectors),
        diversified_metrics=diversity_metrics(reranked, item_vectors),
        relevance_weight=relevance_weight,
    )
