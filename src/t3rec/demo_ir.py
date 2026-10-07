"""Terminal walkthrough of the recommender's inspectable IR stages."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from datetime import datetime

from .candidates import generate_candidates
from .ingest import ProcessedDataset, ingest_mind
from .models import User
from .pipeline import Recommender
from .scoring import ScoreWeights


def _component_contributions(recommendation: object) -> list[tuple[str, float]]:
    # Cosine and Jaccard enter the net score after max-normalization.
    values = {
        "cosine": recommendation.relevance_normalized,
        "jaccard": recommendation.jaccard_normalized,
        "recency": recommendation.recency_score,
        "popularity": recommendation.popularity_score,
    }
    weights = recommendation.score_weights
    total = sum(weights.values())
    return [(name, weights.get("relevance" if name == "cosine" else name, 0.0) * value / total)
            for name, value in values.items()] if total else [(name, 0.0) for name in values]


def run_demo(dataset: ProcessedDataset, user_id: str, *, k: int = 5,
             show_candidates: int = 8, inspect_term: str | None = None,
             multi_interest: bool = False) -> None:
    """Print a complete inspectable recommendation trace for a dataset user."""
    if k < 1 or show_candidates < 1:
        raise ValueError("k and show_candidates must be positive")
    user_impressions = [row for row in dataset.impressions if row.user_id == user_id]
    if not user_impressions:
        known = sorted(dataset.users)
        raise ValueError(f"Unknown user {user_id!r}. Available users include: {known[:20]}")
    latest = max(user_impressions,
                 key=lambda row: (row.timestamp is not None,
                                  row.timestamp or datetime.min, row.impression_id))
    user = User(user_id, latest.history)
    recommender = Recommender.from_processed(dataset)
    profile = recommender.profile(user, as_of=latest.timestamp)
    results = recommender.recommend(user, limit=show_candidates, as_of=latest.timestamp,
                                    candidate_mode="champions", diverse=False,
                                    multi_interest=multi_interest)
    diagnostics = recommender.last_diagnostics
    profile_vector = profile.vector
    profile_terms = sorted(profile_vector.items(), key=lambda pair: (-pair[1], pair[0]))

    if profile_vector:
        raw_candidates = generate_candidates(
            recommender.article_by_id, set(), profile=profile_vector,
            champions=recommender.champions, item_vectors=recommender.item_vectors,
            limit=None, interest_limit=recommender.candidate_interest_limit,
            per_interest_limit=recommender.champion_per_interest_limit,
        )
    else:
        raw_candidates = list(recommender.articles)
    filtered_candidates = [article for article in raw_candidates
                           if article.article_id not in profile.consumed
                           and article.article_id in recommender.available_article_ids]

    print("=" * 78)
    print("T3 IR RECOMMENDER INSPECTION")
    print("=" * 78)
    print(f"User: {user_id} | history interactions: {len(user.history)} | "
          f"mode: {diagnostics['recommendation_mode']} | profile source: latest impression's pre-impression history")

    print("\n1. USER PROFILE (normalized sparse TF-IDF query)")
    print(f"Historical article IDs: {', '.join(user.history) if user.history else '(none)'}")
    if profile_terms:
        for term, weight in profile_terms[:12]:
            print(f"  {term:<24} TF-IDF={weight:.6f}")
    else:
        print("  (empty profile; the recommender is in cold-start mode)")
    if multi_interest and diagnostics["interests"]:
        print(f"\n1b. INTERESTS (leader clustering of the history, threshold "
              f"{recommender.interest_threshold}, at most {recommender.max_interests})")
        quotas = diagnostics["interest_quotas"] or [show_candidates]
        for interest, quota in zip(diagnostics["interests"], quotas):
            print(f"  '{interest['label']}' [{interest['category'] or '-'}]: "
                  f"{interest['size']} read articles -> {quota} of {show_candidates} slots; "
                  f"members {', '.join(interest['members'][:6])}"
                  f"{' ...' if interest['size'] > 6 else ''}")

    print("\n2. CANDIDATE GENERATION")
    print(f"  Before consumed-item filtering: {len(raw_candidates)} champion candidates")
    print(f"  After validity/availability/history filtering: {len(filtered_candidates)}")
    print(f"  Candidate method: {diagnostics['candidate_mode']} | scoring candidates: {diagnostics['candidate_count']}")
    print("  Top candidates (cosine-ordered shortlist): " +
          (", ".join(diagnostics["candidate_ids"][:min(show_candidates, 12)]) or "(none)"))
    copies = diagnostics["history_copies_removed"]
    print(f"  Copies of already-read articles skipped: {len(copies)}" +
          (" (" + ", ".join(f"{cand}~{read}" for cand, read in copies[:6]) + ")" if copies else ""))
    maxima = diagnostics["normalization_max"]
    print(f"  Score normalization: cosine / {maxima['relevance']:.4f}, Jaccard / {maxima['jaccard']:.4f} "
          f"(candidate-set maxima); recency "
          f"{'active' if diagnostics['recency_available'] else 'unavailable (no article timestamps), weight set to 0'}")

    print("\n3. ARTICLE SCORING (highest net-score candidates before diversity)")
    if not results:
        print("  No eligible candidates were generated.")
    for item in results:
        print(f"  {item.article_id} | {item.title}" +
              (f"  [interest: {item.interest}]" if item.interest else ""))
        print(f"    cosine={item.cosine_score:.4f} (scaled {item.relevance_normalized:.3f})  "
              f"Jaccard={item.jaccard_score:.4f} (scaled {item.jaccard_normalized:.3f})  "
              f"recency={item.recency_score:.4f}  popularity={item.popularity_score:.4f}  "
              f"net={item.net_score:.4f}")
        parts = ", ".join(f"{name}={contribution:.4f}"
                           for name, contribution in _component_contributions(item))
        print(f"    weighted net-score contributions: {parts}")

    print("\n4. FINAL RANKING AND EXPLANATIONS")
    comparison = recommender.compare_diversity(
        user, top_n=max(k * 5, show_candidates), top_k=k,
        as_of=latest.timestamp, candidate_mode="champions",
        relevance_weight=0.75, multi_interest=multi_interest,
    )
    for item in comparison.with_diversity:
        print(f"  {item.reranked_rank}. {item.article_id} | {item.title} | "
              f"category={item.category or '(missing)'} | final={item.final_score:.4f}")
        terms = ", ".join(f"{term} ({value:.4f})" for term, value in item.term_contributions[:6])
        print(f"    matching TF-IDF contributions: {terms or '(no shared terms)'}")
        parts = ", ".join(f"{name}={contribution:.4f}"
                           for name, contribution in _component_contributions(item))
        print(f"    score contributions: {parts}; net={item.net_score:.4f}; "
              f"redundancy penalty={item.redundancy_penalty:.4f}")
        print(f"    Explanation: {item.explanation}")

    print("\n5. DIVERSITY COMPARISON (same top-N net-score candidate pool)")
    print(f"  Original ranking: {[item.article_id for item in comparison.without_diversity]}")
    print(f"  Reranked ranking: {[item.article_id for item in comparison.with_diversity]}")
    for label, metrics in (("original", comparison.baseline_metrics),
                           ("reranked", comparison.diversified_metrics)):
        print(f"  {label}: unique categories={metrics.unique_category_count}, "
              f"category coverage={metrics.category_coverage:.3f}, "
              f"mean pairwise cosine={metrics.average_pairwise_similarity:.3f}")
    print(f"  Category coverage change: "
          f"{comparison.baseline_metrics.category_coverage:+.3f} -> "
          f"{comparison.diversified_metrics.category_coverage:+.3f} "
          f"(delta {comparison.diversified_metrics.category_coverage-comparison.baseline_metrics.category_coverage:+.3f})")
    for item in comparison.with_diversity:
        print(f"  {item.article_id}: original rank={item.original_rank}, "
              f"reranked rank={item.reranked_rank}, relevance={item.relevance_score:.4f}, "
              f"redundancy={item.redundancy_score:.4f}, penalty={item.redundancy_penalty:.4f}, "
              f"adjusted={item.diversity_adjusted_score:.4f}")

    print("\n6. OPTIONAL SPARSE TF-IDF INSPECTION")
    index = recommender.item_vectors
    term = inspect_term or (profile_terms[0][0] if profile_terms else
                            (index.vocabulary[0] if index.vocabulary else ""))
    postings = [(article_id, vector[term]) for article_id, vector in index.vectors.items()
                if term and term in vector]
    postings.sort(key=lambda pair: (-pair[1], pair[0]))
    print(f"  Vocabulary size: {len(index.vocabulary)}")
    print(f"  Term: {term or '(no vocabulary)'} | DF={index.document_frequency.get(term, 0)} | "
          f"IDF={index.idf.get(term, 0.0):.6f} | posting count={len(postings)}")
    print("  Example postings / normalized TF-IDF weights: " +
          (", ".join(f"{article_id}:{weight:.4f}" for article_id, weight in postings[:8])
           or "(none)"))
    print("  Sparse representation: article ID -> {term: normalized TF-IDF weight}; "
          "cosine is the sparse-vector dot product for these L2-normalized rows.")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True, help="MIND user ID to inspect")
    parser.add_argument("--k", type=int, default=5, help="Number of final recommendations")
    parser.add_argument("--show-candidates", type=int, default=8,
                        help="Number of net-scored candidate rows to print")
    parser.add_argument("--raw-dir", default="data/sample_mind",
                        help="MIND TSV folder; defaults to the included sample dataset")
    parser.add_argument("--cache-dir", default="data/processed/demo_ir",
                        help="Processed ingestion cache folder")
    parser.add_argument("--term", help="Optional vocabulary term for postings inspection")
    parser.add_argument("--multi-interest", action="store_true",
                        help="Split the history into interests and share slots across them")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    dataset = ingest_mind(args.raw_dir, args.cache_dir)
    try:
        run_demo(dataset, args.user, k=args.k, show_candidates=args.show_candidates,
                 inspect_term=args.term, multi_interest=args.multi_interest)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
