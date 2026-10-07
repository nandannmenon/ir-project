"""Run inspectable personalized recommendations against a JSON fixture."""

from __future__ import annotations

import argparse
from datetime import datetime

from .data import load_json_fixture
from .models import User
from .pipeline import Recommender
from .scoring import ScoreWeights


def _show(recommender: Recommender, user: User, *, top_k: int,
          weights: ScoreWeights, label: str) -> list[str]:
    results = recommender.recommend(user, limit=top_k, as_of=datetime(2024, 1, 1),
                                    score_weights=weights)
    mode = recommender.last_diagnostics["recommendation_mode"]
    print(f"\n{label} | user={user.user_id} | history={list(user.history)} | mode={mode}")
    print(f"profile terms: {sorted(recommender.last_diagnostics['profile_vector'].items(), key=lambda x: -x[1])[:8]}")
    print(f"candidate generation: {recommender.last_diagnostics['candidate_mode']} "
          f"({recommender.last_diagnostics['candidate_count']} candidates; "
          f"weights={recommender.last_diagnostics['score_weights']})")
    print(f"candidate IDs: {recommender.last_diagnostics['candidate_ids']}")
    for item in results:
        print(f"{item.rank}. {item.article_id} | {item.title} | category={item.category}")
        print(f"   cosine={item.cosine_score:.3f} jaccard={item.jaccard_score:.3f} "
              f"recency={item.recency_score:.3f} popularity={item.popularity_score:.3f} "
              f"net={item.net_score:.3f} final={item.final_score:.3f}")
        print(f"   diversity: original_rank={item.original_rank} reranked_rank={item.reranked_rank} "
              f"redundancy={item.redundancy_score:.3f} penalty={item.redundancy_penalty:.3f} "
              f"adjusted={item.diversity_adjusted_score:.3f}")
        print(f"   top TF-IDF contributions: {item.term_contributions}")
        print(f"   {item.explanation}")
    return [item.article_id for item in results]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default="data/synthetic.json")
    parser.add_argument("--user", default="U1")
    parser.add_argument("--all-users", action="store_true", help="Run every fixture user")
    parser.add_argument("--cold-start", action="store_true", help="Also run a no-history user")
    parser.add_argument("--compare-weights", action="store_true",
                        help="Compare primary-cosine and popularity-heavy score weights")
    parser.add_argument("--compare-diversity", action="store_true",
                        help="Compare net-score top K with diversity-aware top K")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    articles, users = load_json_fixture(args.fixture)
    if args.all_users:
        selected = users
    else:
        selected = [user for user in users if user.user_id == args.user]
        if not selected:
            parser.error(f"unknown user {args.user!r}")
    if args.cold_start:
        selected = [*selected, User("COLD_START", ())]
    recommender = Recommender(articles, click_counts={"A3": 7, "A4": 3, "A5": 2, "A6": 5, "A8": 2})
    primary = ScoreWeights()
    popularity_heavy = ScoreWeights(alpha=0.25, beta=0.05, gamma=0.05, delta=0.65)
    for user in selected:
        primary_order = _show(recommender, user, top_k=args.top_k,
                              weights=primary, label="cosine-primary")
        if args.compare_weights:
            altered_order = _show(recommender, user, top_k=args.top_k,
                                  weights=popularity_heavy, label="popularity-heavy")
            print(f"weight change altered ordering: {primary_order != altered_order}")
        if args.compare_diversity:
            comparison = recommender.compare_diversity(
                user, top_n=max(args.top_k * 5, args.top_k), top_k=args.top_k,
                as_of=datetime(2024, 1, 1), relevance_weight=0.75,
            )
            print("\nDiversity comparison (same top-N candidate pool)")
            for label, rows, metrics in (
                ("without", comparison.without_diversity, comparison.baseline_metrics),
                ("with", comparison.with_diversity, comparison.diversified_metrics),
            ):
                print(f"{label}: {[row.article_id for row in rows]}")
                print(f"  unique categories={metrics.unique_category_count}, "
                      f"coverage={metrics.category_coverage:.3f}, "
                      f"average pairwise cosine={metrics.average_pairwise_similarity:.3f}")
                for row in rows:
                    print(f"  {row.article_id}: original={row.original_rank}, "
                          f"reranked={row.reranked_rank}, relevance={row.relevance_score:.3f}, "
                          f"redundancy={row.redundancy_score:.3f}, "
                          f"penalty={row.redundancy_penalty:.3f}, "
                          f"adjusted={row.diversity_adjusted_score:.3f}")


if __name__ == "__main__":
    main()
