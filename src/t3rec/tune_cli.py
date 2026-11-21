"""Choose interest-clustering and recency settings on a validation slice.

The slice is the latest part of the *training* period of the chronological
split used by ``eval_cli``; popularity and the freshness proxy come only from
impressions before it. The held-out test impressions are never read, so the
chosen settings can be fixed before the final evaluation.
"""

from __future__ import annotations

import argparse
import logging
import random
from collections import Counter
from dataclasses import replace
from math import log2
from pathlib import Path
from statistics import mean

from .entities import EntityIndex
from .evaluate import chronological_split
from .ingest import ingest_mind
from .models import User
from .pipeline import Recommender
from .scoring import ScoreWeights


def _metrics(ranked: list[str], relevant: set[str]) -> tuple[float, float]:
    hits = sum(article_id in relevant for article_id in ranked[:5])
    dcg = sum(1 / log2(position + 1) for position, article_id in enumerate(ranked[:10], 1)
              if article_id in relevant)
    ideal = sum(1 / log2(position + 1) for position in range(1, min(10, len(relevant)) + 1))
    return hits / 5, dcg / ideal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/MINDsmall_train")
    parser.add_argument("--cache-dir", default="data/processed/MINDsmall_train")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.125,
                        help="Latest fraction of the training period used for validation")
    parser.add_argument("--impressions", type=int, default=4000)
    parser.add_argument("--grid", choices=("text", "entity"), default="text",
                        help="text: recency half-life and word-only interests; "
                             "entity: entity weight and threshold for entity-aware interests")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    dataset = ingest_mind(Path(args.raw_dir), Path(args.cache_dir))
    train = chronological_split(dataset.impressions, args.test_fraction).train
    inner = chronological_split(train, args.validation_fraction)
    clicks: Counter[str] = Counter()
    first_shown, read = {}, set()
    for row in inner.train:
        clicks.update(row.clicked)
        read.update(row.history)
        for article_id in row.shown:
            if article_id not in first_shown or row.timestamp < first_shown[article_id]:
                first_shown[article_id] = row.timestamp
    log_start = min(row.timestamp for row in inner.train)
    articles = {
        article_id: replace(article, timestamp=first_shown.get(
            article_id, log_start if article_id in read else inner.cutoff))
        for article_id, article in dataset.articles.items()
    }
    entity_index = (EntityIndex.from_mind(args.raw_dir, dataset.articles)
                    if args.grid == "entity" else None)
    recommender = Recommender(articles.values(), item_vectors=dataset.item_vectors,
                              click_counts=dict(clicks), entity_index=entity_index)
    rows = []
    for row in inner.test:
        shown = {a for a in row.shown if a in recommender.available_article_ids} - set(row.history)
        relevant = set(row.clicked) & shown
        if relevant and len([a for a in row.history if a in recommender.available_article_ids]) >= 3:
            rows.append((row, shown, relevant))
    rows = random.Random(7).sample(rows, min(args.impressions, len(rows)))
    print(f"validation: {len(inner.test)} impressions after {inner.cutoff}; sampled {len(rows)}; "
          f"test period after the outer cutoff is not used")

    def run(*, multi: bool, half_life: float = 14.0,
            entity_weight: float = 0.0) -> tuple[float, float, float, float]:
        """P@5, NDCG@10, interests per user and mean category purity of the interests."""
        p5, ndcg, sizes, purity = [], [], [], []
        for row, shown, relevant in rows:
            ranked = recommender.recommend(
                User(row.user_id, row.history), limit=10, as_of=row.timestamp,
                candidate_mode="all", diverse=False, score_weights=ScoreWeights(),
                candidate_ids=shown, multi_interest=multi, recency_half_life_days=half_life,
                interest_entity_weight=entity_weight)
            if multi:
                found = recommender.last_diagnostics["interests"]
                sizes.append(len(found))
                for interest in found:
                    cats = Counter(articles[a].category for a in interest["members"])
                    purity.append(cats.most_common(1)[0][1] / interest["size"])
            precision, gain = _metrics([item.article_id for item in ranked], relevant)
            p5.append(precision)
            ndcg.append(gain)
        return (mean(p5), mean(ndcg), mean(sizes) if sizes else 1.0,
                mean(purity) if purity else 1.0)

    def show(label: str, result: tuple[float, float, float, float]) -> None:
        p5, ndcg, size, purity = result
        print(f"{label:<44}{p5:>8.4f}{ndcg:>9.4f}{size:>16.2f}{purity:>9.1%}")

    print(f"{'setting':<44}{'P@5':>8}{'NDCG@10':>9}{'interests/user':>16}{'purity':>9}")
    if args.grid == "text":
        for half_life in (14.0, 3.0, 1.0):
            show(f"single profile, recency half-life {half_life:g}d", run(multi=False, half_life=half_life))
        for max_interests in (3, 5, 8):
            for threshold in (0.05, 0.10, 0.15, 0.20, 0.30):
                recommender.interest_threshold = threshold
                recommender.max_interests = max_interests
                show(f"multi-interest tau={threshold:.2f} M={max_interests}", run(multi=True))
        return
    # Entity-aware interests: purity = share of an interest's articles in its
    # most common category, a coherence check that uses no clicks.
    show("single profile", run(multi=False))
    recommender.interest_threshold, recommender.max_interests = 0.10, 3
    show("words only, tau=0.10 M=3", run(multi=True, entity_weight=0.0))
    for entity_weight in (0.3, 0.5, 0.7):
        for threshold in (0.10, 0.20, 0.30):
            recommender.interest_threshold = threshold
            show(f"entity weight={entity_weight:.1f}, tau={threshold:.2f} M=3",
                 run(multi=True, entity_weight=entity_weight))

if __name__ == "__main__":
    main()
