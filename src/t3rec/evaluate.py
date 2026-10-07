"""Chronological offline evaluation for the MIND impression recommender.

The global cutoff separates train-only popularity/availability signals from
held-out impressions. Each held-out impression supplies its own pre-impression
history and displayed candidate set; its clicked labels are used only as truth.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Iterable
from xml.sax.saxutils import escape

from .ingest import Impression, ProcessedDataset
from .models import Article, User
from .pipeline import Recommender
from .scoring import ScoreWeights, calculate_net_score
from .diversity import diversity_rerank
from .vectorize import ItemVectors, cosine_similarity

LOGGER = logging.getLogger("t3rec.evaluate")
MODEL_CONFIGS: dict[str, dict[str, object]] = {
    "A. Popularity baseline": {
        "description": "Rank displayed eligible articles by training-period click count.",
        "weights": None, "diversity": False,
    },
    "B. TF-IDF cosine": {
        "description": "User-profile/article cosine only.",
        "weights": {"relevance": 1.0, "jaccard": 0.0, "recency": 0.0, "popularity": 0.0},
        "diversity": False,
    },
    "C. TF-IDF + Jaccard": {
        "description": "Cosine-primary lexical and set similarity ablation.",
        "weights": {"relevance": 0.90, "jaccard": 0.10, "recency": 0.0, "popularity": 0.0},
        "diversity": False,
    },
    "D. TF-IDF + Jaccard + recency/popularity": {
        "description": "Full weighted net score before diversity reranking.",
        "weights": {"relevance": 0.75, "jaccard": 0.10, "recency": 0.05, "popularity": 0.10},
        "diversity": False,
    },
    "E. Full model + diversity reranking": {
        "description": "Model D followed by greedy cosine redundancy reranking.",
        "weights": {"relevance": 0.75, "jaccard": 0.10, "recency": 0.05, "popularity": 0.10},
        "diversity": True, "diversity_lambda": 0.75,
    },
    "F. Multi-interest profiles": {
        "description": ("Model D weights, but the history is split into leader-clustered interests: "
                        "cosine against the best-matching interest and top-K slots shared in "
                        "proportion to interest size."),
        "weights": {"relevance": 0.75, "jaccard": 0.10, "recency": 0.05, "popularity": 0.10},
        "diversity": False, "multi_interest": True,
    },
}
MODEL_NAMES = tuple(MODEL_CONFIGS)


@dataclass(frozen=True, slots=True)
class Metrics:
    impressions: int
    precision: float
    recall: float
    p_at_k: float


@dataclass(frozen=True, slots=True)
class EvaluationSplit:
    train: tuple[Impression, ...]
    test: tuple[Impression, ...]
    cutoff: datetime
    missing_timestamp_count: int


@dataclass(frozen=True, slots=True)
class EvaluationOutput:
    split: EvaluationSplit
    aggregate: tuple[dict[str, object], ...]
    per_user: tuple[dict[str, object], ...]
    summary: dict[str, object]


def evaluate_impressions(
    recommender: Recommender,
    impressions: Iterable[tuple[User, set[str], set[str]]],
    *, k: int = 5, baseline: bool = False,
) -> Metrics:
    """Compatibility helper for simple impression-level metric checks."""
    total_precision = total_recall = 0.0
    count = 0
    for user, shown, relevant in impressions:
        relevant = relevant - set(user.history)
        if not relevant:
            continue
        count += 1
        if baseline:
            ordered = sorted(shown - set(user.history),
                             key=lambda aid: (-recommender.click_counts.get(aid, 0), aid))
        else:
            ordered = [r.article_id for r in recommender.recommend(
                user, limit=k, candidate_mode="all", diverse=False, candidate_ids=shown)]
        hits = len(set(ordered[:k]) & relevant)
        total_precision += hits / k
        total_recall += hits / len(relevant)
    if not count:
        return Metrics(0, 0.0, 0.0, 0.0)
    precision = total_precision / count
    return Metrics(count, precision, total_recall / count, precision)


def chronological_split(impressions: Iterable[Impression], test_fraction: float = 0.2) -> EvaluationSplit:
    """Use a reproducible timestamp cutoff; impressions without times are omitted."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    rows = list(impressions)
    timed = sorted((row for row in rows if row.timestamp is not None),
                   key=lambda row: (row.timestamp, row.impression_id))
    missing = len(rows) - len(timed)
    if len(timed) < 2:
        raise ValueError("Chronological evaluation requires at least two timestamped impressions")
    # Choose a row boundary, then move the cutoff to its timestamp. Ties remain
    # on the earlier side so no simultaneous impressions are split across time.
    boundary = min(len(timed) - 1, max(1, int(len(timed) * (1.0 - test_fraction))))
    cutoff = timed[boundary - 1].timestamp
    assert cutoff is not None
    train = tuple(row for row in timed if row.timestamp is not None and row.timestamp <= cutoff)
    test = tuple(row for row in timed if row.timestamp is not None and row.timestamp > cutoff)
    if not train or not test:
        raise ValueError("The timestamp cutoff produced an empty train or test split")
    return EvaluationSplit(train, test, cutoff, missing)


def _dcg(ranked: list[str], relevant: set[str], k: int) -> float:
    from math import log2
    return sum((1.0 / log2(position + 1)) for position, article_id in enumerate(ranked[:k], 1)
               if article_id in relevant)


def _query_metrics(ranked: list[str], relevant: set[str], categories: dict[str, str],
                   item_vectors: ItemVectors, k: int) -> dict[str, float]:
    selected = ranked[:k]
    hits = len(set(selected) & relevant)
    ideal_hits = min(k, len(relevant))
    ideal_dcg = sum(1.0 / __import__("math").log2(position + 1) for position in range(1, ideal_hits + 1))
    pair_similarities = [cosine_similarity(item_vectors.vector(left), item_vectors.vector(right))
                         for index, left in enumerate(selected) for right in selected[index + 1:]]
    return {
        f"P@{k}": hits / k,
        f"R@{k}": hits / len(relevant) if relevant else 0.0,
        f"NDCG@{k}": _dcg(selected, relevant, k) / ideal_dcg if ideal_dcg else 0.0,
        f"CategoryDiversity@{k}": (len({categories.get(article_id, "") for article_id in selected
                                         if categories.get(article_id, "")}) / len(selected)
                                   if selected else 0.0),
        f"AvgPairwiseSimilarity@{k}": (mean(pair_similarities) if pair_similarities else 0.0),
    }


def _svg_bars(path: Path, title: str, series: list[tuple[str, list[float]]], labels: list[str]) -> None:
    """Write dependency-free, printable SVG grouped bars in the [0,1] range."""
    width, height = 1000, 520
    left, right, top, bottom = 80, 30, 75, 110
    plot_w, plot_h = width - left - right, height - top - bottom
    colors = ["#276FBF", "#F28E2B", "#59A14F", "#B07AA1"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width/2}" y="36" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold">{escape(title)}</text>']
    for tick in range(6):
        value = tick / 5
        y = top + plot_h * (1 - value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#dddddd"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-family="Arial" font-size="12">{value:.1f}</text>')
    group_w = plot_w / max(len(labels), 1)
    bar_w = min(40, group_w * 0.72 / max(len(series), 1))
    for li, label in enumerate(labels):
        center = left + group_w * (li + 0.5)
        for si, (name, values) in enumerate(series):
            value = values[li] if li < len(values) else 0
            x = center - len(series) * bar_w / 2 + si * bar_w
            bar_h = plot_h * max(0.0, min(value, 1.0))
            y = top + plot_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-2:.1f}" height="{bar_h:.1f}" fill="{colors[si % len(colors)]}"/>')
            parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-5:.1f}" text-anchor="middle" font-family="Arial" font-size="10">{value:.2f}</text>')
        parts.append(f'<text x="{center:.1f}" y="{top+plot_h+25}" text-anchor="middle" font-family="Arial" font-size="12">{escape(label)}</text>')
    legend_y = height - 36
    total_legend_w = len(series) * 215
    start = max(left, (width - total_legend_w) / 2)
    for index, (name, _) in enumerate(series):
        x = start + index * 215
        parts.append(f'<rect x="{x:.1f}" y="{legend_y-12}" width="14" height="14" fill="{colors[index % len(colors)]}"/>')
        parts.append(f'<text x="{x+20:.1f}" y="{legend_y}" font-family="Arial" font-size="12">{escape(name)}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def run_evaluation(dataset: ProcessedDataset, *, test_fraction: float = 0.2,
                   ks: tuple[int, ...] = (5, 10), diversity_lambda: float = 0.75) -> EvaluationOutput:
    """Evaluate fixed ablation models on the same impression-level future clicks."""
    if not ks or any(k < 1 for k in ks):
        raise ValueError("ks must contain positive values")
    split = chronological_split(dataset.impressions, test_fraction)
    train_clicks: Counter[str] = Counter()
    first_train_shown: dict[str, datetime] = {}
    read_in_train: set[str] = set()
    for row in split.train:
        train_clicks.update(row.clicked)
        read_in_train.update(row.history)
        assert row.timestamp is not None
        for article_id in row.shown:
            previous = first_train_shown.get(article_id)
            if previous is None or row.timestamp < previous:
                first_train_shown[article_id] = row.timestamp
    log_start = min(row.timestamp for row in split.train if row.timestamp is not None)

    # MIND news.tsv has no publish timestamp, so freshness uses what the
    # system could know at the cutoff: the first training impression that
    # showed the article; the log start for articles only seen in reading
    # histories (read before the log began); and the cutoff itself for
    # articles never seen before it, which are new in the test period.
    def freshness_proxy(article_id: str) -> datetime:
        if article_id in first_train_shown:
            return first_train_shown[article_id]
        return log_start if article_id in read_in_train else split.cutoff

    eval_articles = {
        article_id: (article if article.timestamp is not None else
                     replace(article, timestamp=freshness_proxy(article_id)))
        for article_id, article in dataset.articles.items()
    }
    eval_dataset = replace(dataset, articles=eval_articles, click_counts=dict(train_clicks))
    recommender = Recommender.from_processed(eval_dataset)
    max_k = max(ks)
    weights_cosine = ScoreWeights(alpha=1, beta=0, gamma=0, delta=0)
    weights_cosine_jaccard = ScoreWeights(alpha=0.90, beta=0.10, gamma=0, delta=0)
    weights_full = ScoreWeights()
    users: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    all_recommended: dict[str, set[str]] = {name: set() for name in MODEL_NAMES}
    user_recommended: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    query_count = skipped_unknown = skipped_no_positive = 0
    copies_removed = relevant_copies_removed = 0
    evaluation_fingerprint_rows: list[str] = []
    categories = {aid: article.category for aid, article in recommender.article_by_id.items()}

    for row in split.test:
        history = set(row.history)
        shown = {aid for aid in row.shown if aid in recommender.available_article_ids}
        relevant = (set(row.clicked) & shown) - history
        if not relevant:
            skipped_no_positive += 1
            continue
        if not shown:
            skipped_unknown += 1
            continue
        query_count += 1
        evaluation_fingerprint_rows.append(
            f"{row.impression_id}\t{row.user_id}\t{','.join(sorted(relevant))}")
        user = User(row.user_id, row.history)
        # Score each displayed candidate once. Earlier evaluation repeated the
        # full explanation and component-scoring path four times per impression;
        # reusing the exposed components keeps the ablation identical while
        # making a 30k-impression MIND run practical.
        scored = recommender.recommend(
            user, limit=len(shown), as_of=row.timestamp, candidate_mode="all", diverse=False,
            score_weights=weights_full, candidate_ids=shown)
        removed = {article_id for article_id, _ in
                   recommender.last_diagnostics["history_copies_removed"]}
        copies_removed += len(removed)
        relevant_copies_removed += len(removed & relevant)
        # The baseline ranks the same candidates: copies of read articles are
        # removed for every model, so only the ranking differs.
        popularity_ranked = sorted((item.article_id for item in scored),
                                   key=lambda aid: (-train_clicks.get(aid, 0), aid))[:max_k]
        def rank_by(weights: ScoreWeights) -> list[str]:
            return [item.article_id for item in sorted(
                scored,
                key=lambda item: (-calculate_net_score(
                    item.relevance_normalized, jaccard=item.jaccard_normalized,
                    recency=item.recency, popularity=item.popularity, weights=weights),
                    item.article_id))[:max_k]]
        cosine_ranked = rank_by(weights_cosine)
        jaccard_ranked = rank_by(weights_cosine_jaccard)
        full_ranked = rank_by(weights_full)
        diverse_ranked = [item.article_id for item in diversity_rerank(
            scored, recommender.item_vectors, limit=max_k,
            relevance_weight=diversity_lambda)]
        # Interest quotas shape which items fill the top K, so F is ranked by
        # its own call with limit=max_k rather than re-sorted from ``scored``.
        multi_interest_ranked = [item.article_id for item in recommender.recommend(
            user, limit=max_k, as_of=row.timestamp, candidate_mode="all", diverse=False,
            score_weights=weights_full, candidate_ids=shown, multi_interest=True)]
        rankings = dict(zip(MODEL_NAMES, (popularity_ranked, cosine_ranked, jaccard_ranked,
                                           full_ranked, diverse_ranked, multi_interest_ranked)))
        for model_name, ranked in rankings.items():
            all_recommended[model_name].update(ranked[:max_k])
            user_recommended[row.user_id][model_name].update(ranked[:max_k])
            per_k = {}
            for k in ks:
                per_k.update(_query_metrics(ranked, relevant, categories,
                                            recommender.item_vectors, k))
            users[row.user_id][model_name].append(per_k)

    if not users:
        raise ValueError("No evaluable held-out impressions have a known, unconsumed clicked candidate")
    catalog_size = len(recommender.available_article_ids)
    per_user_rows: list[dict[str, object]] = []
    for user_id in sorted(users):
        for model in MODEL_NAMES:
            query_rows = users[user_id].get(model, [])
            if not query_rows:
                continue
            row_out: dict[str, object] = {"user_id": user_id, "model": model,
                                          "test_impressions": len(query_rows)}
            metric_names = query_rows[0].keys()
            row_out.update({name: mean(item[name] for item in query_rows) for name in metric_names})
            row_out["Coverage"] = len(user_recommended[user_id][model]) / catalog_size if catalog_size else 0.0
            per_user_rows.append(row_out)

    aggregate_rows: list[dict[str, object]] = []
    for model in MODEL_NAMES:
        model_rows = [item for item in per_user_rows if item["model"] == model]
        aggregate: dict[str, object] = {"Model": model}
        for metric_name in model_rows[0]:
            if metric_name in {"user_id", "model", "test_impressions"}:
                continue
            aggregate[metric_name] = mean(float(item[metric_name]) for item in model_rows)
        aggregate["Coverage"] = len(all_recommended[model]) / catalog_size if catalog_size else 0.0
        aggregate_rows.append(aggregate)

    summary = {
        "split_method": "global chronological cutoff; train timestamp <= cutoff; test timestamp > cutoff",
        "cutoff": split.cutoff.isoformat(), "train_impressions": len(split.train),
        "test_impressions": len(split.test), "missing_timestamp_impressions_omitted": split.missing_timestamp_count,
        "evaluated_impressions": query_count, "evaluated_users": len(users),
        "evaluation_set_sha256": hashlib.sha256("\n".join(sorted(evaluation_fingerprint_rows)).encode("utf-8")).hexdigest(),
        "heldout_impressions_skipped_no_known_unconsumed_positive": skipped_no_positive,
        "heldout_impressions_skipped_no_known_candidates": skipped_unknown,
        "candidate_policy": "Only valid articles shown in each held-out impression; consumed history excluded.",
        "history_copy_policy": ("Shown articles with the same normalized title as, or cosine >= "
                                f"{recommender.history_duplicate_threshold} to, an already-read article "
                                "are removed for every model, including the baseline."),
        "history_copies_removed": copies_removed,
        "relevant_history_copies_removed": relevant_copies_removed,
        "history_policy": "Each held-out impression's recorded pre-impression history only.",
        "popularity_policy": "Click counts from training impressions only.",
        "recency_policy": ("MIND has no publication time. Proxy timestamp: first training impression "
                           "that showed the article; training-log start for articles only seen in "
                           "training histories; the cutoff for articles never seen before it (new)."),
        "aggregation": "Macro-average query metrics within each user, then macro-average users.",
        "ks": list(ks), "diversity_lambda": diversity_lambda,
        "ablation_models": MODEL_CONFIGS,
        "normalization": ("Cosine and Jaccard are divided by their maximum over each impression's "
                          "candidates; then weighted mean: sum(weight_i * component_i) / sum(weight_i)."),
        "tuning_policy": "Fixed component weights specified before evaluation; no evaluation-set tuning.",
        "catalog_size": catalog_size,
    }
    return EvaluationOutput(split, tuple(aggregate_rows), tuple(per_user_rows), summary)


def write_evaluation(output: EvaluationOutput, output_dir: str | Path, ks: tuple[int, ...] = (5, 10)) -> None:
    """Write reproducible tables, policy summary, and report-ready SVG figures."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("aggregate_metrics.csv", output.aggregate), ("per_user_metrics.csv", output.per_user)):
        if rows:
            columns = list(rows[0].keys())
            with (target / filename).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
    (target / "run_summary.json").write_text(json.dumps(output.summary, indent=2, sort_keys=True), encoding="utf-8")
    ablation_payload = {
        "experiment": "Chronological T3 recommender component ablation",
        "evaluation_protocol": output.summary,
        "models": list(output.aggregate),
    }
    (target / "ablation_results.json").write_text(
        json.dumps(ablation_payload, indent=2, sort_keys=True), encoding="utf-8")
    if output.aggregate:
        preferred = ["Model", *(metric for k in ks for metric in (f"P@{k}", f"R@{k}")),
                     "Coverage", f"CategoryDiversity@{max(ks)}",
                     f"AvgPairwiseSimilarity@{max(ks)}"]
        csv_columns = [name for name in preferred if name in output.aggregate[0]]
        csv_columns.extend(name for name in output.aggregate[0] if name not in csv_columns)
        with (target / "ablation_results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerows(output.aggregate)
    models = [str(row["Model"]) for row in output.aggregate]
    plot_labels = ["A Popularity", "B Cosine", "C + Jaccard", "D + quality", "E + diversity",
                   "F Multi-interest"]
    if len(plot_labels) != len(models):
        plot_labels = models
    quality_metrics = [name for k in ks for name in (f"P@{k}", f"R@{k}")]
    _svg_bars(target / "ranking_metrics.svg", "Chronological recommendation quality",
              [(name, [float(row.get(name, 0.0)) for row in output.aggregate]) for name in quality_metrics], plot_labels)
    _svg_bars(target / "ablation_plot.svg", "Fixed-weight model ablation",
              [(name, [float(row.get(name, 0.0)) for row in output.aggregate])
               for name in quality_metrics], plot_labels)
    diversity_metrics_names = ["Coverage", f"CategoryDiversity@{max(ks)}",
                               f"AvgPairwiseSimilarity@{max(ks)}"]
    _svg_bars(target / "coverage_diversity.svg", "Catalog coverage and category diversity",
              [(name, [float(row.get(name, 0.0)) for row in output.aggregate])
               for name in diversity_metrics_names], plot_labels)
