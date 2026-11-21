"""Learning to rank: a pairwise logistic (linear RankNet) ranker over IR features.

Instead of hand-set fusion weights (0.75 cosine, 0.10 Jaccard, ...), the
ranker learns how much each signal matters from clicks. Within one impression
every (clicked, skipped) pair is a training example, and the model learns
w such that w . x_clicked > w . x_skipped:

    loss = log(1 + exp(-w . (z(x_clicked) - z(x_skipped)))) + l2 / 2 * |w|^2

where z standardizes each feature with the training mean and deviation, so
the learned weights are comparable across features. Training uses only
impressions from the training period (``evaluate.run_evaluation`` takes its
latest slice); the held-out test impressions are never seen.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from math import exp, sqrt
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .entities import dot
from .models import Recommendation

FEATURES: tuple[str, ...] = (
    "cosine",               # raw TF-IDF cosine to the single profile
    "cosine_scaled",        # cosine / best cosine among this impression's candidates
    "jaccard_scaled",       # typed term + category set overlap, scaled the same way
    "recency",              # freshness proxy (first exposure), 14-day half-life
    "popularity",           # static quality g(d): log(1 + training clicks) / max
    "interest_similarity",  # similarity to the best-matching interest (multi-interest)
    "entity_similarity",    # centred entity-vector cosine to the reading history
    "category_match",       # share of the history in the article's category
)


def impression_features(
    rows: Sequence[Recommendation],
    interest_similarity: Mapping[str, float],
    entity_profile,
    entity_index,
) -> dict[str, list[float]]:
    """One feature vector per candidate, from a single-profile ``recommend`` call.

    ``interest_similarity`` maps article IDs to their best-interest similarity
    from a multi-interest call over the same candidates; ``entity_profile`` is
    the reader's centred entity vector (``EntityIndex.profile``) or None.
    """
    features: dict[str, list[float]] = {}
    for row in rows:
        entity = entity_index.vector(row.article_id) if entity_index is not None else None
        features[row.article_id] = [
            row.relevance,
            row.relevance_normalized,
            row.jaccard_normalized,
            row.recency,
            row.popularity,
            interest_similarity.get(row.article_id, row.relevance),
            dot(entity_profile, entity),
            row.category_match,
        ]
    return features


@dataclass(slots=True)
class LinearRanker:
    """score(x) = sum_i weight_i * (x_i - mean_i) / scale_i."""

    features: tuple[str, ...]
    weights: list[float]
    mean: list[float]
    scale: list[float]

    def score(self, values: Sequence[float]) -> float:
        return sum(weight * (value - mean) / scale
                   for weight, value, mean, scale in zip(self.weights, values, self.mean, self.scale))

    def rank(self, features: Mapping[str, Sequence[float]]) -> list[str]:
        """Article IDs by descending score, ties broken by ID."""
        return sorted(features, key=lambda article_id: (-self.score(features[article_id]), article_id))

    def as_dict(self) -> dict[str, object]:
        return {"features": list(self.features), "weights": self.weights,
                "mean": self.mean, "scale": self.scale}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LinearRanker":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(tuple(data["features"]), data["weights"], data["mean"], data["scale"])


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    z = exp(value)
    return z / (1.0 + z)


def train_pairwise(
    groups: Iterable[tuple[Sequence[Sequence[float]], Sequence[Sequence[float]]]],
    *,
    features: tuple[str, ...] = FEATURES,
    epochs: int = 8,
    learning_rate: float = 0.05,
    l2: float = 1e-4,
    negatives_per_positive: int = 4,
    seed: int = 0,
) -> LinearRanker:
    """Fit a linear ranker on (clicked rows, skipped rows) groups, one per impression.

    Each clicked row is paired with up to ``negatives_per_positive`` skipped
    rows of the same impression, drawn with a fixed seed. Plain SGD with a
    decaying step; everything is deterministic for a given input order.
    """
    groups = [(list(pos), list(neg)) for pos, neg in groups if pos and neg]
    if not groups:
        raise ValueError("training needs impressions with both clicked and skipped candidates")
    width = len(features)
    rows = [row for pos, neg in groups for row in (*pos, *neg)]
    mean = [sum(row[i] for row in rows) / len(rows) for i in range(width)]
    scale = []
    for i in range(width):
        variance = sum((row[i] - mean[i]) ** 2 for row in rows) / len(rows)
        scale.append(sqrt(variance) if variance > 1e-12 else 1.0)
    z = lambda row: [(row[i] - mean[i]) / scale[i] for i in range(width)]  # noqa: E731
    rng = random.Random(seed)
    pairs: list[list[float]] = []
    for pos, neg in groups:
        for clicked in pos:
            chosen = neg if len(neg) <= negatives_per_positive else rng.sample(neg, negatives_per_positive)
            zc = z(clicked)
            for skipped in chosen:
                zs = z(skipped)
                pairs.append([a - b for a, b in zip(zc, zs)])
    weights = [0.0] * width
    for epoch in range(epochs):
        rng.shuffle(pairs)
        step = learning_rate / (1.0 + epoch)
        for diff in pairs:
            margin = sum(w * d for w, d in zip(weights, diff))
            gradient = -_sigmoid(-margin)
            for i in range(width):
                weights[i] -= step * (gradient * diff[i] + l2 * weights[i])
    return LinearRanker(tuple(features), weights, mean, scale)


def pairwise_accuracy(ranker: LinearRanker,
                      groups: Iterable[tuple[Sequence[Sequence[float]], Sequence[Sequence[float]]]]) -> float:
    """Share of (clicked, skipped) pairs the ranker orders correctly; ties count half."""
    correct = total = 0.0
    for pos, neg in groups:
        neg_scores = [ranker.score(row) for row in neg]
        for row in pos:
            score = ranker.score(row)
            for other in neg_scores:
                correct += 1.0 if score > other else 0.5 if score == other else 0.0
                total += 1
    return correct / total if total else 0.0
