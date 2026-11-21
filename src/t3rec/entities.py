"""Wikidata entities from MIND news rows: centred entity vectors and labels.

MIND tags each title and abstract with Wikidata entities and ships a 100-d
TransE embedding per entity (``entity_embedding.vec``). An article's entity
vector is the confidence-weighted mean of its entities' embeddings (title
entities count double). Raw embeddings share a common direction, so random
article pairs already score about 0.29; subtracting the catalog mean centres
them (random pairs about 0.0, same-subcategory pairs about 0.2).
"""

from __future__ import annotations

import json
import logging
from array import array
from collections import Counter
from dataclasses import dataclass
from math import sqrt
from operator import mul
from pathlib import Path
from typing import Iterable, Mapping

from .models import Article

LOGGER = logging.getLogger("t3rec.entities")
EntityVector = array


def article_entities(article: Article, title_weight: float = 2.0) -> list[tuple[str, str, float]]:
    """Return (Wikidata ID, label, weight) for each entity tagged on the article."""
    found: list[tuple[str, str, float]] = []
    for key, weight in (("title_entities", title_weight), ("abstract_entities", 1.0)):
        raw = (article.metadata or {}).get(key) or "[]"
        try:
            rows = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for row in rows if isinstance(rows, list) else ():
            entity_id = str(row.get("WikidataId") or "").strip()
            if entity_id:
                confidence = float(row.get("Confidence", 1.0) or 0.0)
                found.append((entity_id, str(row.get("Label") or entity_id), weight * confidence))
    return found


def load_embeddings(path: str | Path) -> dict[str, EntityVector]:
    """Read ``entity_embedding.vec``: a Wikidata ID then its floats, tab-separated."""
    embeddings: dict[str, EntityVector] = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            parts = line.split()
            if len(parts) > 1:
                embeddings[parts[0]] = array("d", map(float, parts[1:]))
    return embeddings


def dot(left: EntityVector | None, right: EntityVector | None) -> float:
    """Dot product of two entity vectors; 0 when either is missing."""
    if left is None or right is None:
        return 0.0
    return sum(map(mul, left, right))


def _normalized(values: list[float]) -> EntityVector | None:
    norm = sqrt(sum(value * value for value in values))
    return array("d", (value / norm for value in values)) if norm else None


@dataclass(slots=True)
class EntityIndex:
    """Per-article centred entity vectors and entity labels."""

    vectors: dict[str, EntityVector]
    labels: dict[str, tuple[tuple[str, float], ...]]

    @classmethod
    def empty(cls) -> "EntityIndex":
        return cls({}, {})

    @classmethod
    def from_mind(cls, raw_dir: str | Path, articles: Mapping[str, Article],
                  title_weight: float = 2.0) -> "EntityIndex":
        """Build the index from a MIND folder; without an embedding file only labels are kept."""
        path = Path(raw_dir) / "entity_embedding.vec"
        embeddings = load_embeddings(path) if path.is_file() else {}
        if not embeddings:
            LOGGER.info("No entity embeddings at %s; entity similarity will be 0", path)
        return cls.build(articles, embeddings, title_weight=title_weight)

    @classmethod
    def build(cls, articles: Mapping[str, Article], embeddings: Mapping[str, EntityVector],
              *, title_weight: float = 2.0) -> "EntityIndex":
        means: dict[str, list[float]] = {}
        labels: dict[str, tuple[tuple[str, float], ...]] = {}
        for article_id, article in articles.items():
            entities = article_entities(article, title_weight)
            weights: Counter[str] = Counter()
            for _, label, weight in entities:
                weights[label] += weight
            labels[article_id] = tuple(sorted(weights.items(), key=lambda pair: (-pair[1], pair[0])))
            total, accumulated = 0.0, None
            for entity_id, _, weight in entities:
                vector = embeddings.get(entity_id)
                if vector is None or weight <= 0:
                    continue
                if accumulated is None:
                    accumulated = [0.0] * len(vector)
                for position, value in enumerate(vector):
                    accumulated[position] += weight * value
                total += weight
            if accumulated is not None and total:
                means[article_id] = [value / total for value in accumulated]
        vectors: dict[str, EntityVector] = {}
        if means:
            count = len(means)
            centre = [sum(column) / count for column in zip(*means.values())]
            for article_id, values in means.items():
                centred = _normalized([value - mean for value, mean in zip(values, centre)])
                if centred is not None:
                    vectors[article_id] = centred
        LOGGER.info("Entity vectors for %d of %d articles", len(vectors), len(articles))
        return cls(vectors, labels)

    def vector(self, article_id: str) -> EntityVector | None:
        return self.vectors.get(article_id)

    def profile(self, article_ids: Iterable[str]) -> EntityVector | None:
        """Normalized sum of the entity vectors of the given articles."""
        accumulated: list[float] | None = None
        for article_id in dict.fromkeys(article_ids):
            vector = self.vectors.get(article_id)
            if vector is None:
                continue
            if accumulated is None:
                accumulated = list(vector)
            else:
                for position, value in enumerate(vector):
                    accumulated[position] += value
        return _normalized(accumulated) if accumulated is not None else None
