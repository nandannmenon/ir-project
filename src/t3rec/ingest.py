"""Deterministic MIND TSV ingestion, validation, and processed-data caching."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .data import parse_mind_time
from .models import Article, User
from .text import TextConfig
from .vectorize import ItemVectors, build_item_vectors

LOGGER = logging.getLogger("t3rec.ingest")
CACHE_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class Impression:
    impression_id: str
    user_id: str
    timestamp: datetime | None
    history: tuple[str, ...]
    shown: tuple[str, ...]
    clicked: tuple[str, ...]


@dataclass(slots=True)
class ProcessedDataset:
    articles: dict[str, Article]
    users: dict[str, User]
    impressions: tuple[Impression, ...]
    click_counts: dict[str, int]
    item_vectors: ItemVectors
    validation: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_signature(news_path: Path, behaviors_path: Path, config: TextConfig) -> dict[str, Any]:
    stemmer = config.stemmer
    stemmer_name = None if stemmer is None else f"{stemmer.__module__}.{stemmer.__qualname__}"
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "news_sha256": _sha256(news_path),
        "behaviors_sha256": _sha256(behaviors_path),
        "preprocessing": {
            "stopwords": sorted(config.stopwords),
            "min_token_length": config.min_token_length,
            "stemmer": stemmer_name,
        },
        "tfidf": "log-tf-smoothed-idf-l2-v1",
    }


def _parse_news(path: Path, missing: Counter[str]) -> dict[str, Article]:
    articles: dict[str, Article] = {}
    row_count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        for line_number, row in enumerate(reader, start=1):
            row_count += 1
            if len(row) < 8:
                LOGGER.warning("Skipping malformed news row %d: expected 8 columns, got %d", line_number, len(row))
                missing["malformed_news_rows"] += 1
                continue
            fields = row[:8]
            article_id, category, subcategory, title, abstract, url, title_entities, abstract_entities = fields
            article_id = article_id.strip()
            if not article_id:
                LOGGER.warning("Skipping news row %d with an empty article ID", line_number)
                missing["missing_article_id"] += 1
                continue
            for name, value in (("title", title), ("abstract", abstract),
                                ("category", category), ("subcategory", subcategory)):
                if not value.strip():
                    missing[f"missing_{name}"] += 1
            if article_id in articles:
                LOGGER.warning("Duplicate article ID %s at news row %d; keeping first row", article_id, line_number)
                missing["duplicate_article_id"] += 1
                continue
            articles[article_id] = Article(
                article_id=article_id,
                title=title.strip(),
                abstract=abstract.strip(),
                category=category.strip(),
                subcategory=subcategory.strip(),
                # Standard MIND news.tsv contains no publication timestamp.
                timestamp=None,
                metadata={
                    "url": url.strip(),
                    "title_entities": title_entities.strip(),
                    "abstract_entities": abstract_entities.strip(),
                },
            )
    LOGGER.info("Read %d news rows; retained %d unique articles", row_count, len(articles))
    if not articles:
        raise ValueError(f"No valid articles found in {path}")
    return articles


def _parse_behaviors(path: Path, missing: Counter[str]) -> tuple[Impression, ...]:
    impressions: list[Impression] = []
    impression_ids: set[str] = set()
    row_count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        for line_number, row in enumerate(reader, start=1):
            row_count += 1
            if len(row) < 5:
                LOGGER.warning("Skipping malformed behavior row %d: expected 5 columns, got %d", line_number, len(row))
                missing["malformed_behavior_rows"] += 1
                continue
            impression_id, user_id, time_text, history_text, impression_text = row[:5]
            impression_id, user_id = impression_id.strip(), user_id.strip()
            if not impression_id:
                missing["missing_impression_id"] += 1
                LOGGER.warning("Skipping behavior row %d with an empty impression ID", line_number)
                continue
            if not user_id:
                missing["missing_user_id"] += 1
                LOGGER.warning("Skipping behavior row %d with an empty user ID", line_number)
                continue
            if impression_id in impression_ids:
                missing["duplicate_impression_id"] += 1
                LOGGER.warning("Duplicate impression ID %s; keeping first row", impression_id)
                continue
            impression_ids.add(impression_id)
            if not time_text.strip():
                missing["missing_impression_time"] += 1
            timestamp = parse_mind_time(time_text.strip())
            if time_text.strip() and timestamp is None:
                missing["invalid_impression_time"] += 1
                LOGGER.warning("Could not parse timestamp %r on behavior row %d", time_text, line_number)
            if not history_text.strip():
                missing["empty_history"] += 1
            if not impression_text.strip():
                missing["empty_impressions"] += 1
            history = tuple(history_text.split())
            shown: list[str] = []
            clicked: list[str] = []
            for candidate in impression_text.split():
                article_id, separator, label = candidate.rpartition("-")
                if not separator or label not in {"0", "1"} or not article_id:
                    missing["malformed_impression_candidates"] += 1
                    LOGGER.warning("Ignoring malformed candidate %r on behavior row %d", candidate, line_number)
                    continue
                shown.append(article_id)
                if label == "1":
                    clicked.append(article_id)
            impressions.append(Impression(impression_id, user_id, timestamp, history,
                                          tuple(shown), tuple(clicked)))
    LOGGER.info("Read %d behavior rows; retained %d impressions", row_count, len(impressions))
    if not impressions:
        raise ValueError(f"No valid impressions found in {path}")
    return tuple(impressions)


def _make_users(impressions: tuple[Impression, ...]) -> tuple[dict[str, User], dict[str, int]]:
    histories: dict[str, list[tuple[datetime | None, int, int, str]]] = defaultdict(list)
    fallback: dict[str, tuple[datetime | None, int, tuple[str, ...]]] = {}
    click_counts: Counter[str] = Counter()
    for row_order, impression in enumerate(impressions):
        fallback[impression.user_id] = (impression.timestamp, row_order, impression.history)
        for click_order, article_id in enumerate(impression.clicked):
            histories[impression.user_id].append((impression.timestamp, row_order, click_order, article_id))
            click_counts[article_id] += 1
    users: dict[str, User] = {}
    for user_id, snapshot in fallback.items():
        events = histories.get(user_id)
        if events:
            # Stable order: timestamp (missing last), original row, then candidate order.
            events.sort(key=lambda event: (event[0] is None,
                                           event[0] or datetime.max,
                                           event[1], event[2]))
            users[user_id] = User(user_id,
                                  tuple(event[3] for event in events),
                                  tuple(event[0] for event in events))
        else:
            users[user_id] = User(user_id, snapshot[2])
    return users, dict(click_counts)


def _validate(articles: dict[str, Article], users: dict[str, User],
              impressions: tuple[Impression, ...], item_vectors: ItemVectors,
              missing: Counter[str]) -> dict[str, Any]:
    known_ids = set(articles)
    history_refs = {article_id for row in impressions for article_id in row.history}
    shown_refs = {article_id for row in impressions for article_id in row.shown}
    missing["unknown_history_article_ids"] += len(history_refs - known_ids)
    missing["unknown_impression_article_ids"] += len(shown_refs - known_ids)
    vocabulary_size = len(item_vectors.vocabulary)
    matrix_cells = len(articles) * vocabulary_size
    nonzero = sum(len(vector) for vector in item_vectors.vectors.values())
    sparsity = 1.0 - nonzero / matrix_cells if matrix_cells else 0.0
    checked_fields = (
        "missing_article_id", "missing_title", "missing_abstract", "missing_category",
        "missing_subcategory", "missing_impression_id", "missing_user_id",
        "missing_impression_time", "invalid_impression_time", "empty_history",
        "empty_impressions", "malformed_news_rows", "malformed_behavior_rows",
        "malformed_impression_candidates", "duplicate_article_id", "duplicate_impression_id",
        "unknown_history_article_ids", "unknown_impression_article_ids",
    )
    return {
        "article_count": len(articles),
        "user_count": len(users),
        "impression_count": len(impressions),
        "interaction_count": sum(len(row.clicked) for row in impressions),
        "shown_candidate_count": sum(len(row.shown) for row in impressions),
        "missing_fields": {key: missing.get(key, 0) for key in checked_fields},
        "vocabulary_size": vocabulary_size,
        "tfidf_nonzero_count": nonzero,
        "tfidf_matrix_cells": matrix_cells,
        "tfidf_sparsity": sparsity,
        "unknown_history_article_count": len(history_refs - known_ids),
        "unknown_impression_article_count": len(shown_refs - known_ids),
    }


def _serialize(dataset: ProcessedDataset, signature: dict[str, Any]) -> dict[str, Any]:
    def dt(value: datetime | None) -> str | None:
        return value.isoformat() if value else None

    return {
        "signature": signature,
        "articles": [{"article_id": a.article_id, "title": a.title, "abstract": a.abstract,
                      "category": a.category, "subcategory": a.subcategory,
                      "timestamp": dt(a.timestamp), "metadata": dict(a.metadata)}
                     for a in sorted(dataset.articles.values(), key=lambda a: a.article_id)],
        "users": [{"user_id": u.user_id, "history": list(u.history),
                   "timestamps": [dt(t) for t in u.timestamps]}
                  for u in sorted(dataset.users.values(), key=lambda u: u.user_id)],
        "impressions": [{"impression_id": row.impression_id, "user_id": row.user_id,
                         "timestamp": dt(row.timestamp), "history": list(row.history),
                         "shown": list(row.shown), "clicked": list(row.clicked)}
                        for row in dataset.impressions],
        "click_counts": dict(sorted(dataset.click_counts.items())),
        "item_vectors": {
            "vectors": {key: value for key, value in sorted(dataset.item_vectors.vectors.items())},
            "document_frequency": dict(sorted(dataset.item_vectors.document_frequency.items())),
            "idf": dict(sorted(dataset.item_vectors.idf.items())),
            "vocabulary": list(dataset.item_vectors.vocabulary),
        },
        "validation": dataset.validation,
    }


def _deserialize(record: dict[str, Any]) -> ProcessedDataset:
    def dt(value: str | None) -> datetime | None:
        return datetime.fromisoformat(value) if value else None

    articles = {
        row["article_id"]: Article(row["article_id"], row["title"], row["abstract"],
                                    row["category"], row["subcategory"], dt(row["timestamp"]),
                                    row["metadata"])
        for row in record["articles"]
    }
    users = {row["user_id"]: User(row["user_id"], tuple(row["history"]),
                                  tuple(dt(t) for t in row["timestamps"]))
             for row in record["users"]}
    impressions = tuple(Impression(row["impression_id"], row["user_id"], dt(row["timestamp"]),
                                   tuple(row["history"]), tuple(row["shown"]), tuple(row["clicked"]))
                       for row in record["impressions"])
    vector_record = record["item_vectors"]
    item_vectors = ItemVectors(
        {key: {term: float(weight) for term, weight in value.items()}
         for key, value in vector_record["vectors"].items()},
        vector_record["document_frequency"],
        {term: float(value) for term, value in vector_record["idf"].items()},
        tuple(vector_record["vocabulary"]),
    )
    return ProcessedDataset(articles, users, impressions, record["click_counts"],
                            item_vectors, record["validation"])


def _write_cache(cache_path: Path, record: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(encoded)
    os.replace(temporary, cache_path)


def _read_cache(cache_path: Path) -> dict[str, Any]:
    with gzip.open(cache_path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def ingest_mind(raw_dir: str | Path, output_dir: str | Path, *,
                config: TextConfig | None = None, force: bool = False) -> ProcessedDataset:
    """Load MIND train/dev folder and reuse processed records when inputs match."""
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    news_path, behaviors_path = raw_dir / "news.tsv", raw_dir / "behaviors.tsv"
    for source in (news_path, behaviors_path):
        if not source.is_file():
            raise FileNotFoundError(f"Expected MIND file {source}; place news.tsv and behaviors.tsv in raw_dir")
    cfg = config or TextConfig()
    signature = _source_signature(news_path, behaviors_path, cfg)
    cache_path = output_dir / "dataset.json.gz"
    if cache_path.is_file() and not force:
        try:
            cached = _read_cache(cache_path)
            if cached.get("signature") == signature:
                dataset = _deserialize(cached)
                LOGGER.info("Cache hit: loaded cleaned records and TF-IDF vectors from %s", cache_path)
                LOGGER.info("Validation summary: %s", json.dumps(dataset.validation, sort_keys=True))
                return dataset
            LOGGER.info("Cache source or preprocessing settings changed; rebuilding cache")
        except (OSError, EOFError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            LOGGER.warning("Ignoring unreadable processed cache %s: %s", cache_path, exc)
    else:
        LOGGER.info("Cache miss: building processed data")

    missing: Counter[str] = Counter()
    articles = _parse_news(news_path, missing)
    impressions = _parse_behaviors(behaviors_path, missing)
    users, click_counts = _make_users(impressions)
    item_vectors = build_item_vectors(articles.values(), cfg)
    validation = _validate(articles, users, impressions, item_vectors, missing)
    dataset = ProcessedDataset(articles, users, impressions, click_counts, item_vectors, validation)
    _write_cache(cache_path, _serialize(dataset, signature))
    LOGGER.info("Wrote deterministic processed cache to %s", cache_path)
    LOGGER.info("Validation summary: %s", json.dumps(validation, sort_keys=True))
    return dataset
