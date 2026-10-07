"""Load MIND TSV files or compact JSON fixtures without retaining raw rows."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

from .models import Article, User


def parse_mind_time(value: str) -> datetime | None:
    if not value:
        return None
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def load_mind_news(path: str | Path) -> list[Article]:
    articles: list[Article] = []
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream, delimiter="\t"):
            if len(row) < 7:
                continue
            article_id, category, subcategory, title, abstract, url = row[:6]
            articles.append(Article(article_id, title, abstract, category, subcategory,
                                    metadata={"url": url}))
    return articles


def load_mind_behaviors(path: str | Path) -> list[User]:
    """Return one user row per impression; histories are the pre-impression clicks."""
    users: list[User] = []
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream, delimiter="\t"):
            if len(row) < 5:
                continue
            _impression_id, user_id, timestamp, history, _impressions = row[:5]
            # The row timestamp belongs to this impression, not individual history clicks.
            # MIND does not provide per-click timestamps inside the history field.
            users.append(User(user_id, tuple(history.split()) if history else ()))
    return users


def load_mind_impressions(path: str | Path) -> Iterator[tuple[User, set[str], set[str]]]:
    """Yield (pre-impression user, shown article IDs, clicked article IDs)."""
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream, delimiter="\t"):
            if len(row) < 5:
                continue
            _impression_id, user_id, _time, history, impressions = row[:5]
            shown: set[str] = set()
            clicked: set[str] = set()
            for item in impressions.split():
                article_id, separator, label = item.rpartition("-")
                if not separator:
                    continue
                shown.add(article_id)
                if label == "1":
                    clicked.add(article_id)
            yield User(user_id, tuple(history.split()) if history else ()), shown, clicked


def load_json_fixture(path: str | Path) -> tuple[list[Article], list[User]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    articles = [Article(**{**row, "timestamp": parse_mind_time(row.get("timestamp", ""))})
                for row in data["articles"]]
    users = [User(row["user_id"], tuple(row["history"]),
                  tuple(parse_mind_time(t) for t in row.get("timestamps", [])))
             for row in data["users"]]
    return articles, users
