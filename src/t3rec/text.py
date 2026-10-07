"""Transparent, configurable text normalization and tokenization."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

TOKEN_RE = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", re.UNICODE)
DEFAULT_STOPWORDS = frozenset(
    "a an and are as at be been being by for from has have he her hers him his i "
    "in into is it its me my of on or our ours she that the their theirs them "
    "they this to was we were what when where which who will with you your".split()
)


@dataclass(frozen=True, slots=True)
class TextConfig:
    stopwords: frozenset[str] = field(default_factory=lambda: DEFAULT_STOPWORDS)
    stemmer: Callable[[str], str] | None = None
    min_token_length: int = 2


def preprocess_text(text: str, config: TextConfig | None = None) -> list[str]:
    """Lowercase, normalize apostrophes, tokenize, remove stopwords, optionally stem."""
    cfg = config or TextConfig()
    normalized = text.lower().replace("’", "'")
    tokens = TOKEN_RE.findall(normalized)
    result: list[str] = []
    for token in tokens:
        token = token.strip("'")
        if len(token) < cfg.min_token_length or token in cfg.stopwords:
            continue
        result.append(cfg.stemmer(token) if cfg.stemmer else token)
    return result
