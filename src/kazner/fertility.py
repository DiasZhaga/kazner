"""Tokenizer fertility: how many subwords a tokenizer produces per word.

Definitions follow the pilot (``ner-project/analysis/tokenizer_fertility.py``): every word
is tokenized on its own with ``tokenizer.tokenize(word)``; an empty result counts as one
unknown token; a word is unknown when its tokenization is exactly ``[unk_token]``.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kazner.data import KAZNERD_SPLITS, read_splits


@dataclass
class FertilityStats:
    """Running counts over words; :meth:`summary` turns them into rates."""

    words: int = 0
    subwords: int = 0
    unknown_words: int = 0
    histogram: Counter[int] = field(default_factory=Counter)

    def add(self, n_subwords: int, unknown: bool) -> None:
        self.words += 1
        self.subwords += n_subwords
        self.unknown_words += unknown
        self.histogram[n_subwords] += 1

    def merge(self, other: FertilityStats) -> None:
        self.words += other.words
        self.subwords += other.subwords
        self.unknown_words += other.unknown_words
        self.histogram.update(other.histogram)

    def summary(self) -> dict[str, Any]:
        if self.words == 0:
            raise ValueError("no words to analyse")
        single = self.histogram[1]
        split = self.words - single
        split_subwords = self.subwords - single
        mean = self.subwords / self.words
        mean_sq = sum(n * n * c for n, c in self.histogram.items()) / self.words
        return {
            "words": self.words,
            "subword_tokens": self.subwords,
            "fertility": mean,
            "single_token_words": single,
            "split_words": split,
            "single_token_rate": single / self.words,
            "split_word_rate": split / self.words,
            "mean_subwords_of_split_words": split_subwords / split if split else 0.0,
            "median_subwords_per_word": self._median(),
            "std_subwords_per_word": math.sqrt(max(mean_sq - mean * mean, 0.0)),
            "max_subwords_per_word": max(self.histogram),
            "unknown_words": self.unknown_words,
            "unk_rate": self.unknown_words / self.words,
        }

    def _median(self) -> float:
        low_rank, high_rank = (self.words - 1) // 2, self.words // 2
        low = high = None
        seen = 0
        for value in sorted(self.histogram):
            seen += self.histogram[value]
            if low is None and seen > low_rank:
                low = value
            if seen > high_rank:
                high = value
                break
        return (low + high) / 2.0


class WordTokenizer:
    """Tokenizes single words with a cache; returns ``(n_subwords, is_unknown)``."""

    def __init__(self, tokenizer: Any, lowercase: bool = False) -> None:
        self.tokenizer = tokenizer
        self.lowercase = lowercase
        self.unk_token = tokenizer.unk_token or "[UNK]"
        self._cache: dict[str, tuple[int, bool]] = {}

    def __call__(self, word: str) -> tuple[int, bool]:
        key = word.lower() if self.lowercase else word
        cached = self._cache.get(key)
        if cached is None:
            pieces = self.tokenizer.tokenize(key) or [self.unk_token]
            cached = (len(pieces), pieces == [self.unk_token])
            self._cache[key] = cached
        return cached


def word_stats(words: Iterable[str], word_tokenizer: WordTokenizer) -> FertilityStats:
    stats = FertilityStats()
    for word in words:
        stats.add(*word_tokenizer(word))
    return stats


def analyse_corpus(
    data_dir: str | Path,
    tokenizer: Any,
    *,
    tokenizer_name: str | None = None,
    lowercase: bool = False,
    splits: Mapping[str, str] = KAZNERD_SPLITS,
) -> dict[str, Any]:
    """Fertility per split and overall for an IOB2 corpus in ``data_dir``."""
    word_tokenizer = WordTokenizer(tokenizer, lowercase=lowercase)
    overall = FertilityStats()
    results: dict[str, Any] = {}
    for split, (tokens, _labels) in read_splits(data_dir, splits).items():
        stats = word_stats((w for sentence in tokens for w in sentence), word_tokenizer)
        overall.merge(stats)
        results[split] = stats.summary()
    results["overall"] = overall.summary()
    return {
        "tokenizer": tokenizer_name or getattr(tokenizer, "name_or_path", None),
        "lowercase": lowercase,
        "data_dir": str(data_dir),
        "splits": results,
    }


def write_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path
