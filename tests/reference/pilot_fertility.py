"""Reference oracle: fertility code copied VERBATIM from the pilot.

Source: ner-project @ b6ee209 — analysis/tokenizer_fertility.py (RunningStats,
TokenizerAnalyzer). Tests replace ``TokenizerAnalyzer._load_tokenizer`` to run offline.
Do not edit.
"""

# ruff: noqa
import importlib.metadata
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict

from transformers import AutoTokenizer


@dataclass
class RunningStats:
    total_words: int = 0
    total_subword_tokens: int = 0
    single_token_words: int = 0
    split_words: int = 0
    split_subword_total: int = 0
    maximum_subwords_per_word: int = 0
    unknown_token_words: int = 0
    malformed_rows: int = 0
    sum_of_squares: int = 0
    subword_histogram: Counter = field(default_factory=Counter)

    def update(self, subword_count: int, is_unknown: bool) -> None:
        self.total_words += 1
        self.total_subword_tokens += subword_count
        self.sum_of_squares += subword_count * subword_count
        self.subword_histogram[subword_count] += 1
        self.maximum_subwords_per_word = max(self.maximum_subwords_per_word, subword_count)

        if subword_count == 1:
            self.single_token_words += 1
        else:
            self.split_words += 1
            self.split_subword_total += subword_count

        if is_unknown:
            self.unknown_token_words += 1

    def add_malformed(self) -> None:
        self.malformed_rows += 1

    def to_result_row(self, split_name: str, model_name: str) -> Dict[str, object]:
        if self.total_words <= 0:
            raise ValueError(f"Split '{split_name}' has no valid words to analyze.")

        fertility = self.total_subword_tokens / self.total_words
        single_token_retention_rate = self.single_token_words / self.total_words
        split_word_rate = self.split_words / self.total_words
        mean_subwords_for_split_words = (
            self.split_subword_total / self.split_words if self.split_words else 0.0
        )
        median_subwords_per_word = self._median_from_histogram()
        mean_subwords = fertility
        variance = (self.sum_of_squares / self.total_words) - (mean_subwords * mean_subwords)
        std_subwords_per_word = math.sqrt(max(variance, 0.0))
        unknown_token_rate = self.unknown_token_words / self.total_words

        return {
            "split": split_name,
            "model_name": model_name,
            "total_words": self.total_words,
            "total_subword_tokens": self.total_subword_tokens,
            "fertility": fertility,
            "single_token_words": self.single_token_words,
            "split_words": self.split_words,
            "single_token_retention_rate": single_token_retention_rate,
            "split_word_rate": split_word_rate,
            "mean_subwords_for_split_words": mean_subwords_for_split_words,
            "median_subwords_per_word": median_subwords_per_word,
            "std_subwords_per_word": std_subwords_per_word,
            "maximum_subwords_per_word": self.maximum_subwords_per_word,
            "unknown_token_words": self.unknown_token_words,
            "unknown_token_rate": unknown_token_rate,
            "malformed_rows": self.malformed_rows,
        }

    def _median_from_histogram(self) -> float:
        if self.total_words <= 0:
            return 0.0

        midpoint_low = (self.total_words - 1) // 2
        midpoint_high = self.total_words // 2

        cumulative = 0
        low_value = None
        high_value = None

        for subword_count in sorted(self.subword_histogram):
            cumulative += self.subword_histogram[subword_count]
            if low_value is None and cumulative > midpoint_low:
                low_value = subword_count
            if high_value is None and cumulative > midpoint_high:
                high_value = subword_count
                break

        if low_value is None or high_value is None:
            raise ValueError("Could not compute median from histogram.")
        return (low_value + high_value) / 2.0


class TokenizerAnalyzer:
    def __init__(self, model_name: str, lowercase: bool):
        self.model_name = model_name
        self.lowercase = lowercase
        self.tokenizer = self._load_tokenizer(model_name)
        self.cache: Dict[str, tuple[int, list[str], bool]] = {}
        self.unk_token = self.tokenizer.unk_token or "[UNK]"
        self.version_info = {
            "transformers": self._safe_version("transformers"),
            "tokenizers": self._safe_version("tokenizers"),
        }

    def _load_tokenizer(self, model_name: str):
        try:
            return AutoTokenizer.from_pretrained(model_name, use_fast=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load tokenizer '{model_name}'. "
                "Check the model name/path and local Hugging Face availability."
            ) from exc

    @staticmethod
    def _safe_version(package_name: str) -> str:
        try:
            return importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            return "not-installed"

    def normalize_word(self, word: str) -> str:
        return word.lower() if self.lowercase else word

    def analyze_word(self, word: str) -> tuple[int, list[str], bool]:
        normalized_word = self.normalize_word(word)
        cached = self.cache.get(normalized_word)
        if cached is not None:
            return cached

        pieces = self.tokenizer.tokenize(normalized_word)
        if not pieces:
            pieces = [self.unk_token]

        subword_count = len(pieces)
        is_unknown = pieces == [self.unk_token]
        result = (subword_count, pieces, is_unknown)
        self.cache[normalized_word] = result
        return result
