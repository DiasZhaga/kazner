"""Unlabeled Kazakh text for continued pre-training (#34).

A seeded, size-bounded sample of a parquet corpus (FineWeb-2 ``kaz_Cyrl``), cleaned line by
line. Documents are split into lines (paragraphs). A line is kept if, after Unicode NFC and
whitespace normalisation,

1. it has between :data:`MIN_WORDS` and :data:`MAX_WORDS` whitespace-separated words;
2. it is Kazakh: at least :data:`MIN_CYRILLIC_SHARE` of its letters are Cyrillic, and it
   contains a letter specific to Kazakh (:data:`KAZAKH_LETTERS`). This drops Russian, English
   and mostly Latin lines;
3. it is not an exact duplicate of a line kept before;
4. it contains no KazNERD sentence (any split) of at least :data:`MIN_MATCH_WORDS` words.
   Lines and sentences are compared as case-folded word sequences without punctuation, so
   KazNERD's split punctuation (``барды .``) still matches running text (``барды.``).

Row groups of the parquet files are visited in an order fixed by a seed and read one at a
time until the target number of words is reached. Only the row groups needed are
downloaded, and the sample still spreads over the whole corpus rather than its first rows.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, BinaryIO

import numpy as np

MIN_WORDS = 5
#: Longer "lines" are mostly lists, tables and menus.
MAX_WORDS = 500
MIN_CYRILLIC_SHARE = 0.5
KAZAKH_LETTERS = frozenset("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
#: Shorter KazNERD sentences are too generic to be evidence of overlap ("Рахмет .").
MIN_MATCH_WORDS = 6
#: Order of the counters in the manifest: every line is counted once, at the first rule it fails.
STAGES = ("lines", "too_short", "too_long", "not_kazakh", "duplicate", "kaznerd", "kept")

_WORD = re.compile(r"\w+")

class CountingReader:
    """Read-only file wrapper that counts the bytes read through it (#42).

    A parquet reader seeks and reads exact ranges; with ``cache_type="none"``
    on the Hub filesystem nothing else is fetched, so the count is the number
    of bytes downloaded."""

    def __init__(self, raw: BinaryIO) -> None:
        self._raw = raw
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        data = self._raw.read(size)
        self.bytes_read += len(data)
        return data

    def readinto(self, buffer: Any) -> int:
        count = self._raw.readinto(buffer)
        self.bytes_read += count or 0
        return count or 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)




def normalize_line(line: str) -> str:
    """Unicode NFC, whitespace runs collapsed to one space, stripped."""
    return " ".join(unicodedata.normalize("NFC", line).split())


def is_kazakh(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    if not letters:
        return False
    cyrillic = sum("Ѐ" <= c <= "ӿ" for c in letters)
    return cyrillic >= MIN_CYRILLIC_SHARE * len(letters) and any(c in KAZAKH_LETTERS for c in line)


def match_words(text: str) -> tuple[str, ...]:
    """Case-folded words without punctuation: the form in which overlap is checked."""
    return tuple(_WORD.findall(text.casefold()))


class SentenceIndex:
    """Finds any indexed sentence of at least ``min_words`` words inside a line."""

    def __init__(self, sentences: Iterable[Sequence[str]], min_words: int = MIN_MATCH_WORDS):
        self.min_words = min_words
        self._by_prefix: dict[tuple[str, ...], set[tuple[str, ...]]] = {}
        for tokens in sentences:
            words = match_words(" ".join(tokens))
            if len(words) >= min_words:
                self._by_prefix.setdefault(words[:min_words], set()).add(words)
        self.size = sum(len(group) for group in self._by_prefix.values())

    def contains(self, words: Sequence[str]) -> bool:
        words = tuple(words)
        for i in range(len(words) - self.min_words + 1):
            for sentence in self._by_prefix.get(words[i : i + self.min_words], ()):
                if words[i : i + len(sentence)] == sentence:
                    return True
        return False


class CorpusBuilder:
    """Applies the line rules and writes kept lines (UTF-8, LF) until ``target_words``."""

    def __init__(self, index: SentenceIndex, target_words: int, out: BinaryIO) -> None:
        self.index = index
        self.target_words = target_words
        self.out = out
        self.counts: Counter[str] = Counter(dict.fromkeys(STAGES, 0))
        self.documents = 0
        self.words = 0
        self._seen: set[bytes] = set()
        self._sha256 = hashlib.sha256()

    @property
    def done(self) -> bool:
        return self.words >= self.target_words

    @property
    def sha256(self) -> str:
        return self._sha256.hexdigest()

    def _rule(self, line: str, n_words: int) -> str:
        if n_words < MIN_WORDS:
            return "too_short"
        if n_words > MAX_WORDS:
            return "too_long"
        if not is_kazakh(line):
            return "not_kazakh"
        key = hashlib.blake2b(line.encode("utf-8"), digest_size=16).digest()
        if key in self._seen:
            return "duplicate"
        self._seen.add(key)
        if self.index.contains(match_words(line)):
            return "kaznerd"
        return "kept"

    def add_document(self, text: str) -> None:
        """Process the lines of one document; stops as soon as the target is reached."""
        self.documents += 1
        for raw in text.splitlines():
            if self.done:
                return
            line = normalize_line(raw)
            if not line:
                continue
            n_words = len(line.split())
            self.counts["lines"] += 1
            rule = self._rule(line, n_words)
            self.counts[rule] += 1
            if rule == "kept":
                data = (line + "\n").encode("utf-8")
                self.out.write(data)
                self._sha256.update(data)
                self.words += n_words


def seeded_row_groups(files: Sequence[Any], seed: int) -> list[tuple[int, int]]:
    """Every ``(file index, row group)`` of the parquet files, in an order fixed by ``seed``."""
    groups = [(f, g) for f, parquet in enumerate(files) for g in range(parquet.num_row_groups)]
    return [groups[i] for i in np.random.default_rng(seed).permutation(len(groups))]


def iter_documents(
    files: Sequence[Any], seed: int, column: str = "text"
) -> Iterator[tuple[int, int, str]]:
    """``(file index, row group, text)`` for the documents of the seeded row groups, read one
    row group at a time (``pyarrow.parquet.ParquetFile`` or anything with the same methods)."""
    for f, g in seeded_row_groups(files, seed):
        table = files[f].read_row_group(g, columns=[column])
        for text in table.column(column).to_pylist():
            yield f, g, text


def build_corpus(
    files: Sequence[Any],
    names: Sequence[str],
    index: SentenceIndex,
    out: BinaryIO,
    *,
    target_words: int,
    seed: int,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Write the cleaned sample to ``out``; returns the manifest fields that describe it.
    ``progress(row groups started, words kept)`` is called whenever a row group starts."""
    builder = CorpusBuilder(index, target_words, out)
    used: list[dict[str, Any]] = []
    for f, g, text in iter_documents(files, seed):
        if not used or (used[-1]["file"], used[-1]["row_group"]) != (names[f], g):
            used.append({"file": names[f], "row_group": g, "documents": 0})
            if progress is not None:
                progress(len(used), builder.words)
        used[-1]["documents"] += 1
        builder.add_document(text)
        if builder.done:
            break
    return {
        "complete": builder.done,
        "target_words": target_words,
        "words": builder.words,
        "lines": builder.counts["kept"],
        "sha256": builder.sha256,
        "seed": seed,
        "documents": builder.documents,
        "row_groups_total": sum(parquet.num_row_groups for parquet in files),
        "row_groups_used": used,
        "line_counts": {stage: builder.counts[stage] for stage in STAGES},
        "rules": {
            "min_words": MIN_WORDS,
            "max_words": MAX_WORDS,
            "min_cyrillic_share": MIN_CYRILLIC_SHARE,
            "kazakh_letters": "".join(sorted(KAZAKH_LETTERS)),
            "min_match_words": index.min_words,
            "kaznerd_sentences_indexed": index.size,
        },
    }
