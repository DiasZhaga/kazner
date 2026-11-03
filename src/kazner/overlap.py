"""Subword overlap between a source and a target corpus under one tokenizer (#59).

``kazner overlap``. Cross-lingual transfer through a shared encoder can reuse the embeddings
only of pieces that occur in both languages. TWNERTC is Turkish in Latin script and KazNERD is
Kazakh in Cyrillic, so they should share few of mBERT's pieces. For the target corpus the
report gives, over all words and over the words inside gold entities:

- ``piece_types_shared``: the share of its distinct pieces that occur in the source;
- ``piece_tokens_covered``: the share of its piece occurrences whose piece occurs in the source;
- ``words_fully_covered``: the share of its words all of whose pieces occur in the source;
- ``word_types_in_source``: the share of its distinct words that occur in the source as words.

Words are tokenized one at a time with ``tokenizer.tokenize(word)``, as in ``kazner fertility``
(for mBERT these are the pieces seen in context). The unknown token is never shared: a target
word with an unknown piece is not covered.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from kazner.data import Sentences

OUTSIDE = "O"


def _pieces(tokenizer: Any, words: Iterable[str]) -> dict[str, tuple[str, ...]]:
    unk = tokenizer.unk_token
    return {w: tuple(tokenizer.tokenize(w)) or (unk,) for w in words}


def _shares(
    counts: Counter[str], pieces: dict[str, tuple[str, ...]], shared: set[str], words: set[str]
) -> dict[str, Any]:
    piece_counts: Counter[str] = Counter()
    covered_words = 0
    for word, n in counts.items():
        for piece in pieces[word]:
            piece_counts[piece] += n
        covered_words += n * all(p in shared for p in pieces[word])
    n_words = sum(counts.values())
    n_tokens = sum(piece_counts.values())

    def share(part: int, whole: int) -> float | None:
        return part / whole if whole else None

    return {
        "words": n_words,
        "word_types": len(counts),
        "piece_tokens": n_tokens,
        "piece_types": len(piece_counts),
        "piece_types_shared": share(sum(p in shared for p in piece_counts), len(piece_counts)),
        "piece_tokens_covered": share(
            sum(n for p, n in piece_counts.items() if p in shared), n_tokens
        ),
        "words_fully_covered": share(covered_words, n_words),
        "word_types_in_source": share(sum(w in words for w in counts), len(counts)),
    }


def piece_overlap(
    tokenizer: Any,
    source_words: Sentences,
    target_words: Sentences,
    target_labels: Sentences | None = None,
) -> dict[str, Any]:
    """Overlap of the target corpus with the source corpus (module docstring)."""
    source_types = {w for s in source_words for w in s}
    source_pieces = _pieces(tokenizer, source_types)
    shared = {p for ps in source_pieces.values() for p in ps} - {tokenizer.unk_token}

    all_counts = Counter(w for s in target_words for w in s)
    entity_counts: Counter[str] = Counter()
    if target_labels is not None:
        for words, tags in zip(target_words, target_labels, strict=True):
            entity_counts.update(w for w, t in zip(words, tags, strict=True) if t != OUTSIDE)
    pieces = _pieces(tokenizer, all_counts)

    report = {
        "source": {
            "sentences": len(source_words),
            "words": sum(len(s) for s in source_words),
            "word_types": len(source_types),
            "piece_types": len(shared),
        },
        "all_words": _shares(all_counts, pieces, shared, source_types),
    }
    if target_labels is not None:
        report["entity_words"] = _shares(entity_counts, pieces, shared, source_types)
    return report


def format_overlap(report: dict[str, Any]) -> str:
    lines = [
        f"source: {report['source']['sentences']} sentences, {report['source']['words']} words, "
        f"{report['source']['piece_types']} distinct pieces"
    ]
    for key in ("all_words", "entity_words"):
        if key not in report:
            continue
        r = report[key]
        lines.append(
            f"target {key.replace('_', ' ')}: {r['words']} words, {r['piece_types']} distinct "
            f"pieces; shared piece types {_pct(r['piece_types_shared'])}, covered piece tokens "
            f"{_pct(r['piece_tokens_covered'])}, fully covered words "
            f"{_pct(r['words_fully_covered'])}, word types in source "
            f"{_pct(r['word_types_in_source'])}"
        )
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"
