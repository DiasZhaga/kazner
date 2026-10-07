"""Word-level prediction contract: one CSV schema for every model.

Every model, encoder or generative, writes its test predictions as one row per gold word:

``sentence_id, word_idx, word, gold, pred`` (+ optional ``truncated``)

``truncated`` is ``1`` for words the model never saw (cut off at ``max_len``); such words
carry ``pred = O``. A single evaluator (:mod:`kazner.evaluate`) scores every file.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from kazner.data import read_iob2

COLUMNS = ("sentence_id", "word_idx", "word", "gold", "pred")
OPTIONAL_COLUMNS = ("truncated",)
TRUNCATED_PRED = "O"
_IOB2_LABEL = re.compile(r"^(O|[BI]-\S+)$")


class ContractError(ValueError):
    """A prediction file violates the contract."""


@dataclass(frozen=True)
class Predictions:
    """Word-level gold and predicted labels, one inner list per sentence."""

    sentence_ids: list[int]
    words: list[list[str]]
    gold: list[list[str]]
    pred: list[list[str]]
    truncated: list[list[bool]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.truncated:
            object.__setattr__(self, "truncated", [[False] * len(w) for w in self.words])
        sizes = {len(self.sentence_ids), len(self.words), len(self.gold), len(self.pred)}
        sizes.add(len(self.truncated))
        if len(sizes) != 1:
            raise ContractError("sentence_ids, words, gold, pred and truncated differ in length")
        for sid, *rows in zip(
            self.sentence_ids, self.words, self.gold, self.pred, self.truncated, strict=True
        ):
            if len({len(r) for r in rows}) != 1:
                raise ContractError(f"sentence {sid}: words, gold, pred differ in length")

    @property
    def n_words(self) -> int:
        return sum(len(w) for w in self.words)

    @property
    def n_truncated_words(self) -> int:
        return sum(sum(t) for t in self.truncated)


def write_predictions(path: str | Path, predictions: Predictions) -> Path:
    """Write ``predictions`` as a contract CSV (UTF-8)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([*COLUMNS, *OPTIONAL_COLUMNS])
        p = predictions
        for sid, words, gold, pred, trunc in zip(
            p.sentence_ids, p.words, p.gold, p.pred, p.truncated, strict=True
        ):
            for idx, row in enumerate(zip(words, gold, pred, trunc, strict=True)):
                word, g, pr, t = row
                writer.writerow([sid, idx, word, g, pr, int(t)])
    return path


def read_predictions(path: str | Path, labels: Collection[str] | None = None) -> Predictions:
    """Read and validate a contract CSV.

    Every value is read as a string (a token such as ``NA`` stays a token). ``labels`` is the
    known label set; without it, labels must at least look like IOB2 (``O``, ``B-X``, ``I-X``).
    """
    path = Path(path)
    known = set(labels) if labels is not None else None
    sentences: dict[int, dict[int, tuple[str, str, str, bool]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        missing = [c for c in COLUMNS if c not in header]
        if missing:
            raise ContractError(f"{path}: missing column(s): {', '.join(missing)}")
        unknown = [c for c in header if c not in COLUMNS and c not in OPTIONAL_COLUMNS]
        if unknown:
            raise ContractError(f"{path}: unknown column(s): {', '.join(unknown)}")
        for row in reader:
            where = f"{path}:{reader.line_num}"
            if None in row or any(row[c] in (None, "") for c in COLUMNS):
                raise ContractError(f"{where}: empty value or wrong number of fields")
            try:
                sid, idx = int(row["sentence_id"]), int(row["word_idx"])
            except ValueError:
                raise ContractError(f"{where}: sentence_id and word_idx must be integers") from None
            for column in ("gold", "pred"):
                label = row[column]
                if (known is not None and label not in known) or (
                    known is None and not _IOB2_LABEL.match(label)
                ):
                    raise ContractError(f"{where}: unknown {column} label {label!r}")
            truncated = row.get("truncated") or "0"
            if truncated not in ("0", "1"):
                raise ContractError(f"{where}: truncated must be 0 or 1, got {truncated!r}")
            words = sentences.setdefault(sid, {})
            if idx in words:
                raise ContractError(f"{where}: duplicate word_idx {idx} in sentence {sid}")
            words[idx] = (row["word"], row["gold"], row["pred"], truncated == "1")

    ids, words, gold, pred, trunc = [], [], [], [], []
    for sid, rows in sentences.items():
        if sorted(rows) != list(range(len(rows))):
            raise ContractError(f"{path}: sentence {sid}: word_idx must run 0..{len(rows) - 1}")
        ordered = [rows[i] for i in range(len(rows))]
        ids.append(sid)
        words.append([r[0] for r in ordered])
        gold.append([r[1] for r in ordered])
        pred.append([r[2] for r in ordered])
        trunc.append([r[3] for r in ordered])
    return Predictions(ids, words, gold, pred, trunc)


# --- pilot prediction files ------------------------------------------------------------

#: ``run_cross_lingual_twnertc_kaznerd.py`` (no word index; rows are in word order).
PILOT_CROSS_LINGUAL = ("sentence_id", "token", "true_label", "pred_label")
#: ``train_ner_mbert.py``.
PILOT_MBERT = ("sent_id", "token_id", "token", "gold", "pred")


def convert_pilot_predictions(path: str | Path, gold_path: str | Path | None = None) -> Predictions:
    """Convert a pilot ``test_predictions*.csv`` to the contract.

    The pilot files contain only the words the model saw. With ``gold_path`` (the IOB2 test
    split), words lost to truncation are appended with ``pred = O`` and ``truncated = 1``.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = tuple(reader.fieldnames or ())
        if header == PILOT_CROSS_LINGUAL:
            sid_col, word_col, gold_col, pred_col, idx_col = (
                "sentence_id", "token", "true_label", "pred_label", None,
            )  # fmt: skip
        elif header == PILOT_MBERT:
            sid_col, word_col, gold_col, pred_col, idx_col = (
                "sent_id", "token", "gold", "pred", "token_id",
            )  # fmt: skip
        else:
            raise ContractError(f"{path}: not a known pilot prediction format: {header}")
        sentences: dict[int, list[tuple[str, str, str]]] = {}
        for row in reader:
            rows = sentences.setdefault(int(row[sid_col]), [])
            if idx_col is not None and int(row[idx_col]) != len(rows):
                raise ContractError(f"{path}:{reader.line_num}: unexpected {idx_col}")
            rows.append((row[word_col], row[gold_col], row[pred_col]))

    ids = sorted(sentences)
    words = [[r[0] for r in sentences[s]] for s in ids]
    gold = [[r[1] for r in sentences[s]] for s in ids]
    pred = [[r[2] for r in sentences[s]] for s in ids]
    truncated = [[False] * len(w) for w in words]
    if gold_path is not None:
        _append_truncated(ids, words, gold, pred, truncated, *read_iob2(gold_path))
    return Predictions(ids, words, gold, pred, truncated)


def _append_truncated(
    ids: Sequence[int],
    words: list[list[str]],
    gold: list[list[str]],
    pred: list[list[str]],
    truncated: list[list[bool]],
    gold_words: list[list[str]],
    gold_labels: list[list[str]],
) -> None:
    if list(ids) != list(range(len(gold_words))):
        raise ContractError(
            f"prediction file has {len(ids)} sentences, gold file has {len(gold_words)}"
        )
    for i, (all_words, all_labels) in enumerate(zip(gold_words, gold_labels, strict=True)):
        seen = len(words[i])
        if all_words[:seen] != words[i] or all_labels[:seen] != gold[i]:
            raise ContractError(f"sentence {i}: predictions do not match the gold file")
        lost = len(all_words) - seen
        words[i] += all_words[seen:]
        gold[i] += all_labels[seen:]
        pred[i] += [TRUNCATED_PRED] * lost
        truncated[i] += [True] * lost
