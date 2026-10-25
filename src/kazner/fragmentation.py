"""Fragmentation vs NER errors on word-level contract files (experiment B2).

Measures how test errors depend on how strongly a fixed *reference* tokenizer splits the
words, with the words' rarity in the training subsample as the main confounder.

* Word table: pieces under the reference tokenizer (isolated tokenization, as in
  ``kazner.fertility``; at least one piece), whether a piece is ``[UNK]``, characters,
  frequency of the word form in the training subsample, gold entity type and B/I position.
* Entity table: every gold entity (seqeval default mode, as the evaluator) with its mean and
  maximum pieces per word, whether one of its words contains ``[UNK]``, and whether the same
  surface string with the same type is an entity in the training subsample ("seen").

Both tables depend only on the data, the training subsample and the reference tokenizer,
never on a run, so runs of different models are scored on the same words and entities. Runs
are grouped by config fields, as in ``kazner aggregate``. Per group the report gives error
rates by pieces (raw and within frequency strata), exact-span recall by pieces per word for
seen and unseen entities, and a logistic regression of entity-word errors on pieces with
frequency, length, B/I and entity type as controls.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from seqeval.metrics.sequence_labeling import get_entities

from kazner.aggregate import config_field
from kazner.config import data_fingerprint
from kazner.contract import Predictions, read_predictions
from kazner.data import read_splits, training_subsample

#: Error tables bin pieces as 1, 2, 3, 4 and "5+".
PIECE_CAP = 5
#: The regression clips pieces here, so that a few extreme words do not drive the slope.
REGRESSION_PIECE_CAP = 8
FREQUENCY_BINS = (("unseen", 0, 0), ("1-4", 1, 4), ("5+", 5, math.inf))
#: Mean pieces per word of an entity: label and inclusive upper bound.
FERTILITY_BINS = (("1.0", 1.0), ("(1,2]", 2.0), ("(2,3]", 3.0), (">3", math.inf))
#: Strata with fewer words or entities than this are not reported.
MIN_CELL = 30
DEFAULT_BY = ("model", "method")

PieceCounter = Callable[[str], tuple[int, bool]]


@dataclass(frozen=True)
class Run:
    run_id: str
    config: dict[str, Any]
    predictions: Predictions

    @property
    def seed(self) -> int:
        return int(self.config["seed"])


def load_run(run_dir: str | Path) -> Run:
    """A run directory written by ``kazner train``: ``config.json`` + ``predictions.csv``."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    return Run(run_dir.name, config, read_predictions(run_dir / "predictions.csv"))


def piece_counter(tokenizer: Any) -> PieceCounter:
    """``word -> (pieces, has_unk)``, cached. An empty tokenization is one unknown piece."""
    unk = tokenizer.unk_token
    cache: dict[str, tuple[int, bool]] = {}

    def count(word: str) -> tuple[int, bool]:
        hit = cache.get(word)
        if hit is None:
            pieces = tokenizer.tokenize(word)
            hit = (max(1, len(pieces)), not pieces or unk in pieces)
            cache[word] = hit
        return hit

    return count


def frequency_bin(freq: int) -> str:
    return next(label for label, low, high in FREQUENCY_BINS if low <= freq <= high)


def fertility_bin(mean_pieces: float) -> str:
    return next(label for label, high in FERTILITY_BINS if mean_pieces <= high)


def piece_bin(pieces: int) -> str:
    return f"{PIECE_CAP}+" if pieces >= PIECE_CAP else str(pieces)


def _entities(tags: Sequence[str]) -> list[tuple[str, int, int]]:
    """``(type, start, end_exclusive)`` per entity, seqeval default mode."""
    return [(etype, start, end + 1) for etype, start, end in get_entities(list(tags))]


# --------------------------------------------------------------------------- tables


def word_table(
    test: Predictions, train_sentences: Sequence[Sequence[str]], pieces_of: PieceCounter
) -> pd.DataFrame:
    """One row per test word, in contract order."""
    freq = Counter(word for sentence in train_sentences for word in sentence)
    rows = []
    for sid, words, tags in zip(test.sentence_ids, test.words, test.gold, strict=True):
        for i, (word, tag) in enumerate(zip(words, tags, strict=True)):
            pieces, unk = pieces_of(word)
            prefix, _, etype = tag.partition("-")
            rows.append(
                (sid, i, word, tag, pieces, unk, len(word), freq.get(word, 0), etype, prefix)
            )
    columns = ["sentence_id", "word_idx", "word", "gold", "pieces", "unk", "chars", "freq"]
    df = pd.DataFrame(rows, columns=[*columns, "type", "prefix"])
    df["is_entity"] = df["gold"] != "O"
    df["is_begin"] = df["prefix"] == "B"
    df["piece_bin"] = df["pieces"].map(piece_bin)
    df["freq_bin"] = df["freq"].map(frequency_bin)
    return df.drop(columns="prefix")


def entity_table(
    test: Predictions,
    train_sentences: Sequence[Sequence[str]],
    train_labels: Sequence[Sequence[str]],
    pieces_of: PieceCounter,
) -> pd.DataFrame:
    """One row per gold test entity with its fragmentation and the "seen" flag."""
    seen = Counter()
    for words, tags in zip(train_sentences, train_labels, strict=True):
        for etype, start, end in _entities(tags):
            seen[(" ".join(words[start:end]), etype)] += 1
    rows = []
    for sid, words, tags in zip(test.sentence_ids, test.words, test.gold, strict=True):
        for etype, start, end in _entities(tags):
            counts = [pieces_of(word) for word in words[start:end]]
            pieces = [n for n, _ in counts]
            text = " ".join(words[start:end])
            mean = sum(pieces) / len(pieces)
            rows.append(
                {
                    "sentence_id": sid,
                    "start": start,
                    "end": end,
                    "type": etype,
                    "text": text,
                    "n_words": end - start,
                    "mean_pieces": mean,
                    "max_pieces": max(pieces),
                    "has_unk": any(unk for _, unk in counts),
                    "seen": seen[(text, etype)] > 0,
                    "fertility_bin": fertility_bin(mean),
                }
            )
    return pd.DataFrame(rows)


def entity_hits(entities: pd.DataFrame, predictions: Predictions) -> np.ndarray:
    """``True`` where the exact gold span is predicted with the gold type."""
    by_sentence = dict(zip(predictions.sentence_ids, predictions.pred, strict=True))
    predicted: dict[int, set[tuple[str, int, int]]] = {}
    hits = np.zeros(len(entities), dtype=bool)
    keys = entities[["sentence_id", "type", "start", "end"]].itertuples(index=False)
    for k, (sid, etype, start, end) in enumerate(keys):
        if sid not in predicted:
            predicted[sid] = set(_entities(by_sentence[sid]))
        hits[k] = (etype, start, end) in predicted[sid]
    return hits


def word_errors(words: pd.DataFrame, predictions: Predictions) -> np.ndarray:
    """``True`` where the predicted label differs from the gold label (contract order)."""
    pred = np.array([label for sentence in predictions.pred for label in sentence])
    return pred != words["gold"].to_numpy()


# --------------------------------------------------------------------------- statistics


def _mean_sd(values: Sequence[float]) -> tuple[float, float | None]:
    mean = float(np.mean(values))
    return mean, (float(np.std(values, ddof=1)) if len(values) > 1 else None)


def error_by_pieces(words: pd.DataFrame, errors: np.ndarray) -> list[dict[str, Any]]:
    """Entity words: error rate by piece bin (mean ± sd over runs). ``errors``: runs × words."""
    entity = words["is_entity"].to_numpy()
    rows = []
    for label in sorted(words["piece_bin"].unique(), key=lambda b: int(b.rstrip("+"))):
        sel = entity & (words["piece_bin"] == label).to_numpy()
        if not sel.any():
            continue
        mean, sd = _mean_sd(errors[:, sel].mean(axis=1))
        unseen = float((words.loc[sel, "freq"] == 0).mean())
        rows.append({"pieces": label, "n": int(sel.sum()), "share_unseen": unseen,
                     "error_mean": mean, "error_sd": sd})  # fmt: skip
    return rows


def error_by_pieces_within_frequency(
    words: pd.DataFrame, errors: np.ndarray
) -> list[dict[str, Any]]:
    """Entity words: error rate by piece bin within training-frequency strata."""
    entity = words["is_entity"].to_numpy()
    rows = []
    for freq_label, _, _ in FREQUENCY_BINS:
        in_freq = entity & (words["freq_bin"] == freq_label).to_numpy()
        for label in sorted(words["piece_bin"].unique(), key=lambda b: int(b.rstrip("+"))):
            sel = in_freq & (words["piece_bin"] == label).to_numpy()
            if sel.sum() < MIN_CELL:
                continue
            mean, sd = _mean_sd(errors[:, sel].mean(axis=1))
            rows.append({"train_freq": freq_label, "pieces": label, "n": int(sel.sum()),
                         "error_mean": mean, "error_sd": sd})  # fmt: skip
    return rows


def recall_by_fertility(entities: pd.DataFrame, hits: np.ndarray) -> list[dict[str, Any]]:
    """Exact-span recall by mean pieces per word, seen and unseen entities. ``hits``: runs ×
    entities."""
    rows = []
    for seen in (True, False):
        for label, _ in FERTILITY_BINS:
            sel = ((entities["seen"] == seen) & (entities["fertility_bin"] == label)).to_numpy()
            if sel.sum() < MIN_CELL:
                continue
            mean, sd = _mean_sd(hits[:, sel].mean(axis=1))
            rows.append({"seen_in_train": seen, "pieces_per_word": label, "n": int(sel.sum()),
                         "recall_mean": mean, "recall_sd": sd})  # fmt: skip
    return rows


def design_matrix(words: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Intercept, pieces (clipped), log(1 + training frequency), characters, B vs I, and one
    dummy per entity type except the alphabetically first."""
    types = sorted(words["type"].unique())
    columns = {
        "intercept": np.ones(len(words)),
        "pieces": words["pieces"].clip(upper=REGRESSION_PIECE_CAP).to_numpy(float),
        "log1p_freq": np.log1p(words["freq"].to_numpy(float)),
        "chars": words["chars"].to_numpy(float),
        "is_begin": words["is_begin"].to_numpy(float),
    }
    for etype in types[1:]:
        columns[f"type={etype}"] = (words["type"] == etype).to_numpy(float)
    return np.column_stack(list(columns.values())), list(columns)


def fit_logistic(
    x: np.ndarray, y: np.ndarray, ridge: float = 1e-6, max_iter: int = 100, tol: float = 1e-10
) -> np.ndarray:
    """Maximum-likelihood logistic regression by Newton's method (IRLS).

    A tiny ridge keeps the optimum finite and the Hessian invertible when a category separates
    the outcome perfectly; it changes the estimates negligibly otherwise.
    """
    beta = np.zeros(x.shape[1])
    penalty = np.full(x.shape[1], ridge)
    for _ in range(max_iter):
        p = 0.5 * (1.0 + np.tanh(0.5 * (x @ beta)))  # numerically stable sigmoid
        gradient = x.T @ (y - p) - penalty * beta
        hessian = (x * (p * (1.0 - p))[:, None]).T @ x + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def odds_ratios(x: np.ndarray, names: Sequence[str], y: np.ndarray) -> dict[str, float | None]:
    """Odds ratios per extra piece, per e-fold of (1 + training frequency), per character;
    ``None`` when ``y`` does not vary (a run without errors, or with errors only)."""
    keys = {"OR_per_piece": "pieces", "OR_per_freq_x_e": "log1p_freq", "OR_per_char": "chars"}
    if y.min() == y.max():
        return dict.fromkeys(keys)
    beta = dict(zip(names, fit_logistic(x, y), strict=True))
    return {key: math.exp(beta[name]) for key, name in keys.items()}


def bootstrap_piece_or(
    x: np.ndarray,
    errors: np.ndarray,
    sentence_ids: np.ndarray,
    names: Sequence[str],
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    """95% percentile CI of the group's geometric-mean odds ratio per piece.

    Test sentences are resampled with replacement; in every replicate each run is refitted on
    the same resample, so the interval covers sampling of the test set, not seed variance.
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(sentence_ids, kind="stable")
    unique, starts = np.unique(sentence_ids[order], return_index=True)
    rows_of = np.split(order, starts[1:])
    piece = list(names).index("pieces")
    stats = []
    for _ in range(replicates):
        rows = np.concatenate([rows_of[i] for i in rng.integers(0, len(unique), len(unique))])
        betas = [fit_logistic(x[rows], run[rows])[piece] for run in errors]
        stats.append(math.exp(float(np.mean(betas))))
    low, high = np.percentile(stats, [2.5, 97.5])
    return float(low), float(high)


# --------------------------------------------------------------------------- analysis


def _check_runs(runs: Sequence[Run], data_dir: Path) -> tuple[float, int, int | None]:
    if not runs:
        raise ValueError("no runs given")
    subsamples = {
        (r.config["data"]["train_fraction"], r.config["data"]["subsample_seed"],
         r.config["data"].get("max_train_samples"))
        for r in runs
    }  # fmt: skip
    if len(subsamples) > 1:
        raise ValueError(
            f"runs were trained on different training subsamples: {sorted(subsamples)}"
        )
    fingerprint = data_fingerprint(data_dir)
    first = runs[0].predictions
    for r in runs:
        trained_on = r.config.get("data_fingerprint")
        if trained_on and trained_on != fingerprint:
            raise ValueError(f"run {r.run_id} was trained on other data than {data_dir}")
        p = r.predictions
        if p.sentence_ids != first.sentence_ids or p.words != first.words or p.gold != first.gold:
            raise ValueError(f"run {r.run_id} has another test set than run {runs[0].run_id}")
    return next(iter(subsamples))


def analyse(
    runs: Sequence[Run],
    data_dir: str | Path,
    tokenizer: Any,
    *,
    by: Sequence[str] = DEFAULT_BY,
    bootstrap: int = 200,
    seed: int = 0,
    tokenizer_info: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Report (JSON-ready) and the entity table with one ``hit_<run_id>`` column per run."""
    data_dir = Path(data_dir)
    fraction, subsample_seed, max_samples = _check_runs(runs, data_dir)
    train_tokens, train_labels = read_splits(data_dir)["train"]
    sub_tokens, sub_labels = training_subsample(
        train_tokens, train_labels, fraction, subsample_seed, max_samples
    )
    pieces_of = piece_counter(tokenizer)
    test = runs[0].predictions
    words = word_table(test, sub_tokens, pieces_of)
    entities = entity_table(test, sub_tokens, sub_labels, pieces_of)
    entity_words = words[words["is_entity"]].reset_index(drop=True)
    x, names = design_matrix(entity_words)
    entity_mask = words["is_entity"].to_numpy()
    sentence_ids = entity_words["sentence_id"].to_numpy()

    groups: dict[tuple, list[Run]] = {}
    for run in runs:
        groups.setdefault(tuple(config_field(run.config, key) for key in by), []).append(run)

    report_groups: dict[str, Any] = {}
    for key, members in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        members = sorted(members, key=lambda r: r.seed)
        errors = np.vstack([word_errors(words, r.predictions) for r in members])
        hits = np.vstack([entity_hits(entities, r.predictions) for r in members])
        for r, row in zip(members, hits, strict=True):
            entities[f"hit_{r.run_id}"] = row
        y = errors[:, entity_mask].astype(float)
        per_run = [
            {"run_id": r.run_id, **odds_ratios(x, names, yr)}
            for r, yr in zip(members, y, strict=True)
        ]
        piece_ors = [d["OR_per_piece"] for d in per_run]
        geomean, ci = None, None
        if None not in piece_ors:  # every run has both errors and correct entity words
            geomean = math.exp(float(np.mean(np.log(piece_ors))))
            ci = list(bootstrap_piece_or(x, y, sentence_ids, names, bootstrap, seed))
        label = ", ".join(f"{k}={v}" for k, v in zip(by, key, strict=True))
        report_groups[label] = {
            "by": dict(zip(by, key, strict=True)),
            "run_ids": [r.run_id for r in members],
            "seeds": [r.seed for r in members],
            "word_error_by_pieces": error_by_pieces(words, errors),
            "word_error_by_pieces_within_frequency": error_by_pieces_within_frequency(
                words, errors
            ),
            "logistic_regression": {
                "per_run": per_run,
                "OR_per_piece_geomean": geomean,
                "OR_per_piece_CI95": ci,
            },
            "entity_recall_by_fertility": recall_by_fertility(entities, hits),
            "truncated_words": [r.predictions.n_truncated_words for r in members],
        }

    report = {
        "reference_tokenizer": dict(tokenizer_info or {}),
        "data": {
            "dir": str(data_dir),
            "fingerprint": data_fingerprint(data_dir),
            "train_fraction": fraction,
            "subsample_seed": subsample_seed,
            "max_train_samples": max_samples,
            "train_sentences": len(sub_tokens),
        },
        "n_test_words": int(len(words)),
        "n_entity_words": int(entity_mask.sum()),
        "n_gold_entities": int(len(entities)),
        "n_entities_with_unk": int(entities["has_unk"].sum()),
        "regression": {
            "outcome": "error on an entity word (pred != gold)",
            "predictors": names[1:],
            "piece_cap": REGRESSION_PIECE_CAP,
        },
        "bootstrap": {"unit": "test sentence", "replicates": bootstrap, "seed": seed},
        "groups": report_groups,
    }
    return report, entities


def write_outputs(out_dir: str | Path, report: Mapping[str, Any], entities: pd.DataFrame) -> Path:
    """``fragmentation.json`` and ``entities.csv`` (the fixed entity subsets + hits)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    (out_dir / "fragmentation.json").write_text(text, encoding="utf-8", newline="\n")
    entities.to_csv(out_dir / "entities.csv", index=False, encoding="utf-8", lineterminator="\n")
    return out_dir


def format_report(report: Mapping[str, Any]) -> str:
    """Terminal summary: the tables of every group."""

    def table(rows: Sequence[Mapping[str, Any]]) -> str:
        frame = pd.DataFrame(rows)
        return frame.to_string(index=False, float_format=lambda v: f"{v:.3f}") if rows else "-"

    lines = [
        f"test words {report['n_test_words']}, entity words {report['n_entity_words']}, "
        f"gold entities {report['n_gold_entities']} ({report['n_entities_with_unk']} with [UNK])"
    ]

    def number(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    for label, g in report["groups"].items():
        lr = g["logistic_regression"]
        ors = ", ".join(number(d["OR_per_piece"]) for d in lr["per_run"])
        low, high = lr["OR_per_piece_CI95"] or (None, None)
        lines += [
            f"\n=== {label} (seeds {g['seeds']}) ===",
            "Entity words: error rate by pieces",
            table(g["word_error_by_pieces"]),
            "Within training-frequency strata",
            table(g["word_error_by_pieces_within_frequency"]),
            f"Odds ratio per extra piece: {ors}; geometric mean "
            f"{number(lr['OR_per_piece_geomean'])}, 95% CI [{number(low)}, {number(high)}]",
            "Exact-span recall by pieces per word",
            table(g["entity_recall_by_fertility"]),
        ]
    return "\n".join(lines)
