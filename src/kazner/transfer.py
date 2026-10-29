"""Transfer to another NER dataset through an explicit label mapping (#51): ``kazner transfer``.

A model trained on KazNERD predicts its 25 entity types; WikiANN annotates PER, LOC and ORG. A
mapping file assigns every KazNERD type to one target type or to ``O``
(``configs/mappings/kaznerd_to_wikiann.yaml``). Three scores per run separate the effect of the
label space from that of the domain:

1. ``kaznerd``: the KazNERD test split with all types (the run's ``predictions.csv``);
2. ``kaznerd_mapped``: the same predictions with gold and prediction both mapped;
3. ``<extra>``: the extra test set (``predictions_<name>.csv``, written for the config's
   ``extra_tests``), with the predictions mapped and the gold labels as published.

Mapping keeps the B-/I- prefix, so adjacent entities that map to the same type stay separate.
Scores come from :func:`kazner.evaluate.compute_metrics` (seqeval, default mode). Runs are
grouped by config fields and averaged over seeds. The seed-mean F1 of step 3 gets a 95%
percentile CI from a bootstrap over the extra set's sentences. A secondary table gives the
exact-span recall of the extra set's entities by mean pieces per word under a reference
tokenizer, in the bins of ``kazner fragmentation``.

The extra set's scores are also given by sentence kind (:func:`sentence_kind`). WikiANN ``kk``
contains Kazakh Wikipedia redirect lines and article first lines, labelled by other conventions
than running text; ``other_unique`` keeps the remaining sentences, each distinct one once. For a
set without such lines every sentence is ``other``.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from seqeval.metrics.sequence_labeling import get_entities

from kazner.aggregate import config_field
from kazner.config import ConfigError
from kazner.contract import Predictions, read_predictions
from kazner.evaluate import compute_metrics
from kazner.fragmentation import FERTILITY_BINS, fertility_bin

OUTSIDE = "O"
SOURCE_STEPS = ("kaznerd", "kaznerd_mapped")
MAPPING_KEYS = ("source", "target", "target_types", "mapping")
METRICS = ("precision", "recall", "f1")
CHUNK = 500  # bootstrap replicates drawn at once
#: Kazakh Wikipedia's redirect keyword (``#REDIRECT``): the line names the target article.
REDIRECT_WORD = "АЙДАУ"
#: An article's first line starts with what is left of the bold-title markup ``'''``.
BOLD_TITLE_MARKS = ("'", "''")
SENTENCE_KINDS = ("redirect", "bold_title", "other")

PieceCounter = Callable[[str], tuple[int, bool]]


# --------------------------------------------------------------------------- label mapping


@dataclass(frozen=True)
class LabelMapping:
    """Entity type of the source dataset -> entity type of the target dataset, or ``O``."""

    source: str
    target: str
    target_types: tuple[str, ...]
    types: Mapping[str, str]

    def check_covers(self, types: Iterable[str], where: str) -> None:
        missing = sorted(set(types) - set(self.types))
        if missing:
            raise ConfigError(f"mapping: {where} contains types it does not list: {missing}")

    def map_tags(self, tags: Sequence[str]) -> list[str]:
        """Map one IOB2 sentence; ``B-GPE I-GPE`` -> ``B-LOC I-LOC``, ``B-DATE`` -> ``O``."""
        out = []
        for tag in tags:
            if tag == OUTSIDE:
                out.append(tag)
                continue
            prefix, _, etype = tag.partition("-")
            if etype not in self.types:
                raise ConfigError(f"mapping: entity type {etype!r} is not listed")
            target = self.types[etype]
            out.append(OUTSIDE if target == OUTSIDE else f"{prefix}-{target}")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "target_types": list(self.target_types),
            "mapping": dict(self.types),
        }


def parse_mapping(raw: Any) -> LabelMapping:
    """Build and validate a :class:`LabelMapping` from a parsed YAML mapping."""
    if not isinstance(raw, dict):
        raise ConfigError(f"mapping: expected a mapping, got {raw!r}")
    unknown = [k for k in raw if k not in MAPPING_KEYS]
    if unknown:
        raise ConfigError(f"mapping: unknown key(s) {unknown}; expected {list(MAPPING_KEYS)}")
    missing = [k for k in MAPPING_KEYS if k not in raw]
    if missing:
        raise ConfigError(f"mapping: missing key(s) {missing}")
    for key in ("source", "target"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ConfigError(f"mapping.{key}: expected a non-empty string")
    targets = raw["target_types"]
    if (
        not isinstance(targets, list)
        or not targets
        or not all(isinstance(t, str) and t and t != OUTSIDE for t in targets)
        or len(set(targets)) != len(targets)
    ):
        raise ConfigError("mapping.target_types: expected a list of distinct entity types (not O)")
    types = raw["mapping"]
    if not isinstance(types, dict) or not types:
        raise ConfigError("mapping.mapping: expected a non-empty mapping of entity types")
    allowed = {*targets, OUTSIDE}
    for source, target in types.items():
        if not isinstance(source, str) or not source or "-" in source:
            raise ConfigError(f"mapping.mapping: invalid source type {source!r}")
        if target not in allowed:
            raise ConfigError(
                f"mapping.mapping.{source}: unknown target {target!r}; expected one of "
                f"{sorted(allowed)}"
            )
    return LabelMapping(raw["source"], raw["target"], tuple(targets), dict(types))


def load_mapping(path: str | Path) -> LabelMapping:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    try:
        return parse_mapping(raw)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from None


def entity_types(sentences: Iterable[Sequence[str]]) -> set[str]:
    return {tag.partition("-")[2] for tags in sentences for tag in tags if tag != OUTSIDE}


# --------------------------------------------------------------------------- runs and scores


@dataclass(frozen=True)
class TransferRun:
    run_id: str
    config: dict[str, Any]
    source: Predictions
    extra: Predictions


def load_transfer_run(run_dir: str | Path, extra: str) -> TransferRun:
    """A ``kazner train`` run with ``predictions.csv`` and ``predictions_<extra>.csv``."""
    run_dir = Path(run_dir)
    extra_path = run_dir / f"predictions_{extra}.csv"
    if not extra_path.is_file():
        raise FileNotFoundError(
            f"{run_dir}: no {extra_path.name}; was the run trained with extra_tests '{extra}'?"
        )
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    return TransferRun(
        run_dir.name,
        config,
        read_predictions(run_dir / "predictions.csv"),
        read_predictions(extra_path),
    )


def score_run(run: TransferRun, mapping: LabelMapping, extra: str) -> dict[str, Any]:
    """The three scores of one run (module docstring), each from ``compute_metrics``."""
    src, ext = run.source, run.extra
    predicted = entity_types(src.gold) | entity_types(src.pred) | entity_types(ext.pred)
    mapping.check_covers(predicted, f"run {run.run_id}")
    foreign = sorted(entity_types(ext.gold) - set(mapping.target_types))
    if foreign:
        raise ConfigError(
            f"mapping: the {extra} gold labels of run {run.run_id} have types {foreign} "
            f"outside target_types {list(mapping.target_types)}"
        )
    return {
        "kaznerd": compute_metrics(src.gold, src.pred),
        "kaznerd_mapped": compute_metrics(
            [mapping.map_tags(s) for s in src.gold], [mapping.map_tags(s) for s in src.pred]
        ),
        extra: compute_metrics(ext.gold, [mapping.map_tags(s) for s in ext.pred]),
    }


def entity_counts(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]]) -> np.ndarray:
    """Per sentence: correct, predicted and gold entities (exact span and type).

    Summed over sentences they give seqeval's micro scores: F1 = 2·correct / (pred + gold).
    """
    counts = np.zeros((len(gold), 3), dtype=np.int64)
    for i, (g, p) in enumerate(zip(gold, pred, strict=True)):
        gold_set, pred_set = set(get_entities(list(g))), set(get_entities(list(p)))
        counts[i] = (len(gold_set & pred_set), len(pred_set), len(gold_set))
    return counts


def bootstrap_mean_f1(
    counts: Sequence[np.ndarray], replicates: int = 1000, seed: int = 0
) -> list[float]:
    """95% percentile CI of the seed-mean F1; each replicate resamples sentences once."""
    stacked = np.stack(counts).astype(np.float64)  # runs x sentences x 3
    n = stacked.shape[1]
    rng = np.random.default_rng(seed)
    values = np.empty(replicates)
    for start in range(0, replicates, CHUNK):
        size = min(CHUNK, replicates - start)
        weights = rng.multinomial(n, np.full(n, 1 / n), size=size).astype(np.float64)
        sums = np.einsum("bn,rnk->brk", weights, stacked)
        denominator = sums[..., 1] + sums[..., 2]
        f1 = np.divide(2 * sums[..., 0], denominator, out=np.zeros_like(denominator),
                       where=denominator > 0)  # fmt: skip
        values[start : start + size] = f1.mean(axis=1)
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def sentence_kind(words: Sequence[str]) -> str:
    """``redirect``, ``bold_title`` (an article's first line) or ``other``."""
    if words and words[0] == REDIRECT_WORD:
        return "redirect"
    if words and words[0] in BOLD_TITLE_MARKS:
        return "bold_title"
    return "other"


def sentence_subsets(sentences: Sequence[Sequence[str]]) -> dict[str, np.ndarray]:
    """Boolean masks over the sentences: one per kind, and ``other_unique``.

    ``other_unique`` marks the first occurrence of each distinct ``other`` sentence.
    """
    kinds = [sentence_kind(words) for words in sentences]
    subsets = {kind: np.array([k == kind for k in kinds], dtype=bool) for kind in SENTENCE_KINDS}
    seen: set[tuple[str, ...]] = set()
    unique = np.zeros(len(sentences), dtype=bool)
    for i, (kind, words) in enumerate(zip(kinds, sentences, strict=True)):
        if kind == "other" and tuple(words) not in seen:
            seen.add(tuple(words))
            unique[i] = True
    subsets["other_unique"] = unique
    return subsets


def subset_scores(
    counts: Sequence[np.ndarray], mask: np.ndarray, bootstrap: int, seed: int
) -> dict[str, Any]:
    """Micro P/R/F1 (mean ± sd over runs) and the bootstrap CI on the sentences in ``mask``."""
    per_run = []
    for c in counts:
        correct, n_pred, n_gold = (int(x) for x in c[mask].sum(axis=0))
        per_run.append({
            "precision": correct / n_pred if n_pred else 0.0,
            "recall": correct / n_gold if n_gold else 0.0,
            "f1": 2 * correct / (n_pred + n_gold) if n_pred + n_gold else 0.0,
        })  # fmt: skip
    out: dict[str, Any] = {
        "sentences": int(mask.sum()),
        "gold_entities": int(counts[0][mask, 2].sum()),
    }
    for metric in METRICS:
        out[metric], out[f"{metric}_sd"] = _mean_sd([s[metric] for s in per_run])
    out["f1_ci95"] = bootstrap_mean_f1([c[mask] for c in counts], bootstrap, seed)
    return out


def recall_by_pieces(
    runs: Sequence[TransferRun],
    mapping: LabelMapping,
    pieces_of: PieceCounter,
    sentences: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Exact-span recall of the extra set's gold entities by mean pieces per word.

    ``sentences`` (a boolean mask) restricts the entities to some sentences.
    """
    first = runs[0].extra
    bins: list[str] = []
    keys: list[tuple[int, str, int, int]] = []
    for i, (words, gold) in enumerate(zip(first.words, first.gold, strict=True)):
        if sentences is not None and not sentences[i]:
            continue
        for etype, start, end in get_entities(list(gold)):
            pieces = [pieces_of(w)[0] for w in words[start : end + 1]]
            bins.append(fertility_bin(sum(pieces) / len(pieces)))
            keys.append((i, etype, start, end))
    hits = np.zeros((len(runs), len(keys)), dtype=bool)
    for r, run in enumerate(runs):
        predicted = [set(get_entities(mapping.map_tags(p))) for p in run.extra.pred]
        hits[r] = [(etype, start, end) in predicted[i] for i, etype, start, end in keys]
    rows = []
    labels = np.array(bins)
    for label, _ in FERTILITY_BINS:
        mask = labels == label
        recalls = [float(h[mask].mean()) for h in hits] if mask.any() else []
        mean, sd = _mean_sd(recalls)
        rows.append({"bin": label, "n": int(mask.sum()), "recall": mean, "sd": sd})
    return rows


def _mean_sd(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None


def _summarise(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean and sample sd over runs of P/R/F1, and the mean F1 per type."""
    out: dict[str, Any] = {}
    for metric in METRICS:
        out[metric], out[f"{metric}_sd"] = _mean_sd([s[metric] for s in scores])
    types = sorted({t for s in scores for t in s["per_type"]})
    out["per_type"] = {
        t: {
            "f1": _mean_sd([s["per_type"][t]["f1"] if t in s["per_type"] else 0.0 for s in scores])[
                0
            ],
            "support": max(s["per_type"].get(t, {}).get("support", 0) for s in scores),
        }
        for t in types
    }
    return out


def _check_same_test_sets(runs: Sequence[TransferRun], extra: str) -> None:
    first = runs[0]
    for run in runs[1:]:
        if run.source.words != first.source.words:
            raise ValueError(f"runs {first.run_id} and {run.run_id} have different KazNERD tests")
        if run.extra.words != first.extra.words or run.extra.gold != first.extra.gold:
            raise ValueError(f"runs {first.run_id} and {run.run_id} have different {extra} sets")


def analyse(
    runs: Sequence[TransferRun],
    mapping: LabelMapping,
    extra: str,
    *,
    by: Sequence[str] = ("model",),
    pieces_of: PieceCounter | None = None,
    bootstrap: int = 1000,
    seed: int = 0,
    tokenizer_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Scores of every group of runs (one group per value of the ``by`` fields)."""
    if not runs:
        raise ValueError("no runs to analyse")
    if extra in SOURCE_STEPS:
        raise ValueError(f"the extra test set cannot be called {extra!r}")
    _check_same_test_sets(runs, extra)
    groups: dict[tuple[Any, ...], list[TransferRun]] = {}
    for run in runs:
        groups.setdefault(tuple(config_field(run.config, key) for key in by), []).append(run)
    report: dict[str, Any] = {
        "mapping": mapping.to_dict(),
        "extra_test": extra,
        "steps": [*SOURCE_STEPS, extra],
        "by": list(by),
        "bootstrap": {"replicates": bootstrap, "seed": seed, "unit": "sentence"},
        "reference_tokenizer": dict(tokenizer_info) if tokenizer_info else None,
        "groups": [],
    }
    for key, members in sorted(groups.items(), key=lambda item: str(item[0])):
        members = sorted(members, key=lambda r: int(r.config.get("seed", 0)))
        scores = [score_run(run, mapping, extra) for run in members]
        steps = {step: _summarise([s[step] for s in scores]) for step in report["steps"]}
        counts = [
            entity_counts(run.extra.gold, [mapping.map_tags(p) for p in run.extra.pred])
            for run in members
        ]
        steps[extra]["f1_ci95"] = bootstrap_mean_f1(counts, bootstrap, seed)
        group: dict[str, Any] = {
            "group": dict(zip(by, key, strict=True)),
            "runs": [run.run_id for run in members],
            "seeds": [run.config.get("seed") for run in members],
            "steps": steps,
            "per_run_f1": {
                run.run_id: {step: s[step]["f1"] for step in report["steps"]}
                for run, s in zip(members, scores, strict=True)
            },
        }
        if pieces_of is not None:
            group["recall_by_pieces"] = recall_by_pieces(members, mapping, pieces_of)
        group["subsets"] = {}
        for name, mask in sentence_subsets(members[0].extra.words).items():
            if not mask.any():
                continue
            subset = subset_scores(counts, mask, bootstrap, seed)
            if pieces_of is not None:
                subset["recall_by_pieces"] = recall_by_pieces(members, mapping, pieces_of, mask)
            group["subsets"][name] = subset
        report["groups"].append(group)
    return report


# --------------------------------------------------------------------------- output


def _fmt(value: float | None, sd: float | None = None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}" if sd is None else f"{value:.4f} ± {sd:.4f}"


def format_report(report: Mapping[str, Any]) -> str:
    extra = report["extra_test"]
    lines = [
        f"Transfer to {extra} ({report['mapping']['source']} -> {report['mapping']['target']})"
    ]
    for group in report["groups"]:
        label = ", ".join(f"{k}={v}" for k, v in group["group"].items())
        lines.append(f"\n== {label}: {len(group['runs'])} run(s), seeds {group['seeds']}")
        for step in report["steps"]:
            s = group["steps"][step]
            line = f"  {step:<16} P {_fmt(s['precision'])}  R {_fmt(s['recall'])}  "
            line += f"F1 {_fmt(s['f1'], s['f1_sd'])}"
            if "f1_ci95" in s:
                low, high = s["f1_ci95"]
                line += f"  95% CI [{low:.4f}, {high:.4f}]"
            lines.append(line)
        targets = report["mapping"]["target_types"]
        for step in report["steps"][1:]:
            per_type = group["steps"][step]["per_type"]
            cells = [f"{t} {_fmt(per_type[t]['f1'])}" for t in targets if t in per_type]
            lines.append(f"  {step:<16} per type F1: {', '.join(cells)}")
        for row in group.get("recall_by_pieces", []):
            lines.append(
                f"  {extra} recall, pieces/word {row['bin']:<6} n={row['n']:<5} "
                f"{_fmt(row['recall'], row['sd'])}"
            )
        for name, s in group["subsets"].items():
            low, high = s["f1_ci95"]
            lines.append(
                f"  {extra}/{name:<12} {s['sentences']:>5} sent. {s['gold_entities']:>5} gold  "
                f"P {_fmt(s['precision'])}  R {_fmt(s['recall'])}  F1 {_fmt(s['f1'], s['f1_sd'])}"
                f"  95% CI [{low:.4f}, {high:.4f}]"
            )
        unique = group["subsets"].get("other_unique", {})
        for row in unique.get("recall_by_pieces", []):
            lines.append(
                f"  {extra}/other_unique recall, pieces/word {row['bin']:<6} n={row['n']:<5} "
                f"{_fmt(row['recall'], row['sd'])}"
            )
    return "\n".join(lines)


def write_outputs(out_dir: str | Path, report: Mapping[str, Any]) -> Path:
    """``transfer.json`` (full report) and ``transfer.csv`` (one row per group and step)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    (out_dir / "transfer.json").write_text(text, encoding="utf-8", newline="\n")
    by = report["by"]
    columns = [
        *by,
        "step",
        "n_runs",
        "precision",
        "recall",
        "f1",
        "f1_sd",
        "f1_ci_low",
        "f1_ci_high",
    ]
    with (out_dir / "transfer.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for group in report["groups"]:
            for step in report["steps"]:
                s = group["steps"][step]
                low, high = s.get("f1_ci95", (None, None))
                writer.writerow({
                    **group["group"], "step": step, "n_runs": len(group["runs"]),
                    "precision": s["precision"], "recall": s["recall"], "f1": s["f1"],
                    "f1_sd": s["f1_sd"], "f1_ci_low": low, "f1_ci_high": high,
                })  # fmt: skip
            for name, s in group["subsets"].items():
                low, high = s["f1_ci95"]
                writer.writerow({
                    **group["group"], "step": f"{report['extra_test']}/{name}",
                    "n_runs": len(group["runs"]), "precision": s["precision"],
                    "recall": s["recall"], "f1": s["f1"], "f1_sd": s["f1_sd"],
                    "f1_ci_low": low, "f1_ci_high": high,
                })  # fmt: skip
    return out_dir / "transfer.json"
