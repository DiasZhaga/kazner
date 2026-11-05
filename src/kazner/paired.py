"""Paired comparison of two arms' test F1 over seeds and test sentences (#59): ``kazner paired``.

An arm is the set of completed runs of one experiment: the runs under a directory whose
``config.json`` has that ``name``, one per seed. Both arms must have the same seeds and the
same test set. The outcome is

    Δ = mean test F1 over arm B's seeds − mean test F1 over arm A's seeds,

with entity-level micro F1 from the runs' ``predictions.csv`` (exact span and type, as seqeval's
default mode counts them). Runs are paired by seed: one seed gives the same training order and,
for arms with the same start model, the same initial weights.

Two sources of variation enter the uncertainty: the training seed and the test sample. Each
bootstrap replicate resamples the seeds (the same draws in both arms) and the test sentences
(the same draws for every run) and recomputes Δ. The report gives a percentile 95% CI and a
two-sided bootstrap p-value, ``(1 + #{Δ* on the far side of 0}) / (1 + replicates)``, doubled
and capped at 1, as ``kazner compare`` does. A sign-flip permutation test over the per-seed
differences, exact over all sign patterns, treats the test set as fixed.

``types`` restricts the scores to some entity types and ``exclude_types`` to all the others: the
remaining types become ``O`` on both sides before entities are counted.
"""

from __future__ import annotations

import itertools
import json
import statistics
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kazner.compare import _ci, _p_two_sided
from kazner.contract import Predictions, read_predictions
from kazner.transfer import OUTSIDE, entity_counts, entity_types

CHUNK = 500  # bootstrap replicates drawn at once
MAX_EXACT_SEEDS = 16  # 2**16 sign patterns


@dataclass(frozen=True)
class ArmRun:
    seed: int
    run_id: str
    predictions: Predictions


def select_runs(runs_dir: str | Path, name: str) -> list[ArmRun]:
    """The completed runs of experiment ``name`` under ``runs_dir``, sorted by seed."""
    runs_dir = Path(runs_dir)
    found: dict[int, list[Path]] = {}
    for config_path in sorted(runs_dir.glob("*/config.json")):
        run_dir = config_path.parent
        if not (run_dir / "metrics.json").is_file():
            continue  # incomplete
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("name") == name:
            found.setdefault(int(config["seed"]), []).append(run_dir)
    if not found:
        raise ValueError(f"{runs_dir}: no completed runs named {name!r}")
    repeated = {seed: [d.name for d in dirs] for seed, dirs in found.items() if len(dirs) > 1}
    if repeated:
        raise ValueError(f"{runs_dir}: several runs of {name!r} for one seed: {repeated}")
    return [
        ArmRun(seed, dirs[0].name, read_predictions(dirs[0] / "predictions.csv"))
        for seed, dirs in sorted(found.items())
    ]


def restrict_types(
    sentences: Sequence[Sequence[str]],
    types: Collection[str] | None = None,
    exclude_types: Collection[str] | None = None,
) -> list[list[str]]:
    """Set the labels of entity types outside ``types`` (or inside ``exclude_types``) to ``O``."""

    def dropped(tag: str) -> bool:
        etype = tag.partition("-")[2]
        if types is not None and etype not in types:
            return True
        return exclude_types is not None and etype in exclude_types

    return [[OUTSIDE if tag != OUTSIDE and dropped(tag) else tag for tag in s] for s in sentences]


def f1_from_counts(sums: np.ndarray) -> np.ndarray:
    """Micro F1 from summed (correct, predicted, gold) counts on the last axis."""
    denominator = sums[..., 1] + sums[..., 2]
    return np.divide(
        2 * sums[..., 0], denominator, out=np.zeros(denominator.shape), where=denominator > 0
    )


def sign_flip_p(differences: Sequence[float]) -> float:
    """Exact two-sided sign-flip permutation p-value of the mean difference."""
    d = np.asarray(differences, dtype=np.float64)
    if len(d) > MAX_EXACT_SEEDS:
        raise ValueError(f"the exact sign-flip test supports at most {MAX_EXACT_SEEDS} seeds")
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(d))))
    means = np.abs(signs @ d) / len(d)
    observed = abs(d.mean())
    return float(np.mean(means >= observed - 1e-12))


def _check_same_test_set(runs: Sequence[ArmRun]) -> None:
    first = runs[0].predictions
    for run in runs[1:]:
        p = run.predictions
        if (p.sentence_ids, p.words, p.gold) != (first.sentence_ids, first.words, first.gold):
            raise ValueError(
                f"run {run.run_id} was scored on another test set than {runs[0].run_id}"
            )


def paired(
    arm_a: Sequence[ArmRun],
    arm_b: Sequence[ArmRun],
    types: Collection[str] | None = None,
    exclude_types: Collection[str] | None = None,
    replicates: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """Δ = mean F1 of arm B − arm A, with the two-way bootstrap and the sign-flip test."""
    if types is not None and exclude_types is not None:
        raise ValueError("give types or exclude_types, not both")
    seeds = [r.seed for r in arm_a]
    if seeds != [r.seed for r in arm_b]:
        raise ValueError(f"the arms have different seeds: {seeds} vs {[r.seed for r in arm_b]}")
    runs = [*arm_a, *arm_b]
    _check_same_test_set(runs)
    gold = runs[0].predictions.gold
    named = set(types or ()) | set(exclude_types or ())
    unknown = sorted(named - entity_types(gold))
    if unknown:
        raise ValueError(f"types {unknown} do not occur in the gold labels")
    gold = restrict_types(gold, types, exclude_types)

    counts = np.stack(
        [
            entity_counts(gold, restrict_types(r.predictions.pred, types, exclude_types))
            for r in runs
        ]
    ).astype(np.float64)  # runs x sentences x 3
    k, n = len(seeds), counts.shape[1]
    f1 = f1_from_counts(counts.sum(axis=1))
    f1_a, f1_b = f1[:k], f1[k:]
    differences = f1_b - f1_a

    rng = np.random.default_rng(seed)
    uniform = np.full(n, 1.0 / n)
    boot = np.empty(replicates)
    for start in range(0, replicates, CHUNK):
        size = min(CHUNK, replicates - start)
        weights = rng.multinomial(n, uniform, size=size).astype(np.float64)
        sample = f1_from_counts(np.einsum("bn,rnk->brk", weights, counts))  # size x runs
        drawn = rng.integers(0, k, size=(size, k))  # seed positions, shared by both arms
        mean_a = np.take_along_axis(sample[:, :k], drawn, axis=1).mean(axis=1)
        mean_b = np.take_along_axis(sample[:, k:], drawn, axis=1).mean(axis=1)
        boot[start : start + size] = mean_b - mean_a

    def arm(runs_: Sequence[ArmRun], values: np.ndarray) -> dict[str, Any]:
        return {
            "runs": {
                str(r.seed): {"run_id": r.run_id, "f1": float(v)}
                for r, v in zip(runs_, values, strict=True)
            },
            "f1_mean": float(values.mean()),
            "f1_sd": statistics.stdev(values.tolist()) if k > 1 else None,
        }

    return {
        "types": sorted(types) if types is not None else None,
        "exclude_types": sorted(exclude_types) if exclude_types is not None else None,
        "seeds": seeds,
        "a": arm(arm_a, f1_a),
        "b": arm(arm_b, f1_b),
        "delta": float(differences.mean()),
        "delta_sd": statistics.stdev(differences.tolist()) if k > 1 else None,
        "per_seed_delta": {str(s): float(d) for s, d in zip(seeds, differences, strict=True)},
        "b_better_seeds": int(np.sum(differences > 0)),
        "bootstrap": {
            "unit": "seed and test sentence",
            "replicates": replicates,
            "seed": seed,
            "sentences": int(n),
            "ci95": _ci(boot),
            "p_two_sided": _p_two_sided(boot),
        },
        "sign_flip": {"patterns": 2**k, "p_two_sided": sign_flip_p(differences)},
    }


def format_paired(report: dict[str, Any], name_a: str, name_b: str) -> str:
    def f(value: float | None) -> str:
        return "-" if value is None else f"{value:.4f}"

    b = report["bootstrap"]
    restriction = ""
    if report["types"] is not None:
        restriction = f" (types: {', '.join(report['types'])})"
    elif report["exclude_types"] is not None:
        restriction = f" (without: {', '.join(report['exclude_types'])})"
    lines = [
        f"{name_b} - {name_a}{restriction}, seeds {report['seeds'][0]}-{report['seeds'][-1]} "
        f"(n={len(report['seeds'])})",
        f"  A {name_a}: F1 {f(report['a']['f1_mean'])} ± {f(report['a']['f1_sd'])}",
        f"  B {name_b}: F1 {f(report['b']['f1_mean'])} ± {f(report['b']['f1_sd'])}",
        f"  delta {report['delta']:+.4f} (sd {f(report['delta_sd'])}; B better in "
        f"{report['b_better_seeds']} of {len(report['seeds'])} seeds)",
        f"  bootstrap over seeds and {b['sentences']} sentences: 95% CI "
        f"[{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}], p = {b['p_two_sided']:.4f}",
        f"  sign-flip test over seeds: p = {report['sign_flip']['p_two_sided']:.4f}",
    ]
    return "\n".join(lines)


def write_paired(path: str | Path, report: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path
