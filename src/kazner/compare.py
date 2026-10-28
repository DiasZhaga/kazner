"""Paired comparison of two arms on fixed entity subsets (#37, ``docs/preregistration-vocab.md``).

Input: the ``entities.csv`` written by ``kazner fragmentation`` for all runs of both arms. It
holds one row per gold test entity with the subset columns (``seen``, ``mean_pieces``,
``has_unk``) and one ``hit_<run_id>`` column per run (exact span and type predicted).

For every pair of arms (e.g. full fine-tuning: original vs extended vocabulary), and for every
preregistered subset, the outcome is

    Δ = mean recall over arm B's runs − mean recall over arm A's runs.

Inference is a paired bootstrap over test sentences. Each replicate resamples the sentences
that contain entities once and recomputes both arms' seed-mean recall on the resampled
entities, for all subsets at once, so that the difference-in-differences
Δ(primary) − Δ(specificity) is bootstrapped jointly. The report gives percentile 95% CIs and
two-sided bootstrap p-values, ``(1 + #{Δ* on the far side of 0}) / (1 + replicates)``, doubled
and capped at 1. The primary tests of all pairs are Holm-corrected.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: Preregistered subsets; "pieces" is the mean number of original-mBERT pieces per word.
SUBSETS = {
    "primary": "unseen, more than 3 pieces per word, no [UNK]",
    "specificity": "unseen, at most 2 pieces per word, no [UNK]",
    "negative_control": "seen, no [UNK]",
    "unseen_with_unk": "unseen, with an [UNK] word (reported separately)",
}
ALPHA = 0.05
CHUNK = 500  # bootstrap replicates drawn per matrix product


@dataclass(frozen=True)
class Pair:
    """Two arms to compare: ``b`` (treatment) minus ``a`` (control)."""

    name: str
    a: tuple[str, ...]
    b: tuple[str, ...]


def subset_masks(entities: pd.DataFrame) -> dict[str, np.ndarray]:
    seen = entities["seen"].astype(bool).to_numpy()
    unk = entities["has_unk"].astype(bool).to_numpy()
    pieces = entities["mean_pieces"].to_numpy(float)
    return {
        "primary": ~seen & (pieces > 3) & ~unk,
        "specificity": ~seen & (pieces <= 2) & ~unk,
        "negative_control": seen & ~unk,
        "unseen_with_unk": ~seen & unk,
    }


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values (step-down, monotone, capped at 1)."""
    order = sorted(pvalues, key=lambda k: pvalues[k])
    adjusted, running = {}, 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvalues[key]))
        adjusted[key] = running
    return {k: adjusted[k] for k in pvalues}


def _p_two_sided(replicates: np.ndarray) -> float:
    replicates = replicates[~np.isnan(replicates)]
    n = len(replicates)
    below = (1 + np.sum(replicates <= 0)) / (1 + n)
    above = (1 + np.sum(replicates >= 0)) / (1 + n)
    return float(min(1.0, 2 * min(below, above)))


def _ci(replicates: np.ndarray) -> list[float]:
    low, high = np.nanpercentile(replicates, [2.5, 97.5])
    return [float(low), float(high)]


def _check(entities: pd.DataFrame, pairs: Sequence[Pair]) -> None:
    if not pairs:
        raise ValueError("no pairs to compare")
    names = [p.name for p in pairs]
    if len(set(names)) != len(names):
        raise ValueError(f"pair names must be unique: {names}")
    for pair in pairs:
        if not pair.a or not pair.b:
            raise ValueError(f"pair {pair.name!r}: both arms need at least one run")
        if set(pair.a) & set(pair.b):
            raise ValueError(f"pair {pair.name!r}: a run cannot be in both arms")
        missing = [r for r in (*pair.a, *pair.b) if f"hit_{r}" not in entities.columns]
        if missing:
            raise ValueError(f"pair {pair.name!r}: no hit column for runs {missing}")


def compare(
    entities: pd.DataFrame, pairs: Sequence[Pair], replicates: int = 10000, seed: int = 0
) -> dict[str, Any]:
    """Point estimates, bootstrap CIs and p-values for every pair and subset."""
    _check(entities, pairs)
    masks = subset_masks(entities)
    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(f"subset {name!r} is empty")
    runs = sorted({r for p in pairs for r in (*p.a, *p.b)})
    sentence_ids, sentence_index = np.unique(
        entities["sentence_id"].to_numpy(), return_inverse=True
    )
    n_sentences = len(sentence_ids)

    # Per-sentence counts: one column per subset (entities) and per (subset, run) (hits).
    columns: list[np.ndarray] = []
    entity_col, hit_col = {}, {}
    for name, mask in masks.items():
        entity_col[name] = len(columns)
        columns.append(np.bincount(sentence_index, weights=mask, minlength=n_sentences))
        for run in runs:
            hits = entities[f"hit_{run}"].astype(bool).to_numpy() & mask
            hit_col[name, run] = len(columns)
            columns.append(np.bincount(sentence_index, weights=hits, minlength=n_sentences))
    matrix = np.column_stack(columns)

    def deltas(totals: np.ndarray) -> dict[tuple[str, str], np.ndarray]:
        """Δ per (pair, subset) from summed counts; ``totals`` is replicates × columns."""
        with np.errstate(invalid="ignore", divide="ignore"):
            out = {}
            for pair in pairs:
                for name in masks:
                    n = totals[:, entity_col[name]]
                    mean_b = np.mean([totals[:, hit_col[name, r]] / n for r in pair.b], axis=0)
                    mean_a = np.mean([totals[:, hit_col[name, r]] / n for r in pair.a], axis=0)
                    out[pair.name, name] = mean_b - mean_a
            return out

    point_totals = matrix.sum(axis=0, keepdims=True)
    point = {k: float(v[0]) for k, v in deltas(point_totals).items()}

    rng = np.random.default_rng(seed)
    chunks: list[dict[tuple[str, str], np.ndarray]] = []
    uniform = np.full(n_sentences, 1.0 / n_sentences)
    for start in range(0, replicates, CHUNK):
        size = min(CHUNK, replicates - start)
        weights = rng.multinomial(n_sentences, uniform, size=size)
        chunks.append(deltas(weights @ matrix))
    boot = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}

    report_pairs: dict[str, Any] = {}
    for pair in pairs:
        subsets = {}
        for name, mask in masks.items():
            n = int(mask.sum())
            recall = {r: float(entities.loc[mask, f"hit_{r}"].astype(bool).mean()) for r in runs}
            subsets[name] = {
                "n": n,
                "recall_a": float(np.mean([recall[r] for r in pair.a])),
                "recall_b": float(np.mean([recall[r] for r in pair.b])),
                "recall_a_runs": {r: recall[r] for r in pair.a},
                "recall_b_runs": {r: recall[r] for r in pair.b},
                "delta": point[pair.name, name],
                "ci95": _ci(boot[pair.name, name]),
                "p_two_sided": _p_two_sided(boot[pair.name, name]),
            }
        did = boot[pair.name, "primary"] - boot[pair.name, "specificity"]
        report_pairs[pair.name] = {
            "a": list(pair.a),
            "b": list(pair.b),
            "subsets": subsets,
            "did_primary_minus_specificity": {
                "delta": point[pair.name, "primary"] - point[pair.name, "specificity"],
                "ci95": _ci(did),
                "p_two_sided": _p_two_sided(did),
            },
        }

    adjusted = holm(
        {p.name: report_pairs[p.name]["subsets"]["primary"]["p_two_sided"] for p in pairs}
    )
    for pair in pairs:
        r = report_pairs[pair.name]
        primary, did = r["subsets"]["primary"], r["did_primary_minus_specificity"]
        primary["p_holm"] = adjusted[pair.name]
        r["h1_supported"] = bool(primary["delta"] > 0 and adjusted[pair.name] < ALPHA)
        r["h2_supported"] = bool(did["delta"] > 0 and did["ci95"][0] > 0)
    return {
        "subsets": SUBSETS,
        "bootstrap": {
            "unit": "test sentence",
            "replicates": replicates,
            "seed": seed,
            "sentences": int(n_sentences),
        },
        "alpha": ALPHA,
        "correction": "Holm over the primary tests of all pairs",
        "pairs": report_pairs,
    }


def format_comparison(report: Mapping[str, Any]) -> str:
    lines = []
    for name, pair in report["pairs"].items():
        lines.append(f"=== {name}: B ({len(pair['b'])} runs) - A ({len(pair['a'])} runs) ===")
        for subset, s in pair["subsets"].items():
            low, high = s["ci95"]
            holm_p = f", Holm p {s['p_holm']:.4f}" if "p_holm" in s else ""
            lines.append(
                f"{subset:17s} n={s['n']:5d}  A {s['recall_a']:.4f}  B {s['recall_b']:.4f}  "
                f"delta {s['delta']:+.4f} [{low:+.4f}, {high:+.4f}]  p {s['p_two_sided']:.4f}"
                + holm_p
            )
        did = pair["did_primary_minus_specificity"]
        lines.append(
            f"DiD primary - specificity  {did['delta']:+.4f} "
            f"[{did['ci95'][0]:+.4f}, {did['ci95'][1]:+.4f}]  p {did['p_two_sided']:.4f}"
        )
        lines.append(f"H1 supported: {pair['h1_supported']}; H2 supported: {pair['h2_supported']}")
    return "\n".join(lines)


def write_comparison(path: str | Path, report: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path
