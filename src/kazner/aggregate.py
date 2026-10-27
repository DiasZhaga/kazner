"""Aggregate run results across seeds: mean, sample standard deviation and n per group."""

from __future__ import annotations

import csv
import json
import statistics
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: Short grouping keys -> dotted paths into ``config.json``.
ALIASES = {
    "method": "method.name",
    "fraction": "data.train_fraction",
    "model": "model.name",
    "lr": "training.learning_rate",
    "epochs": "training.epochs",
    "r": "method.lora.r",
    "max_len": "model.max_len",
}
METRICS = ("precision", "recall", "f1", "accuracy")
BLOCKS = ("test_contract", "test_trainer")


def config_field(config: dict[str, Any], key: str) -> Any:
    """Value of ``key`` (an alias from :data:`ALIASES` or a dotted path) in ``config.json``."""
    dotted = ALIASES.get(key, key)
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


def load_runs(runs_dir: str | Path) -> list[dict[str, Any]]:
    """Every run directory with ``config.json``; incomplete runs are skipped with a warning."""
    runs = []
    for run_dir in sorted(p for p in Path(runs_dir).iterdir() if p.is_dir()):
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            warnings.warn(
                f"{run_dir.name}: no metrics.json (incomplete run), skipped", stacklevel=2
            )
            continue
        env_path = run_dir / "environment.json"
        env = json.loads(env_path.read_text(encoding="utf-8")) if env_path.exists() else {}
        runs.append(
            {
                "run_id": run_dir.name,
                "config": json.loads(config_path.read_text(encoding="utf-8")),
                "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
                "code_ref": env.get("code", {}).get("code_ref"),
            }
        )
    return runs


def aggregate(
    runs: Sequence[dict[str, Any]], by: Sequence[str], block: str = "test_contract"
) -> list[dict[str, Any]]:
    """One row per group of ``by`` values with mean, sd (n > 1) and n for each metric."""
    if block not in BLOCKS:
        raise ValueError(f"unknown metric block {block!r}; choose from {list(BLOCKS)}")
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for run in runs:
        try:
            key = tuple(config_field(run["config"], field) for field in by)
        except KeyError as exc:
            raise KeyError(f"run {run['run_id']}: config has no field {exc.args[0]!r}") from None
        groups.setdefault(key, []).append(run)

    rows = []
    for key, members in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        code_refs = {m["code_ref"] for m in members}
        if len(code_refs) > 1:
            warnings.warn(
                f"group {dict(zip(by, key, strict=True))} mixes runs from different code "
                f"versions: {sorted(map(str, code_refs))}",
                stacklevel=2,
            )
        row: dict[str, Any] = dict(zip(by, key, strict=True))
        row["n"] = len(members)
        for metric in METRICS:
            values = [m["metrics"][block][metric] for m in members]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else None
        row["seeds"] = " ".join(str(m["config"]["seed"]) for m in members)
        row["run_ids"] = " ".join(m["run_id"] for m in members)
        rows.append(row)
    return rows


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["n"], lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: ("" if v is None else v) for k, v in r.items()} for r in rows)
    return path


def format_table(rows: Sequence[dict[str, Any]], by: Sequence[str]) -> str:
    """Human-readable ``mean ± sd`` summary for the terminal."""
    lines = []
    for r in rows:
        cells = [f"{k}={r[k]}" for k in by] + [f"n={r['n']}"]
        for metric in ("f1", "precision", "recall"):
            sd = r[f"{metric}_sd"]
            spread = f" ± {sd:.4f}" if sd is not None else ""
            cells.append(f"{metric}={r[f'{metric}_mean']:.4f}{spread}")
        lines.append("  ".join(cells))
    return "\n".join(lines)
