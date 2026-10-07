"""One evaluator for every model: entity-level P/R/F1 (seqeval) and token accuracy.

seqeval is called directly, in its default (conlleval-compatible) mode, computing exactly
what the pilot obtained through ``evaluate.load("seqeval")``: overall scores are the
``micro avg`` row of ``classification_report``, accuracy is ``accuracy_score``.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

from seqeval.metrics import accuracy_score, classification_report
from seqeval.metrics.sequence_labeling import get_entities

from kazner.contract import Predictions, read_predictions

SEQEVAL_MODE = "default"


def compute_metrics(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]]) -> dict[str, Any]:
    """Entity-level precision / recall / F1, token accuracy and per-type scores."""
    if len(gold) != len(pred) or any(len(g) != len(p) for g, p in zip(gold, pred, strict=True)):
        raise ValueError("gold and pred must have the same sentences and lengths")
    gold, pred = [list(s) for s in gold], [list(s) for s in pred]
    report = classification_report(gold, pred, output_dict=True, zero_division=0)
    overall = report.pop("micro avg")
    report.pop("macro avg")
    report.pop("weighted avg")
    return {
        "precision": float(overall["precision"]),
        "recall": float(overall["recall"]),
        "f1": float(overall["f1-score"]),
        "accuracy": float(accuracy_score(gold, pred)),
        "n_gold_entities": len(get_entities(gold)),
        "n_pred_entities": len(get_entities(pred)),
        "per_type": {
            name: {
                "precision": float(s["precision"]),
                "recall": float(s["recall"]),
                "f1": float(s["f1-score"]),
                "support": int(s["support"]),
            }
            for name, s in sorted(report.items())
        },
    }


def evaluate_predictions(predictions: Predictions) -> dict[str, Any]:
    """Score a contract object; adds corpus counts to :func:`compute_metrics`."""
    metrics = compute_metrics(predictions.gold, predictions.pred)
    metrics.update(
        n_sentences=len(predictions.words),
        n_words=predictions.n_words,
        n_truncated_words=predictions.n_truncated_words,
        seqeval_mode=SEQEVAL_MODE,
    )
    return metrics


def evaluate_file(
    predictions_path: str | Path,
    out_path: str | Path | None = None,
    labels: Collection[str] | None = None,
) -> dict[str, Any]:
    """Validate and score a contract CSV; optionally write the metrics as JSON."""
    metrics = evaluate_predictions(read_predictions(predictions_path, labels=labels))
    if out_path is not None:
        write_metrics(out_path, metrics)
    return metrics


def write_metrics(path: str | Path, metrics: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path
