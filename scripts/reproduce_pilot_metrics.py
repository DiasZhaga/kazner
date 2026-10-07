"""Re-score the pilot's saved prediction files with kazner and compare with its metrics.

Usage:
    python scripts/reproduce_pilot_metrics.py --pilot ../ner-project [--out-dir DIR]

Reads ``<pilot>/results`` and ``<pilot>/data``; nothing is written into the pilot. For each
run it scores the prediction file as the pilot wrote it (only words the model saw) and as
a complete contract (truncated words appended with ``pred = O``). Results: docs/reproduction.md.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from kazner.contract import convert_pilot_predictions, write_predictions
from kazner.data import build_label2id, read_iob2
from kazner.evaluate import evaluate_file

CROSS_LINGUAL_RUNS = [
    "cross_lingual_matched_baseline_kaznerd1_ep3_len64_bs2/kaznerd_1pct_direct_full_ft",
    "cross_lingual_twnertc10_kaznerd1_ep3_len64_bs2_transfer/kaznerd_1pct_full_ft",
    "cross_lingual_twnertc10_kaznerd1_ep3_len64_bs2_transfer/kaznerd_1pct_lora",
    "cross_lingual_twnertc10_kaznerd1_ep3_len64_bs2_transfer/kaznerd_1pct_frozen_encoder",
]
METRICS = ("precision", "recall", "f1", "accuracy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pilot", type=Path, required=True, help="pilot repository root")
    parser.add_argument("--out-dir", type=Path, help="keep converted files and metrics here")
    args = parser.parse_args()

    results, data = args.pilot / "results", args.pilot / "data"
    gold = data / "IOB2_test.txt"
    labels = build_label2id(read_iob2(data / "IOB2_train.txt")[1])[0]
    out_dir = args.out_dir or Path(tempfile.mkdtemp(prefix="kazner-pilot-"))

    runs = [(r, results / r / "test_predictions.csv", results / r / "test_metrics.json")
            for r in CROSS_LINGUAL_RUNS]  # fmt: skip
    runs.append(("test_predictions_mbert", results / "test_predictions_mbert.csv", None))

    print(f"{'run':<30} {'variant':<9} " + " ".join(f"{m:>12}" for m in METRICS) + "  trunc")
    for name, pred_csv, stored_json in runs:
        short = name.rsplit("/", 1)[-1]
        stored = json.loads(stored_json.read_text(encoding="utf-8")) if stored_json else None
        if stored:
            row = " ".join(f"{stored[m]:>12.10f}" for m in METRICS)
            print(f"{short:<30} {'stored':<9} {row}")
        for variant, gold_path in (("as-is", None), ("complete", gold)):
            contract = out_dir / f"{short}.{variant}.csv"
            write_predictions(contract, convert_pilot_predictions(pred_csv, gold_path))
            m = evaluate_file(contract, out_dir / f"{short}.{variant}.metrics.json", labels)
            row = " ".join(f"{m[k]:>12.10f}" for k in METRICS)
            print(f"{short:<30} {variant:<9} {row}  {m['n_truncated_words']}")
    print(f"\nconverted files and metrics: {out_dir}")


if __name__ == "__main__":
    main()
