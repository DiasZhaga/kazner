"""Command-line interface: ``kazner <command>``."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from kazner import __version__


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from kazner.data import build_label2id, read_iob2
    from kazner.evaluate import evaluate_file

    labels = None
    if args.labels_from is not None:
        labels = build_label2id(read_iob2(args.labels_from)[1])[0]
    metrics = evaluate_file(args.predictions, out_path=args.out, labels=labels)
    summary = {k: v for k, v in metrics.items() if k != "per_type"}
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_fertility(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    from kazner.fertility import analyse_corpus, write_report

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    report = analyse_corpus(
        args.data, tokenizer, tokenizer_name=args.tokenizer, lowercase=args.lowercase
    )
    if args.out is not None:
        write_report(args.out, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    import os

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from kazner.config import load_config
    from kazner.train import run_experiment

    results = run_experiment(
        load_config(args.config), force=args.force, max_train_samples=args.max_train_samples
    )
    done = sum(r["status"] == "done" for r in results)
    print(f"{done} run(s) trained, {len(results) - done} skipped")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kazner",
        description="Kazakh NER toolkit: data, evaluation and tokenizer analysis.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="<command>")

    evaluate = commands.add_parser(
        "evaluate",
        help="score a word-level prediction file (contract CSV)",
        description="Entity-level P/R/F1 (seqeval, default mode) and token accuracy.",
    )
    evaluate.add_argument("--predictions", type=Path, required=True, help="contract CSV")
    evaluate.add_argument("--out", type=Path, help="write metrics.json here")
    evaluate.add_argument(
        "--labels-from",
        type=Path,
        help="IOB2 train file; its labels become the known label set (default: IOB2 syntax)",
    )
    evaluate.set_defaults(func=_cmd_evaluate)

    fertility = commands.add_parser(
        "fertility",
        help="measure tokenizer fertility on an IOB2 corpus",
        description="Subwords per word, split-word rate and UNK rate per split and overall.",
    )
    fertility.add_argument(
        "--data", type=Path, required=True, help="directory with IOB2_{train,valid,test}.txt"
    )
    fertility.add_argument(
        "--tokenizer", default="bert-base-multilingual-cased", help="Hugging Face name or path"
    )
    fertility.add_argument("--lowercase", action="store_true", help="lowercase words first")
    fertility.add_argument("--out", type=Path, help="write the JSON report here")
    fertility.set_defaults(func=_cmd_fertility)

    train = commands.add_parser(
        "train",
        help="train every seed of an experiment config",
        description="Train, evaluate and write metrics and contract predictions per run ID.",
    )
    train.add_argument("--config", type=Path, required=True, help="experiment YAML file")
    train.add_argument("--force", action="store_true", help="re-run completed run IDs")
    train.add_argument(
        "--max-train-samples",
        type=int,
        help="use only the first N sampled training sentences (changes the run ID)",
    )
    train.set_defaults(func=_cmd_train)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return args.func(args)
