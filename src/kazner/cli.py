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

    kwargs = {"add_prefix_space": True} if args.add_prefix_space else {}
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, **kwargs)
    report = analyse_corpus(
        args.data, tokenizer, tokenizer_name=args.tokenizer, lowercase=args.lowercase
    )
    report["add_prefix_space"] = args.add_prefix_space
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


def _cmd_aggregate(args: argparse.Namespace) -> int:
    from kazner.aggregate import aggregate, format_table, load_runs, write_csv

    by = [key.strip() for key in args.by.split(",") if key.strip()]
    rows = aggregate(load_runs(args.runs), by, block=args.metrics)
    if args.out is not None:
        write_csv(args.out, rows)
    print(format_table(rows, by))
    return 0


def _cmd_fragmentation(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    from kazner.fragmentation import analyse, format_report, load_run, write_outputs

    kwargs: dict[str, object] = {}
    if args.revision is not None:
        kwargs["revision"] = args.revision
    if args.add_prefix_space:
        kwargs["add_prefix_space"] = True
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, **kwargs)
    info = {
        "name": args.tokenizer,
        "revision": args.revision,
        "add_prefix_space": args.add_prefix_space,
    }
    report, entities = analyse(
        [load_run(d) for d in args.runs],
        args.data,
        tokenizer,
        by=[key.strip() for key in args.by.split(",") if key.strip()],
        bootstrap=args.bootstrap,
        seed=args.seed,
        tokenizer_info=info,
    )
    if args.out is not None:
        write_outputs(args.out, report, entities)
    print(format_report(report))
    return 0


def _cmd_extend_vocab(args: argparse.Namespace) -> int:
    from kazner.vocab import extend_vocabulary, read_pieces

    extend_vocabulary(
        args.base,
        args.corpus,
        args.data,
        args.out,
        size=args.size,
        sizes=[int(s) for s in args.sizes.split(",") if s.strip()],
        revision=args.revision,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        pieces=None if args.pieces is None else read_pieces(args.pieces),
    )
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
    fertility.add_argument(
        "--add-prefix-space",
        action="store_true",
        help="create the tokenizer with add_prefix_space=True (byte-level BPE, e.g. RoBERTa)",
    )
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

    agg = commands.add_parser(
        "aggregate",
        help="aggregate runs across seeds (mean, sd, n)",
        description="Group completed runs by config fields and report mean ± sd per metric.",
    )
    agg.add_argument("--runs", type=Path, required=True, help="directory with run folders")
    agg.add_argument(
        "--by",
        default="method,fraction",
        help="comma-separated keys: aliases (method, fraction, model, lr, epochs, r, max_len) "
        "or dotted config paths",
    )
    agg.add_argument(
        "--metrics",
        choices=["test_contract", "test_trainer"],
        default="test_contract",
        help="which test-metric block to aggregate",
    )
    agg.add_argument("--out", type=Path, help="write the summary CSV here")
    agg.set_defaults(func=_cmd_aggregate)

    frag = commands.add_parser(
        "fragmentation",
        help="errors vs subword fragmentation on contract files (experiment B2)",
        description="Error rates and exact-span recall by pieces per word under a fixed "
        "reference tokenizer, with training frequency controlled; logistic regression with a "
        "sentence bootstrap.",
    )
    frag.add_argument(
        "--runs", type=Path, nargs="+", required=True, help="run directories (same test set)"
    )
    frag.add_argument(
        "--data", type=Path, required=True, help="KazNERD directory the runs were trained on"
    )
    frag.add_argument(
        "--tokenizer",
        default="bert-base-multilingual-cased",
        help="reference tokenizer that defines pieces per word (name or path)",
    )
    frag.add_argument("--revision", help="Hub revision of the reference tokenizer")
    frag.add_argument(
        "--add-prefix-space",
        action="store_true",
        help="create the reference tokenizer with add_prefix_space=True (byte-level BPE)",
    )
    frag.add_argument(
        "--by", default="model,method", help="comma-separated config fields to group runs by"
    )
    frag.add_argument("--bootstrap", type=int, default=200, help="bootstrap replicates for the CI")
    frag.add_argument("--seed", type=int, default=0, help="bootstrap random seed")
    frag.add_argument("--out", type=Path, help="write fragmentation.json and entities.csv here")
    frag.set_defaults(func=_cmd_fragmentation)

    ext = commands.add_parser(
        "extend-vocab",
        help="extend a WordPiece model's vocabulary with pieces learned on a corpus",
        description="Learn WordPiece candidates on an unlabeled corpus, report pieces per word on "
        "KazNERD validation for several sizes, and with --size write the extended tokenizer and "
        "masked-LM model (old ids kept, new rows initialised by FVT).",
    )
    ext.add_argument("--corpus", type=Path, required=True, help="unlabeled text, one line each")
    ext.add_argument("--data", type=Path, required=True, help="KazNERD directory (validation used)")
    ext.add_argument("--out", type=Path, required=True, help="output directory")
    ext.add_argument("--base", default="bert-base-multilingual-cased", help="base model")
    ext.add_argument("--revision", help="Hub revision of the base model")
    ext.add_argument("--size", type=int, help="pieces to add; without it only report.json")
    ext.add_argument(
        "--sizes", default="5000,10000,20000,30000", help="comma-separated sizes to report"
    )
    ext.add_argument("--vocab-size", type=int, default=60000, help="WordPiece trainer vocab size")
    ext.add_argument("--min-frequency", type=int, default=5, help="WordPiece trainer min frequency")
    ext.add_argument(
        "--pieces",
        type=Path,
        help="use these pieces (e.g. a saved new_pieces.txt) instead of learning candidates",
    )
    ext.set_defaults(func=_cmd_extend_vocab)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return args.func(args)
