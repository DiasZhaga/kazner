"""Training entry point: ``kazner train --config X.yaml`` runs every seed of an experiment.

The Trainer wiring follows the pilot (``ner-project/src/run_peft_comparison.py``:
``build_training_arguments`` and ``run_one_experiment``). Each run writes, into
``<output_root>/<run_id>/``: ``config.json``, ``environment.json``, ``trainer_state.json``,
``predictions.csv`` (the word-level contract) and, last, ``metrics.json``. A run whose
``metrics.json`` exists is complete and is skipped unless ``force`` is set.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import random
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from kazner.align import (
    IGNORE_INDEX,
    first_subword_positions,
    split_mismatches,
    tokenize_and_align,
)
from kazner.config import (
    ConfigError,
    ExperimentConfig,
    RunConfig,
    code_state,
    data_fingerprint,
    environment,
    run_id,
    write_run_files,
)
from kazner.contract import TRUNCATED_PRED, Predictions, write_predictions
from kazner.data import build_label2id, read_splits, sample_fraction
from kazner.evaluate import compute_metrics, evaluate_predictions
from kazner.methods import build_model

METRICS_FILE = "metrics.json"


# --------------------------------------------------------------------------- pure helpers


def set_all_seeds(seed: int) -> None:
    """Pilot's ``set_all_seeds``: Python, NumPy and torch (incl. CUDA)."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_compute_metrics(id2label: Mapping[int, str]):
    """Trainer ``compute_metrics``: seqeval on first-subword positions, as in the pilot."""

    def compute(eval_pred) -> dict[str, float]:
        logits, label_ids = eval_pred
        pred_ids = np.argmax(logits, axis=2)
        gold, pred = [], []
        for pred_row, label_row in zip(pred_ids, label_ids, strict=True):
            keep = label_row != IGNORE_INDEX
            gold.append([id2label[int(i)] for i in label_row[keep]])
            pred.append([id2label[int(i)] for i in pred_row[keep]])
        m = compute_metrics(gold, pred)
        return {k: m[k] for k in ("precision", "recall", "f1", "accuracy")}

    return compute


def word_predictions(
    logits: np.ndarray,
    word_ids: Sequence[Sequence[int | None]],
    words: Sequence[Sequence[str]],
    gold: Sequence[Sequence[str]],
    id2label: Mapping[int, str],
) -> Predictions:
    """Map subword logits to one predicted label per word (first subword); truncated -> O."""
    pred_ids = np.argmax(logits, axis=-1)
    preds, truncated = [], []
    for i, sentence in enumerate(words):
        positions = first_subword_positions(word_ids[i])
        row_pred, row_trunc = [], []
        for w in range(len(sentence)):
            if w in positions:
                row_pred.append(id2label[int(pred_ids[i][positions[w]])])
                row_trunc.append(False)
            else:
                row_pred.append(TRUNCATED_PRED)
                row_trunc.append(True)
        preds.append(row_pred)
        truncated.append(row_trunc)
    return Predictions(
        sentence_ids=list(range(len(words))),
        words=[list(s) for s in words],
        gold=[list(s) for s in gold],
        pred=preds,
        truncated=truncated,
    )


def resolve_fp16(setting: str) -> bool:
    import torch

    cuda = torch.cuda.is_available()
    if setting == "auto":
        return cuda
    if setting == "true" and not cuda:
        raise ValueError("training.fp16 is true but CUDA is not available")
    return setting == "true"


def check_word_tokenization(
    tokenizer: Any, sentences: Sequence[Sequence[str]], add_prefix_space: bool | None
) -> None:
    """Fail unless pre-split words get the same subwords as in running text.

    Otherwise the model is fine-tuned on tokens it did not see in pre-training. transformers
    5 accepts this silently for byte-level BPE tokenizers without ``add_prefix_space``.
    """
    mismatches = split_mismatches(tokenizer, sentences)
    if mismatches:
        i, j = mismatches[0]
        n_words = sum(len(words) for words in sentences)
        raise ConfigError(
            f"model.add_prefix_space: {len(mismatches)} of {n_words} training words are split "
            f"differently as pre-split words than in running text (first: {sentences[i][j]!r}; "
            f"add_prefix_space is {add_prefix_space}). Byte-level BPE tokenizers (RoBERTa "
            "family) need 'add_prefix_space: true'."
        )


def _dataset(tokens, labels):
    from datasets import Dataset

    return Dataset.from_dict({"tokens": tokens, "labels_str": labels})


def _tokenize(dataset, tokenizer, label2id, max_len):
    return dataset.map(
        lambda batch: tokenize_and_align(batch, tokenizer, label2id, max_len),
        batched=True,
        remove_columns=["tokens", "labels_str"],
    )


def _word_ids(tokenizer, sentences, max_len) -> list[list[int | None]]:
    out: list[list[int | None]] = []
    for start in range(0, len(sentences), 1000):
        batch = sentences[start : start + 1000]
        enc = tokenizer(batch, is_split_into_words=True, truncation=True, max_length=max_len)
        out.extend(enc.word_ids(batch_index=i) for i in range(len(batch)))
    return out


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".part")
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


# --------------------------------------------------------------------------- one run


def train_one(
    run: RunConfig,
    rid: str,
    run_dir: Path,
    env: dict[str, Any],
    corpus: dict[str, Any],
    data_ref: str = "",
) -> dict[str, Any]:
    """Train and evaluate a single seed; returns the metrics written to ``metrics.json``."""
    import torch
    from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments

    tokenizer, label2id, id2label = corpus["tokenizer"], corpus["label2id"], corpus["id2label"]
    t = run.training
    fp16 = resolve_fp16(t.fp16)
    cuda = torch.cuda.is_available()

    write_run_files(run_dir, run, rid, env, data_ref)

    set_all_seeds(run.seed)  # immediately before the model is built, as in the pilot
    built = build_model(run.method, run.model.name, label2id, id2label)

    args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=t.logging_steps,
        learning_rate=t.learning_rate,
        per_device_train_batch_size=t.train_batch_size,
        per_device_eval_batch_size=t.eval_batch_size,
        gradient_accumulation_steps=t.grad_accumulation,
        num_train_epochs=t.epochs,
        weight_decay=t.weight_decay,
        fp16=fp16,
        seed=run.seed,
        report_to="none",
        load_best_model_at_end=t.load_best_model_at_end,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        dataloader_pin_memory=cuda,
        use_cpu=not cuda,
    )
    trainer = Trainer(
        model=built.model,
        args=args,
        train_dataset=corpus["train"],
        eval_dataset=corpus["validation"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=build_compute_metrics(id2label),
    )

    start = time.perf_counter()
    trainer.train()
    training_seconds = time.perf_counter() - start

    start = time.perf_counter()
    test_trainer = trainer.evaluate(corpus["test"], metric_key_prefix="test")
    output = trainer.predict(corpus["test"])
    predictions = word_predictions(
        output.predictions,
        corpus["test_word_ids"],
        corpus["test_words"],
        corpus["test_gold"],
        id2label,
    )
    write_predictions(run_dir / "predictions.csv", predictions)
    test_contract = evaluate_predictions(predictions)
    eval_seconds = time.perf_counter() - start

    trainer.state.save_to_json(str(run_dir / "trainer_state.json"))
    history = [h for h in trainer.state.log_history if "eval_f1" in h]
    best = max(history, key=lambda h: h["eval_f1"]) if history else None

    metrics = {
        "run_id": rid,
        "name": run.name,
        "method": run.method.name,
        "train_fraction": run.data.train_fraction,
        "train_sentences": len(corpus["train"]),
        "seed": run.seed,
        "test_contract": test_contract,
        "test_trainer": {
            k.removeprefix("test_"): v
            for k, v in test_trainer.items()
            if k in ("test_precision", "test_recall", "test_f1", "test_accuracy", "test_loss")
        },
        "validation_best": best,
        "validation_history": history,
        "trainable_params": built.trainable_params,
        "total_params": built.total_params,
        "trainable_percent": built.trainable_percent,
        "trainable_params_pilot": built.trainable_params_pilot,
        "total_params_pilot": built.total_params_pilot,
        "training_seconds": round(training_seconds, 2),
        "eval_seconds": round(eval_seconds, 2),
        "fp16": fp16,
        "device": "cuda" if cuda else "cpu",
    }

    if not t.keep_checkpoints:
        shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)
    _write_json(run_dir / METRICS_FILE, metrics)  # written last: marks the run complete

    del trainer, built
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return metrics


# --------------------------------------------------------------------------- experiment


def prepare_corpus(cfg: ExperimentConfig) -> dict[str, Any]:
    """Read, subsample and tokenize the data once for all seeds of an experiment."""
    from transformers import AutoTokenizer

    splits = read_splits(cfg.resolve(cfg.data.dir))
    train_tokens, train_labels = splits["train"]
    label2id, id2label = build_label2id(train_labels)  # always from the FULL train split
    sub_tokens, sub_labels = sample_fraction(
        train_tokens, train_labels, cfg.data.train_fraction, cfg.data.subsample_seed
    )
    if cfg.data.max_train_samples is not None:
        sub_tokens = sub_tokens[: cfg.data.max_train_samples]
        sub_labels = sub_labels[: cfg.data.max_train_samples]

    prefix = cfg.model.add_prefix_space
    kwargs = {} if prefix is None else {"add_prefix_space": prefix}
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, **kwargs)
    check_word_tokenization(tokenizer, sub_tokens, prefix)
    max_len = cfg.model.max_len
    test_tokens, test_labels = splits["test"]
    return {
        "tokenizer": tokenizer,
        "label2id": label2id,
        "id2label": id2label,
        "train": _tokenize(_dataset(sub_tokens, sub_labels), tokenizer, label2id, max_len),
        "validation": _tokenize(_dataset(*splits["validation"]), tokenizer, label2id, max_len),
        "test": _tokenize(_dataset(test_tokens, test_labels), tokenizer, label2id, max_len),
        "test_words": test_tokens,
        "test_gold": test_labels,
        "test_word_ids": _word_ids(tokenizer, test_tokens, max_len),
    }


def run_experiment(
    cfg: ExperimentConfig,
    *,
    force: bool = False,
    max_train_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Run every seed of ``cfg``; completed run IDs are skipped unless ``force``."""
    if max_train_samples is not None:  # part of the resolved config, so it changes the run ID
        cfg = dataclasses.replace(
            cfg, data=dataclasses.replace(cfg.data, max_train_samples=max_train_samples)
        )
    state = code_state()
    env = environment(state)
    output_root = cfg.resolve(cfg.output_root)
    data_ref = data_fingerprint(cfg.resolve(cfg.data.dir))

    results, pending = [], []
    for run in cfg.expand():
        rid = run_id(run, state.code_ref, data_ref)
        run_dir = output_root / rid
        if (run_dir / METRICS_FILE).exists() and not force:
            print(f"[skip] {rid} {run.name} seed={run.seed}: already complete")
            results.append({"run_id": rid, "seed": run.seed, "status": "skipped"})
        else:
            pending.append((run, rid, run_dir))

    corpus = prepare_corpus(cfg) if pending else None
    for run, rid, run_dir in pending:
        if run_dir.exists():
            shutil.rmtree(run_dir)  # incomplete or forced: start clean
        print(f"[run] {rid} {run.name} seed={run.seed} -> {run_dir}")
        m = train_one(run, rid, run_dir, env, corpus, data_ref)
        print(
            f"[done] {rid} test F1 contract={m['test_contract']['f1']:.4f} "
            f"trainer={m['test_trainer']['f1']:.4f}"
        )
        results.append({"run_id": rid, "seed": run.seed, "status": "done", "metrics": m})
    return results
