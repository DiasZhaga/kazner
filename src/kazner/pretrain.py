"""Continued masked-LM pre-training (#36): ``kazner pretrain --config X.yaml``.

The two arms of the causal test (#37), mBERT with its original vocabulary and mBERT with the
extended vocabulary of #35, are trained on the same text in the same way. Only the vocabulary
differs.

* **Same text, same order.** The corpus is cut into chunks of consecutive words that fit
  ``chunker.max_len - 2`` pieces under a fixed *reference* tokenizer (original mBERT), whatever
  model is trained. Both arms therefore get identical word chunks, the same number of steps,
  and (with the same seed) the same batch order.
* **Same masks.** Whole-word masking: in every chunk the same words are selected in both arms,
  from a generator seeded by ``(seed, chunk index)``, together with the same per-word decision
  (``[MASK]`` / random token / kept). All pieces of a selected word are predicted. Only the
  random replacement token depends on the vocabulary. Masks are static: drawn once per chunk.
* **Identity.** A run is written to ``<output_root>/<run_id>/``. The run ID hashes the
  resolved config (without name and paths), the code, the SHA-256 of the corpus and, for a
  local model directory, its content fingerprint. The directory is a complete model
  (``save_pretrained``), usable as ``model.name`` in an experiment config (#31). It also holds
  ``pretrain_config.json``, ``environment.json`` and, written last, ``pretrain_metrics.json``.
  A run with that file is complete and is skipped; an interrupted run resumes from its last
  checkpoint.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from kazner.config import (
    RUN_ID_LENGTH,
    ConfigError,
    ModelConfig,
    build_dataclass,
    canonical_json,
    code_state,
    environment,
    is_local_model,
    model_fingerprint,
)

METRICS_FILE = "pretrain_metrics.json"
#: Per-word masking decisions; 0 = not selected.
MASK, RANDOM, KEEP = 1, 2, 3


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class CorpusConfig:
    path: str
    #: Use only the first N chunks (pilots and smoke tests); part of the run ID.
    max_chunks: int | None = None


@dataclass(frozen=True)
class MaskingConfig:
    probability: float = 0.15
    #: Of the selected words: replaced by [MASK], by a random token, and (the rest) kept.
    mask_share: float = 0.8
    random_share: float = 0.1


@dataclass(frozen=True)
class PretrainTrainingConfig:
    learning_rate: float = 5e-5
    epochs: float = 1.0
    train_batch_size: int = 8
    grad_accumulation: int = 4
    weight_decay: float = 0.01
    #: Fraction of the optimizer steps with a linear warm-up.
    warmup: float = 0.05
    fp16: str = "auto"
    logging_steps: int = 100
    #: Checkpoint interval for resuming an interrupted run.
    save_steps: int = 2000


@dataclass(frozen=True)
class PretrainConfig:
    name: str
    corpus: CorpusConfig
    model: ModelConfig
    #: Reference tokenizer that cuts the corpus into chunks: original mBERT for both arms.
    chunker: ModelConfig = field(default_factory=ModelConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    training: PretrainTrainingConfig = field(default_factory=PretrainTrainingConfig)
    seed: int = 42
    output_root: str = "../models"
    base_dir: Path = field(default=Path("."), compare=False, repr=False)

    def resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    def location(self, model: ModelConfig) -> str:
        return str(self.resolve(model.name)) if is_local_model(model.name) else model.name

    def to_dict(self) -> dict[str, Any]:
        return {
            f.name: _plain(getattr(self, f.name)) for f in dataclasses.fields(self) if f.compare
        }

    def hashable_dict(self) -> dict[str, Any]:
        d = self.to_dict()
        for key in ("name", "output_root"):
            d.pop(key)
        d["corpus"].pop("path")  # content is hashed instead
        for key in ("model", "chunker"):
            if is_local_model(d[key]["name"]):
                d[key].pop("name")  # fingerprinted instead
        return d


def _plain(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {f.name: _plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, tuple | list):
        return [_plain(v) for v in obj]
    return obj


def parse_pretrain_config(raw: Any, base_dir: Path = Path(".")) -> PretrainConfig:
    training = raw.get("training") if isinstance(raw, dict) else None
    if isinstance(training, dict) and isinstance(training.get("fp16"), bool):
        raw = {**raw, "training": {**training, "fp16": str(training["fp16"]).lower()}}
    cfg = build_dataclass(PretrainConfig, raw)
    cfg = dataclasses.replace(cfg, base_dir=Path(base_dir))

    def check(ok: bool, key: str, message: str) -> None:
        if not ok:
            raise ConfigError(f"{key}: {message}")

    m, t = cfg.masking, cfg.training
    check(0 < m.probability < 1, "masking.probability", "must be in (0, 1)")
    check(m.mask_share >= 0 and m.random_share >= 0, "masking", "shares must not be negative")
    check(m.mask_share + m.random_share <= 1, "masking", "mask_share + random_share must be <= 1")
    check(cfg.chunker.max_len > 2, "chunker.max_len", "must be greater than 2")
    check(cfg.model.max_len > 2, "model.max_len", "must be greater than 2")
    mc = cfg.corpus.max_chunks
    check(mc is None or mc > 0, "corpus.max_chunks", "must be positive")
    check(t.learning_rate > 0, "training.learning_rate", "must be positive")
    check(t.epochs > 0, "training.epochs", "must be positive")
    check(0 <= t.warmup < 1, "training.warmup", "must be in [0, 1)")
    for key in ("train_batch_size", "grad_accumulation", "logging_steps", "save_steps"):
        check(getattr(t, key) > 0, f"training.{key}", "must be positive")
    check(t.fp16 in ("auto", "true", "false"), "training.fp16", "must be auto, true or false")
    for key in ("model", "chunker"):
        mcfg = getattr(cfg, key)
        check(
            mcfg.revision is None or not is_local_model(mcfg.name),
            f"{key}.revision",
            "only applies to Hugging Face Hub models, not to a local directory",
        )
    return cfg


def load_pretrain_config(path: str | Path) -> PretrainConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    try:
        return parse_pretrain_config(raw, base_dir=path.resolve().parent)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from None


def corpus_fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pretrain_run_id(cfg: PretrainConfig, code_ref: str, corpus_ref: str, model_ref: str) -> str:
    payload = canonical_json(
        {"config": cfg.hashable_dict(), "code": code_ref, "corpus": corpus_ref, "model": model_ref}
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:RUN_ID_LENGTH]


# --------------------------------------------------------------------------- chunks and masks


def piece_counter(tokenizer: Any) -> Callable[[str], int]:
    """``word -> pieces`` under ``tokenizer`` (isolated, at least one), cached."""
    cache: dict[str, int] = {}

    def count(word: str) -> int:
        n = cache.get(word)
        if n is None:
            n = cache[word] = max(1, len(tokenizer.tokenize(word)))
        return n

    return count


def chunk_words(
    lines: Iterable[str], pieces_of: Callable[[str], int], budget: int, max_chunks: int | None
) -> tuple[list[list[str]], int]:
    """Consecutive words (lines concatenated in order) packed greedily into chunks of at most
    ``budget`` reference pieces. A single word above the budget forms its own chunk (it is
    truncated later). Returns the chunks and the number of such oversized words."""
    chunks: list[list[str]] = []
    current: list[str] = []
    used = oversized = 0
    for line in lines:
        for word in line.split():
            n = pieces_of(word)
            oversized += n > budget
            if current and used + n > budget:
                chunks.append(current)
                if max_chunks is not None and len(chunks) == max_chunks:
                    return chunks, oversized
                current, used = [], 0
            current.append(word)
            used += n
    if current and (max_chunks is None or len(chunks) < max_chunks):
        chunks.append(current)
    return chunks, oversized


def word_actions(n_words: int, chunk_index: int, seed: int, masking: MaskingConfig) -> np.ndarray:
    """Per-word masking decision for one chunk: 0, MASK, RANDOM or KEEP. Depends only on the
    chunk's length, its index and the seed, so every vocabulary gets the same decisions."""
    rng = np.random.default_rng([seed, chunk_index])
    n_selected = min(n_words, max(1, round(masking.probability * n_words)))
    selected = rng.choice(n_words, size=n_selected, replace=False)
    draw = rng.random(n_selected)
    actions = np.zeros(n_words, dtype=np.int8)
    random_until = masking.mask_share + masking.random_share
    actions[selected] = np.where(
        draw < masking.mask_share, MASK, np.where(draw < random_until, RANDOM, KEEP)
    )
    return actions


def mask_example(
    input_ids: Sequence[int],
    word_ids: Sequence[int | None],
    actions: np.ndarray,
    rng: np.random.Generator,
    mask_id: int,
    vocab_size: int,
) -> tuple[list[int], list[int]]:
    """Apply the word decisions to one tokenized chunk: masked inputs and labels (-100 = not
    predicted). Every piece of a selected word is predicted."""
    inputs, labels = list(input_ids), [-100] * len(input_ids)
    for position, word in enumerate(word_ids):
        if word is None or actions[word] == 0:
            continue
        labels[position] = inputs[position]
        if actions[word] == MASK:
            inputs[position] = mask_id
        elif actions[word] == RANDOM:
            inputs[position] = int(rng.integers(vocab_size))
    return inputs, labels


def build_examples(
    chunks: Sequence[Sequence[str]],
    tokenizer: Any,
    max_len: int,
    masking: MaskingConfig,
    seed: int,
) -> dict[str, list[list[int]]]:
    """Tokenize every chunk with the model's tokenizer and apply the shared word masks."""
    inputs: list[list[int]] = []
    labels: list[list[int]] = []
    for start in range(0, len(chunks), 2000):
        batch = [list(c) for c in chunks[start : start + 2000]]
        enc = tokenizer(batch, is_split_into_words=True, truncation=True, max_length=max_len)
        for i, words in enumerate(batch):
            index = start + i
            actions = word_actions(len(words), index, seed, masking)
            rng = np.random.default_rng([seed, index, 1])  # random replacement tokens only
            x, y = mask_example(
                enc["input_ids"][i],
                enc.word_ids(batch_index=i),
                actions,
                rng,
                tokenizer.mask_token_id,
                len(tokenizer),
            )
            inputs.append(x)
            labels.append(y)
    return {"input_ids": inputs, "attention_mask": [[1] * len(x) for x in inputs], "labels": labels}


# --------------------------------------------------------------------------- run


def _read_lines(path: Path) -> Iterable[str]:
    with path.open(encoding="utf-8") as f:
        yield from f


def run_pretraining(
    cfg: PretrainConfig, *, force: bool = False, log: Callable[[str], None] = print
) -> dict[str, Any]:
    """Train (or skip, if complete) one continued pre-training run; returns its metrics."""
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForMaskedLM,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )
    from transformers.trainer_utils import get_last_checkpoint

    from kazner.train import resolve_fp16, set_all_seeds

    state = code_state()
    corpus = cfg.resolve(cfg.corpus.path)
    corpus_ref = corpus_fingerprint(corpus)
    location = cfg.location(cfg.model)
    model_ref = model_fingerprint(location) if is_local_model(cfg.model.name) else ""
    rid = pretrain_run_id(cfg, state.code_ref, corpus_ref, model_ref)
    run_dir = cfg.resolve(cfg.output_root) / rid
    if (run_dir / METRICS_FILE).exists() and not force:
        log(f"[skip] {rid} {cfg.name}: already complete -> {run_dir}")
        return json.loads((run_dir / METRICS_FILE).read_text(encoding="utf-8"))
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    def hub(model: ModelConfig) -> dict[str, Any]:
        return {} if model.revision is None else {"revision": model.revision}

    chunker = AutoTokenizer.from_pretrained(cfg.location(cfg.chunker), **hub(cfg.chunker))
    budget = cfg.chunker.max_len - 2
    chunks, oversized = chunk_words(
        _read_lines(corpus), piece_counter(chunker), budget, cfg.corpus.max_chunks
    )
    tokenizer = AutoTokenizer.from_pretrained(location, **hub(cfg.model))
    examples = build_examples(chunks, tokenizer, cfg.model.max_len, cfg.masking, cfg.seed)
    dataset = Dataset.from_dict(examples)
    n_tokens = sum(len(x) for x in examples["input_ids"])
    n_predicted = sum(sum(v != -100 for v in y) for y in examples["labels"])
    n_words = sum(len(c) for c in chunks)
    log(f"[run] {rid} {cfg.name}: {len(chunks):,} chunks, {n_words:,} words, {n_tokens:,} tokens")

    env = {**environment(state), "model": {"name": cfg.model.name, "fingerprint": model_ref}}
    config = {"run_id": rid, "corpus_sha256": corpus_ref, "model_fingerprint": model_ref or None}
    for name, payload in (
        ("pretrain_config.json", {**config, **cfg.to_dict()}),
        ("environment.json", env),
    ):
        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        (run_dir / name).write_text(text, encoding="utf-8", newline="\n")

    t = cfg.training
    fp16 = resolve_fp16(t.fp16)
    cuda = torch.cuda.is_available()
    set_all_seeds(cfg.seed)
    model = AutoModelForMaskedLM.from_pretrained(location, **hub(cfg.model))
    # The Trainer sees ``**kwargs`` in forward and leaves the loss undivided by the accumulation
    # steps, but the model ignores ``num_items_in_batch``: without this, the logged loss and the
    # gradient would be the sum of the micro-batch means (#49).
    model.accepts_loss_kwargs = False
    args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        per_device_train_batch_size=t.train_batch_size,
        gradient_accumulation_steps=t.grad_accumulation,
        learning_rate=t.learning_rate,
        weight_decay=t.weight_decay,
        num_train_epochs=t.epochs,
        warmup_steps=t.warmup,  # a float below 1 is a fraction of the steps in transformers 5
        lr_scheduler_type="linear",
        fp16=fp16,
        seed=cfg.seed,
        data_seed=cfg.seed,
        logging_strategy="steps",
        logging_steps=t.logging_steps,
        save_strategy="steps",
        save_steps=t.save_steps,
        save_total_limit=1,
        eval_strategy="no",
        report_to="none",
        dataloader_pin_memory=cuda,
        use_cpu=not cuda,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )
    checkpoints = run_dir / "checkpoints"
    resume = get_last_checkpoint(str(checkpoints)) if checkpoints.is_dir() else None
    start = time.perf_counter()
    trainer.train(resume_from_checkpoint=resume)
    seconds = time.perf_counter() - start

    model.save_pretrained(run_dir)
    tokenizer.save_pretrained(run_dir)
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    metrics = {
        "run_id": rid,
        "name": cfg.name,
        "chunks": len(chunks),
        "words": n_words,
        "tokens": n_tokens,
        "predicted_tokens": n_predicted,
        "oversized_words": oversized,
        "steps": trainer.state.global_step,
        "resumed_from": resume,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_history": [h for h in trainer.state.log_history if "loss" in h],
        "training_seconds": round(seconds, 2),
        "fp16": fp16,
        "device": "cuda" if cuda else "cpu",
    }
    shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)
    tmp = run_dir / (METRICS_FILE + ".part")
    tmp.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(run_dir / METRICS_FILE)  # written last: marks the run complete
    log(f"[done] {rid} {cfg.name}: {metrics['steps']} steps, last loss {metrics['loss_last']}")
    return metrics


def with_max_chunks(cfg: PretrainConfig, max_chunks: int | None) -> PretrainConfig:
    """``cfg`` limited to its first ``max_chunks`` chunks (changes the run ID)."""
    if max_chunks is None:
        return cfg
    return dataclasses.replace(cfg, corpus=dataclasses.replace(cfg.corpus, max_chunks=max_chunks))
