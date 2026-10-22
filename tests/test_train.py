import json
from pathlib import Path

import numpy as np
import pytest

from kazner.align import IGNORE_INDEX
from kazner.config import ConfigError, parse_config
from kazner.contract import read_predictions
from kazner.data import read_splits
from kazner.train import (
    build_compute_metrics,
    check_word_tokenization,
    prepare_corpus,
    resolve_fp16,
    word_predictions,
)

TINY = Path(__file__).parent / "data" / "kaznerd_tiny"
#: transformers builds a RoBERTa tokenizer's BPE from vocab.json/merges.txt paths (as for
#: KazRoBERTa), which `tokenizers` deprecates; not ours to fix.
BPE_FROM_FILES = "ignore:Deprecated in 0.9.0:DeprecationWarning"
ID2LABEL = {0: "B-PER", 1: "I-PER", 2: "O"}
X = IGNORE_INDEX


def one_hot(rows):
    """Logits whose argmax is the given label id at every position."""
    out = np.full((len(rows), max(map(len, rows)), 3), -1.0)
    for i, row in enumerate(rows):
        for j, label in enumerate(row):
            out[i, j, label] = 1.0
    return out


def test_compute_metrics_uses_first_subword_positions_only():
    logits = one_hot([[2, 0, 0, 1, 2]])  # predicted: O B B I O
    labels = np.array([[X, 0, X, 1, X]])  # scored positions 1 and 3: gold B-PER I-PER
    m = build_compute_metrics(ID2LABEL)((logits, labels))
    assert m == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0}


def test_compute_metrics_detects_errors():
    logits = one_hot([[2, 2, 2, 2, 2]])
    labels = np.array([[X, 0, X, 1, X]])
    m = build_compute_metrics(ID2LABEL)((logits, labels))
    assert (m["f1"], m["accuracy"]) == (0.0, 0.0)


def test_word_predictions_take_the_first_subword_and_mark_truncation():
    # sentence 0: [CLS] w0 w0 w1 [SEP]; sentence 1 lost its last word to truncation
    logits = one_hot([[2, 0, 2, 1, 2], [2, 2, 2]])
    word_ids = [[None, 0, 0, 1, None], [None, 0, None]]
    words = [["Асан", "Бекұлы"], ["Мен", "кітап"]]
    gold = [["B-PER", "I-PER"], ["O", "O"]]

    p = word_predictions(logits, word_ids, words, gold, ID2LABEL)

    assert p.pred == [["B-PER", "I-PER"], ["O", "O"]]
    assert p.truncated == [[False, False], [False, True]]
    assert p.sentence_ids == [0, 1]
    assert p.n_truncated_words == 1


def test_fp16_resolution_on_cpu(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_fp16("auto") is False
    assert resolve_fp16("false") is False
    with pytest.raises(ValueError, match="CUDA is not available"):
        resolve_fp16("true")


@pytest.mark.filterwarnings(BPE_FROM_FILES)
def test_word_tokenization_check_names_the_config_key(bytelevel_dir):
    from transformers import AutoTokenizer

    sentences = read_splits(TINY)["train"][0]
    fixed = AutoTokenizer.from_pretrained(bytelevel_dir, add_prefix_space=True)
    check_word_tokenization(fixed, sentences, True)  # no error
    with pytest.raises(
        ConfigError,
        match=r"model\.add_prefix_space: 16 of 20 training words .* add_prefix_space is None",
    ):
        check_word_tokenization(AutoTokenizer.from_pretrained(bytelevel_dir), sentences, None)


@pytest.mark.filterwarnings(BPE_FROM_FILES)
def test_prepare_corpus_passes_add_prefix_space_to_the_tokenizer(bytelevel_dir):
    raw = {
        "name": "bytelevel",
        "data": {"dir": str(TINY)},
        "model": {"name": str(bytelevel_dir), "max_len": 32},
    }
    with pytest.raises(ConfigError, match=r"need 'add_prefix_space: true'"):
        prepare_corpus(parse_config(raw))

    raw["model"]["add_prefix_space"] = True
    corpus = prepare_corpus(parse_config(raw))
    assert corpus["tokenizer"].add_prefix_space is True
    assert len(corpus["train"]) == 4


@pytest.mark.smoke
def test_end_to_end_training_on_a_tiny_model(tmp_path, capsys):
    from kazner.cli import main
    from kazner.train import run_experiment

    raw = {
        "name": "smoke",
        "data": {"dir": str(TINY), "train_fraction": 1.0},
        "model": {"name": "hf-internal-testing/tiny-random-bert", "max_len": 8},
        "method": {"name": "lora"},
        "training": {
            "learning_rate": 1e-3,
            "epochs": 1,
            "train_batch_size": 2,
            "eval_batch_size": 2,
            "grad_accumulation": 1,
            "logging_steps": 1,
        },
        "seeds": [1, 2],
        "output_root": str(tmp_path / "runs"),
    }
    cfg = parse_config(raw)

    results = run_experiment(cfg)
    assert [r["status"] for r in results] == ["done", "done"]
    run_dir = tmp_path / "runs" / results[0]["run_id"]
    for name in ("config.json", "environment.json", "trainer_state.json", "predictions.csv"):
        assert (run_dir / name).exists(), name
    assert not (run_dir / "checkpoints").exists()

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["test_contract"]) >= {"precision", "recall", "f1", "accuracy"}
    assert set(metrics["test_trainer"]) >= {"precision", "recall", "f1", "accuracy", "loss"}
    assert metrics["validation_best"]["eval_f1"] >= 0
    assert metrics["trainable_params"] < metrics["trainable_params_pilot"]  # LoRA
    contract = read_predictions(run_dir / "predictions.csv")
    assert contract.n_words == 11  # every gold word of the tiny test split
    assert metrics["test_contract"]["n_truncated_words"] > 0  # max_len 8 cuts sentences

    # Idempotent: completed run IDs are skipped; --force and a new sample size re-run.
    assert [r["status"] for r in run_experiment(cfg)] == ["skipped", "skipped"]
    assert [r["status"] for r in run_experiment(cfg, force=True)][0] == "done"
    small = run_experiment(cfg, max_train_samples=2)
    assert small[0]["run_id"] != results[0]["run_id"]
    assert small[0]["metrics"]["train_sentences"] == 2

    yaml_path = tmp_path / "smoke.yaml"
    yaml_path.write_text(json.dumps(raw), encoding="utf-8")  # JSON is valid YAML
    assert main(["train", "--config", str(yaml_path)]) == 0
    assert "0 run(s) trained, 2 skipped" in capsys.readouterr().out
