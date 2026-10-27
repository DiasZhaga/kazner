"""Continued masked-LM pre-training (#36). Corpus lines are SYNTHETIC, written for these tests."""

import json
from pathlib import Path

import numpy as np
import pytest

from kazner import pretrain as p
from kazner.config import ConfigError
from kazner.vocab import extend_tokenizer

TINY = Path(__file__).parent / "data" / "kaznerd_tiny"
LINES = ["Асан Бекұлы Алматыға барды .", "Кеше барды ."]
MINIMAL = {"name": "arm", "corpus": {"path": "corpus.txt"}, "model": {"name": "org/model"}}


# --- chunks and masks -----------------------------------------------------------------------


def test_chunks_pack_consecutive_words_up_to_the_budget(tiny_tokenizer):
    pieces = p.piece_counter(tiny_tokenizer)  # Асан 1, Бекұлы 2, Алматыға 3, барды 1, . 1, Кеше 1
    chunks, oversized = p.chunk_words(LINES, pieces, budget=4, max_chunks=None)
    assert chunks == [["Асан", "Бекұлы"], ["Алматыға", "барды"], [".", "Кеше", "барды", "."]]
    assert oversized == 0
    assert p.chunk_words(LINES, pieces, budget=4, max_chunks=2)[0] == chunks[:2]
    chunks, oversized = p.chunk_words(LINES, pieces, budget=2, max_chunks=None)
    assert ["Алматыға"] in chunks and oversized == 1  # alone, truncated later


def test_word_actions_are_seeded_per_chunk_and_have_the_configured_shares():
    masking = p.MaskingConfig()
    first = p.word_actions(20, chunk_index=7, seed=42, masking=masking)
    assert np.array_equal(first, p.word_actions(20, 7, 42, masking))
    assert not np.array_equal(first, p.word_actions(20, 8, 42, masking))
    assert (first > 0).sum() == 3  # round(0.15 * 20)
    assert (p.word_actions(2, 0, 42, masking) > 0).sum() == 1  # at least one word
    actions = np.concatenate([p.word_actions(20, i, 42, masking) for i in range(3000)])
    selected = actions[actions > 0]
    assert len(selected) == 9000
    assert np.mean(selected == p.MASK) == pytest.approx(0.8, abs=0.02)
    assert np.mean(selected == p.RANDOM) == pytest.approx(0.1, abs=0.02)


def test_mask_example_predicts_every_piece_of_a_selected_word():
    input_ids = [101, 5, 6, 7, 8, 102]
    word_ids = [None, 0, 0, 1, 2, None]  # word 0 has two pieces
    actions = np.array([p.MASK, 0, p.KEEP], dtype=np.int8)
    inputs, labels = p.mask_example(input_ids, word_ids, actions, np.random.default_rng(0), 103, 50)
    assert inputs == [101, 103, 103, 7, 8, 102]
    assert labels == [-100, 5, 6, -100, 8, -100]


def masked_words(tokenizer, chunks, examples):
    """Which words of each chunk are predicted, recovered through the tokenizer's word ids."""
    out = []
    for i, words in enumerate(chunks):
        enc = tokenizer(list(words), is_split_into_words=True)
        out.append(
            {w for w, y in zip(enc.word_ids(), examples["labels"][i], strict=True) if y != -100}
        )
    return out


def test_both_vocabularies_get_the_same_masked_words(tiny_tokenizer):
    extended = extend_tokenizer(tiny_tokenizer, ["Алматыға", "##тыға"])
    chunks = [["Асан", "Бекұлы", "Алматыға", "барды", "."]] * 3 + [["Кеше", "Алматыға", "."]] * 3
    masking = p.MaskingConfig(probability=0.4)
    base = p.build_examples(chunks, tiny_tokenizer, 16, masking, seed=1)
    ext = p.build_examples(chunks, extended, 16, masking, seed=1)
    assert masked_words(tiny_tokenizer, chunks, base) == masked_words(extended, chunks, ext)
    lengths = [len(x) for x in base["input_ids"]], [len(x) for x in ext["input_ids"]]
    assert sum(lengths[1]) < sum(lengths[0])  # the extended vocabulary needs fewer pieces
    for chunk, inputs, labels in zip(chunks, base["input_ids"], base["labels"], strict=True):
        original = tiny_tokenizer(list(chunk), is_split_into_words=True)["input_ids"]
        pairs = list(zip(original, inputs, labels, strict=True))
        assert labels[0] == labels[-1] == -100  # [CLS], [SEP]
        assert any(y != -100 for y in labels)
        assert all(x == o for o, x, y in pairs if y == -100)  # unmasked positions untouched
        assert all(y == o for o, x, y in pairs if y != -100)  # labels are the original pieces
    assert any(x == tiny_tokenizer.mask_token_id for inputs in base["input_ids"] for x in inputs)


# --- config and run ID ------------------------------------------------------------------------


def cfg(**changes):
    raw = json.loads(json.dumps(MINIMAL))
    for dotted, value in changes.items():
        node = raw
        *parents, last = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[last] = value
    return p.parse_pretrain_config(raw)


def test_defaults_and_validation():
    c = cfg()
    assert (c.masking.probability, c.training.epochs, c.training.warmup) == (0.15, 1.0, 0.05)
    assert c.chunker.name == "bert-base-multilingual-cased"
    for changes, message in [
        ({"masking.probability": 1.0}, r"masking\.probability: must be in \(0, 1\)"),
        ({"masking.mask_share": 0.95}, r"mask_share \+ random_share must be <= 1"),
        ({"training.warmup": 1.0}, r"training\.warmup: must be in \[0, 1\)"),
        ({"corpus.max_chunks": 0}, r"corpus\.max_chunks: must be positive"),
        ({"model.name": "./m", "model.revision": "main"}, r"model\.revision: only applies"),
        ({"training.lr": 1e-4}, r"training\.lr: unknown key"),
    ]:
        with pytest.raises(ConfigError, match=message):
            cfg(**changes)


def test_run_id_hashes_content_not_locations():
    a = cfg(**{"corpus.path": "a/corpus.txt", "model.name": "./models/x"})
    b = cfg(**{"corpus.path": "elsewhere/corpus.txt", "model.name": "../other/x"})
    assert p.pretrain_run_id(a, "tree", "c" * 64, "m" * 64) == p.pretrain_run_id(
        b, "tree", "c" * 64, "m" * 64
    )
    assert p.pretrain_run_id(a, "tree", "c" * 64, "m" * 64) != p.pretrain_run_id(
        a, "tree", "d" * 64, "m" * 64
    )
    assert p.pretrain_run_id(cfg(), "tree", "c", "") != p.pretrain_run_id(
        cfg(**{"masking.probability": 0.2}), "tree", "c", ""
    )
    limited = p.with_max_chunks(cfg(), 10)
    assert limited.corpus.max_chunks == 10 and p.with_max_chunks(cfg(), None) == cfg()


# --- end to end -----------------------------------------------------------------------------


@pytest.mark.smoke
def test_pretraining_writes_a_model_that_ner_training_can_use(tmp_path):
    from transformers import AutoModelForMaskedLM

    from kazner.config import parse_config
    from kazner.train import run_experiment

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(LINES * 10) + "\n", encoding="utf-8")
    tiny = "hf-internal-testing/tiny-random-bert"
    raw = {
        "name": "smoke",
        "corpus": {"path": str(corpus)},
        "model": {"name": tiny, "max_len": 16},
        "chunker": {"name": tiny, "max_len": 16},
        "training": {"train_batch_size": 2, "grad_accumulation": 1, "logging_steps": 1},
        "output_root": str(tmp_path / "models"),
    }
    config = p.parse_pretrain_config(raw)
    metrics = p.run_pretraining(config, log=lambda *_: None)
    run_dir = tmp_path / "models" / metrics["run_id"]
    assert metrics["steps"] > 0 and np.isfinite(metrics["loss_last"])
    assert metrics["predicted_tokens"] > 0 and metrics["chunks"] > 0
    for name in ("model.safetensors", "tokenizer.json", "pretrain_config.json", "environment.json"):
        assert (run_dir / name).exists(), name
    assert not (run_dir / "checkpoints").exists()
    AutoModelForMaskedLM.from_pretrained(run_dir)
    assert p.run_pretraining(config, log=lambda *_: None)["run_id"] == metrics["run_id"]  # skipped

    experiment = parse_config(
        {
            "name": "ner-on-pretrained",
            "data": {"dir": str(TINY)},
            "model": {"name": str(run_dir), "max_len": 16},
            "training": {"epochs": 1, "train_batch_size": 2, "eval_batch_size": 2},
            "output_root": str(tmp_path / "runs"),
        }
    )
    (result,) = run_experiment(experiment)
    assert result["status"] == "done"


def test_shipped_arms_differ_only_in_the_model():
    root = Path(__file__).resolve().parent.parent / "configs" / "pretrain"
    original = p.load_pretrain_config(root / "mbert_kk_original.yaml")
    extended = p.load_pretrain_config(root / "mbert_kk_ext10k.yaml")
    a, b = original.hashable_dict(), extended.hashable_dict()
    assert a.pop("model") != b.pop("model")
    assert a == b  # same corpus rules, chunker, masking, training and seed
    assert original.resolve(original.corpus.path) == extended.resolve(extended.corpus.path)
    assert original.chunker.revision == "3f076fdb1ab68d5b2880cb87a0886f315b8146f8"
    assert extended.location(extended.model).endswith("mbert-kk-10k")


def test_run_pretraining_offline_on_a_tiny_local_model(tiny_tokenizer, tmp_path):
    torch = pytest.importorskip("torch")
    from transformers import BertConfig, BertForMaskedLM

    model_dir = tmp_path / "tiny"
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=len(tiny_tokenizer), hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32, max_position_embeddings=32,
    )  # fmt: skip
    BertForMaskedLM(config).save_pretrained(model_dir)
    tiny_tokenizer.save_pretrained(model_dir)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(LINES * 5) + "\n", encoding="utf-8")
    raw = {
        "name": "offline",
        "corpus": {"path": "corpus.txt", "max_chunks": 6},
        "model": {"name": "./tiny", "max_len": 16},
        "chunker": {"name": "./tiny", "max_len": 8},
        "training": {"train_batch_size": 2, "grad_accumulation": 1, "logging_steps": 1},
        "output_root": "models",
    }
    config = p.parse_pretrain_config(raw, base_dir=tmp_path)
    logged = []
    metrics = p.run_pretraining(config, log=logged.append)
    assert metrics["chunks"] == 6 and metrics["steps"] == 3
    run_dir = tmp_path / "models" / metrics["run_id"]
    saved = json.loads((run_dir / "pretrain_config.json").read_text(encoding="utf-8"))
    assert saved["model_fingerprint"] and saved["corpus_sha256"] == p.corpus_fingerprint(corpus)
    assert (run_dir / p.METRICS_FILE).exists() and (run_dir / "model.safetensors").exists()
    assert p.run_pretraining(config, log=logged.append)["run_id"] == metrics["run_id"]
    assert logged[-1].startswith("[skip]")
    forced = p.run_pretraining(config, force=True, log=logged.append)
    assert forced["run_id"] == metrics["run_id"] and logged[-1].startswith("[done]")
