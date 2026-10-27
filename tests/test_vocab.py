"""Vocabulary extension (#35). Corpus lines are SYNTHETIC, written for these tests."""

import json
from pathlib import Path

import pytest

from kazner import vocab as v
from kazner.cli import main

TINY = Path(__file__).parent / "data" / "kaznerd_tiny"
SENTENCE = ["Асан", "Бекұлы", "Алматыға", "барды", "."]
LABELS = ["B-PERSON", "I-PERSON", "B-LOCATION", "O", "O"]
CORPUS = ["Асан Бекұлы Алматыға барды .", "Кеше Алматыға барды .", "Алматыға жаңа мектеп ашылды ."]


def ids(tokenizer, words):
    return tokenizer(words, is_split_into_words=True, add_special_tokens=False)["input_ids"]


# --- spelling and candidates -------------------------------------------------------------


def test_spell_is_greedy_longest_match_with_the_base_vocabulary(tiny_tokenizer):
    vocab = v.base_vocab(tiny_tokenizer)
    assert v.spell("Алматыға", vocab) == ["Алма", "##ты", "##ға"]
    assert v.spell("##тыға", vocab) == ["##ты", "##ға"]  # continuation pieces only
    assert v.spell("облысында", vocab) == ["облыс", "##ында"]
    assert v.spell("Париж", vocab) is None  # would be [UNK]
    assert v.spell("##", vocab) is None


def test_learned_candidates_are_new_multi_character_spellable_and_used(tiny_tokenizer):
    candidates, counts = v.learn_candidates(tiny_tokenizer, CORPUS * 20, 300, 2)
    vocab = v.base_vocab(tiny_tokenizer)
    assert candidates, counts
    for piece in candidates:
        assert piece not in vocab
        assert len(piece.removeprefix("##")) >= 2
        assert v.spell(piece, vocab) is not None
    skipped = sum(counts.get(k, 0) for k in ("in_base", "single_character", "needs_unk", "unused"))
    assert counts["learned"] == skipped + counts["candidates"] == skipped + len(candidates)
    assert counts["single_character"] > 0 and counts["needs_unk"] > 0


def test_candidates_are_ranked_by_use_then_alphabetically(tiny_tokenizer):
    vocab = v.base_vocab(tiny_tokenizer)
    used = {"##тыға": 5, "Алматы": 9, "Алматыға": 5, "Алма": 50, "А": 70, "Ал": 7, "##ты": 3}
    used["Бекұлыға"] = 0  # spellable (Бек ##ұлы ##ға) but never used by the learner
    candidates, counts = v.select_candidates(used, vocab)
    assert candidates == ["Алматы", "##тыға", "Алматыға"]  # 9, then the tie 5/5 by string
    assert counts == {
        "learned": 8, "in_base": 2, "single_character": 1, "needs_unk": 1, "unused": 1,
        "candidates": 3,
    }  # fmt: skip


# --- tokenizer and model ------------------------------------------------------------------


def test_extension_keeps_old_ids_and_uses_the_new_pieces(tiny_tokenizer):
    vocab = v.base_vocab(tiny_tokenizer)
    extended = v.extend_tokenizer(tiny_tokenizer, ["Алматыға", "##тыға"])
    new_vocab = v.base_vocab(extended)
    assert {p: new_vocab[p] for p in vocab} == vocab
    assert new_vocab["Алматыға"] == len(vocab) and new_vocab["##тыға"] == len(vocab) + 1
    assert extended.tokenize("Алматыға") == ["Алматыға"]
    assert tiny_tokenizer.tokenize("Алматыға") == ["Алма", "##ты", "##ға"]  # base unchanged
    assert ids(extended, ["Асан", "Бекұлы", "барды"]) == ids(
        tiny_tokenizer, ["Асан", "Бекұлы", "барды"]
    )
    assert extended.tokenize("Париж") == tiny_tokenizer.tokenize("Париж") == ["[UNK]"]
    with pytest.raises(ValueError, match="already present"):
        v.extend_tokenizer(tiny_tokenizer, ["Алма"])


def test_an_empty_extension_tokenizes_identically(tiny_tokenizer):
    from kazner.data import read_splits

    extended = v.extend_tokenizer(tiny_tokenizer, [])
    for words, _ in read_splits(TINY).values():
        assert ids(extended, words) == ids(tiny_tokenizer, words)


@pytest.fixture
def tiny_mlm(tiny_tokenizer):
    torch = pytest.importorskip("torch")
    from transformers import BertConfig, BertForMaskedLM

    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=len(v.base_vocab(tiny_tokenizer)),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    model = BertForMaskedLM(config)
    with torch.no_grad():
        model.get_output_embeddings().bias.normal_()  # non-zero, so the FVT mean is visible
    return model


def test_fvt_initialises_new_rows_as_the_mean_of_their_pieces(tiny_tokenizer, tiny_mlm):
    import torch

    vocab = v.base_vocab(tiny_tokenizer)
    old_rows = tiny_mlm.get_input_embeddings().weight.detach().clone()
    old_bias = tiny_mlm.get_output_embeddings().bias.detach().clone()
    sources = v.fvt_sources(["Алматыға", "##тыға"], vocab)
    assert sources[0] == [vocab["Алма"], vocab["##ты"], vocab["##ға"]]

    v.extend_model(tiny_mlm, sources)
    rows = tiny_mlm.get_input_embeddings().weight.detach()
    output = tiny_mlm.get_output_embeddings()
    n = len(vocab)
    assert rows.shape[0] == n + 2 and tiny_mlm.config.vocab_size == n + 2
    assert torch.equal(rows[:n], old_rows)  # old rows untouched
    for i, ids_ in enumerate(sources):
        assert torch.allclose(rows[n + i], old_rows[ids_].mean(dim=0))
        assert torch.isclose(output.bias[n + i], old_bias[ids_].mean())
    assert output.weight.data_ptr() == rows.data_ptr()  # decoder still tied
    with pytest.raises(ValueError, match="cannot be spelled"):
        v.fvt_sources(["Париж"], vocab)


# --- report --------------------------------------------------------------------------------


def test_fertility_report_by_hand(tiny_tokenizer):
    extended = v.extend_tokenizer(tiny_tokenizer, ["Алматыға"])
    report = v.fertility_report(tiny_tokenizer, extended, [SENTENCE], [LABELS])
    assert report["words"] == 5
    assert report["all"] == {"n": 5, "before": 8 / 5, "after": 6 / 5}  # 1 2 3 1 1 -> 1 2 1 1 1
    assert report["entity"] == {"n": 3, "before": 2.0, "after": pytest.approx(4 / 3)}
    assert report["fragmented_entity"] == {"n": 0, "before": None, "after": None}
    assert report["unk_words"] == {"before": 0.0, "after": 0.0}


def test_piece_usage_counts_new_pieces_in_the_corpus(tiny_tokenizer):
    n = len(v.base_vocab(tiny_tokenizer))
    extended = v.extend_tokenizer(tiny_tokenizer, ["Алматыға", "##тыға"])
    usage = v.piece_usage(extended, CORPUS, n, 2)
    assert usage == {
        "min": 0,
        "p10": pytest.approx(0.3),
        "median": 1.5,
        "below_100": 2,
        "unused": 1,
    }


# --- end to end ----------------------------------------------------------------------------


@pytest.fixture
def base_dir(tiny_tokenizer, tiny_mlm, tmp_path):
    path = tmp_path / "base"
    tiny_tokenizer.save_pretrained(path)
    tiny_mlm.save_pretrained(path)
    return path


def test_extend_vocabulary_writes_a_loadable_model(base_dir, tmp_path):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(CORPUS * 20) + "\n", encoding="utf-8")
    report = v.extend_vocabulary(
        str(base_dir), corpus, TINY, tmp_path / "report", sizes=[1], vocab_size=300,
        min_frequency=2, log=lambda *_: None,
    )  # fmt: skip
    assert set(report["reports_on_validation"]) == {"1"}
    assert (tmp_path / "report" / "report.json").exists()
    assert not (tmp_path / "report" / "model.safetensors").exists()

    out = tmp_path / "extended"
    manifest = v.extend_vocabulary(
        str(base_dir), corpus, TINY, out, size=2, sizes=[], vocab_size=300, min_frequency=2,
        log=lambda *_: None,
    )  # fmt: skip
    assert manifest["corpus"]["sha256"] == v.sha256_of(corpus)
    assert manifest["extension"]["first_new_id"] == manifest["base"]["vocab_size"]
    pieces = (out / "new_pieces.txt").read_text(encoding="utf-8").splitlines()
    assert len(pieces) == 2
    assert (
        json.loads((out / "extension.json").read_text(encoding="utf-8"))["extension"]["size"] == 2
    )

    tokenizer = AutoTokenizer.from_pretrained(out)
    model = AutoModelForMaskedLM.from_pretrained(out)
    n = manifest["base"]["vocab_size"]
    assert len(tokenizer) == n + 2 == model.get_input_embeddings().weight.shape[0]
    assert tokenizer.convert_tokens_to_ids(pieces) == [n, n + 1]

    # The saved list rebuilds the same extension without learning again.
    again = tmp_path / "again"
    saved = v.read_pieces(out / "new_pieces.txt")
    rebuilt = v.extend_vocabulary(
        str(base_dir), corpus, TINY, again, size=2, sizes=[], pieces=saved, log=lambda *_: None
    )
    assert rebuilt["learner"] == {"pieces_given": True, "given": 2}
    assert v.read_pieces(again / "new_pieces.txt") == pieces
    assert (again / "tokenizer.json").read_bytes() == (out / "tokenizer.json").read_bytes()


def test_cli_extend_vocab_report(base_dir, tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(CORPUS * 20) + "\n", encoding="utf-8")
    args = ["extend-vocab", "--base", str(base_dir), "--corpus", str(corpus), "--data", str(TINY)]
    assert main([*args, "--out", str(tmp_path / "r"), "--sizes", "1", "--vocab-size", "300",
                 "--min-frequency", "2"]) == 0  # fmt: skip
    report = json.loads((tmp_path / "r" / "report.json").read_text(encoding="utf-8"))
    assert report["learner"]["vocab_size"] == 300


@pytest.mark.network
def test_mbert_extension_shortens_kazakh_words_and_keeps_everything_else():
    from transformers import AutoTokenizer

    base = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    words = ["Біз", "мектептерімізде", "оқимыз", "."]
    assert ids(v.extend_tokenizer(base, []), words) == ids(base, words)
    vocab = v.base_vocab(base)
    assert "мектептері" not in vocab and v.spell("мектептері", vocab) is not None
    extended = v.extend_tokenizer(base, ["мектептері"])
    assert extended.tokenize("мектептерімізде")[0] == "мектептері"
    assert len(extended.tokenize("мектептерімізде")) < len(base.tokenize("мектептерімізде"))
    assert ids(extended, ["Біз", "оқимыз", "."]) == ids(base, ["Біз", "оқимыз", "."])
