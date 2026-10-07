from pathlib import Path

import pytest

from kazner.align import (
    IGNORE_INDEX,
    align_labels,
    first_subword_positions,
    tokenize_and_align,
    truncated_words,
)
from kazner.data import build_label2id, read_splits
from reference import pilot_align

TINY = Path(__file__).parent / "data" / "kaznerd_tiny"
X = IGNORE_INDEX


@pytest.fixture(scope="module")
def corpus():
    splits = read_splits(TINY)
    label2id, _ = build_label2id(splits["train"][1])
    return splits, label2id


def batch(tokens, labels):
    return {"tokens": tokens, "labels_str": labels}


def test_word_split_into_three_pieces(tiny_tokenizer):
    label2id = {"B-LOCATION": 0, "O": 1}
    words, labels = ["Алматыға", "барды"], ["B-LOCATION", "O"]
    enc = tokenize_and_align(batch([words], [labels]), tiny_tokenizer, label2id, max_len=128)

    assert enc.tokens(0) == ["[CLS]", "Алма", "##ты", "##ға", "барды", "[SEP]"]
    assert enc["labels"][0] == [X, 0, X, X, 1, X]


def test_special_tokens_and_continuations_are_ignored(tiny_tokenizer, corpus):
    (tokens, labels), label2id = corpus[0]["train"], corpus[1]
    enc = tokenize_and_align(batch(tokens, labels), tiny_tokenizer, label2id, max_len=128)
    for i, word_labels in enumerate(labels):
        word_ids = enc.word_ids(batch_index=i)
        aligned = enc["labels"][i]
        assert len(aligned) == len(word_ids)
        scored = [lab for lab in aligned if lab != X]
        assert scored == [label2id[lab] for lab in word_labels]  # exactly one per word


def test_unknown_word_still_gets_its_label(tiny_tokenizer):
    enc = tokenize_and_align(
        batch([["Ақтөбе", "."]], [["B-GPE", "O"]]), tiny_tokenizer, {"B-GPE": 0, "O": 1}, 128
    )
    assert enc.tokens(0) == ["[CLS]", "[UNK]", ".", "[SEP]"]
    assert enc["labels"][0] == [X, 0, 1, X]


def test_truncation_drops_trailing_words(tiny_tokenizer):
    words = ["Асан", "Бекұлы", "Алматыға", "барды", "."]
    labels = ["B-PERSON", "I-PERSON", "B-LOCATION", "O", "O"]
    label2id = build_label2id([labels])[0]
    enc = tokenize_and_align(batch([words], [labels]), tiny_tokenizer, label2id, max_len=6)

    word_ids = enc.word_ids(0)
    assert enc.tokens(0) == ["[CLS]", "Асан", "Бек", "##ұлы", "Алма", "[SEP]"]
    assert first_subword_positions(word_ids) == {0: 1, 1: 2, 2: 4}
    assert truncated_words(word_ids, len(words)) == [3, 4]
    # A word cut in the middle keeps its label on the first piece, as in the pilot.
    assert enc["labels"][0] == [X, label2id["B-PERSON"], label2id["I-PERSON"], X, 0, X]


@pytest.mark.parametrize("max_len", [4, 6, 9, 128])
@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_identical_to_pilot(tiny_tokenizer, corpus, monkeypatch, split, max_len):
    (tokens, labels), label2id = corpus[0][split], corpus[1]
    tokens = [*tokens, ["Ақтөбе", "Алматыға"]]  # include an [UNK] word and a 3-piece word
    labels = [*labels, ["B-LOCATION", "B-LOCATION"]]
    monkeypatch.setattr(pilot_align, "MAX_LEN", max_len)

    ours = tokenize_and_align(batch(tokens, labels), tiny_tokenizer, label2id, max_len)
    oracle = pilot_align.tokenize_and_align(batch(tokens, labels), tiny_tokenizer, label2id)

    assert ours["labels"] == oracle["labels"]
    assert ours["input_ids"] == oracle["input_ids"]


def test_align_labels_on_hand_written_word_ids():
    label2id = {"B-PERSON": 0, "I-PERSON": 1, "O": 2}
    word_ids = [None, 0, 0, 1, 2, 2, 2, None]
    assert align_labels(word_ids, ["B-PERSON", "I-PERSON", "O"], label2id) == [
        X, 0, X, 1, 2, X, X, X,
    ]  # fmt: skip


def test_first_subword_positions_without_truncation():
    assert first_subword_positions([None, 0, 0, 1, 2, 2, None]) == {0: 1, 1: 3, 2: 4}
    assert truncated_words([None, 0, 0, 1, 2, 2, None], 3) == []


@pytest.mark.network
def test_mbert_alignment_matches_pilot(corpus, monkeypatch):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    (tokens, labels), label2id = corpus[0]["train"], corpus[1]
    monkeypatch.setattr(pilot_align, "MAX_LEN", 128)
    ours = tokenize_and_align(batch(tokens, labels), tokenizer, label2id, 128)
    oracle = pilot_align.tokenize_and_align(batch(tokens, labels), tokenizer, label2id)
    assert ours["labels"] == oracle["labels"]
    assert any(lab == X for row in ours["labels"] for lab in row[1:-1])  # mBERT splits words
