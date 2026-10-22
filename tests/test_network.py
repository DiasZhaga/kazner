"""Tests that need the Hugging Face Hub. Run in the CI ``smoke`` job only."""

import pytest

TINY_MODEL = "hf-internal-testing/tiny-random-bert"


@pytest.mark.network
def test_tiny_tokenizer_maps_subwords_to_words():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    encoding = tokenizer(["Астана", "қаласы"], is_split_into_words=True)
    word_ids = encoding.word_ids()

    assert word_ids[0] is None and word_ids[-1] is None  # [CLS] and [SEP]
    assert {w for w in word_ids if w is not None} == {0, 1}


# Synthetic sentence; "мектептерімізде" is split by every tokenizer below.
SENTENCE = ["Біз", "Астанада", "мектептерімізде", "оқимыз", "."]


@pytest.mark.network
@pytest.mark.parametrize("name", ["bert-base-multilingual-cased", "FacebookAI/xlm-roberta-base"])
def test_wordpiece_and_sentencepiece_need_no_prefix_space(name):
    from transformers import AutoTokenizer

    from kazner.align import split_mismatches

    default = AutoTokenizer.from_pretrained(name)
    flagged = AutoTokenizer.from_pretrained(name, add_prefix_space=True)
    assert split_mismatches(default, [SENTENCE]) == []
    ids = default(SENTENCE, is_split_into_words=True)["input_ids"]
    assert flagged(SENTENCE, is_split_into_words=True)["input_ids"] == ids


@pytest.mark.network
@pytest.mark.filterwarnings("ignore:Deprecated in 0.9.0:DeprecationWarning")  # BPE from files
def test_kazroberta_needs_add_prefix_space():
    from transformers import AutoTokenizer

    from kazner.align import split_mismatches

    name = "kz-transformers/kaz-roberta-conversational"
    default = AutoTokenizer.from_pretrained(name)
    assert split_mismatches(default, [SENTENCE]) == [(0, j) for j in range(1, len(SENTENCE))]
    assert (
        split_mismatches(AutoTokenizer.from_pretrained(name, add_prefix_space=True), [SENTENCE])
        == []
    )
