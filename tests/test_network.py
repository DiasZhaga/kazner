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
