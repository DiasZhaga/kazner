from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def tiny_tokenizer():
    """Offline WordPiece tokenizer built from ``tests/data/tiny_wordpiece/vocab.txt``."""
    from transformers import BertTokenizer

    lines = (DATA / "tiny_wordpiece" / "vocab.txt").read_text(encoding="utf-8").splitlines()
    # transformers 5 takes the vocabulary as ``vocab`` (``vocab_file`` is silently ignored);
    # a dict avoids the deprecated from-file path of ``tokenizers.WordPiece``.
    return BertTokenizer(vocab={token: i for i, token in enumerate(lines)}, do_lower_case=False)
