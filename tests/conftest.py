import json
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


@pytest.fixture(scope="session")
def bytelevel_dir(tmp_path_factory):
    """Offline byte-level BPE tokenizer in the layout of a RoBERTa checkpoint without
    ``tokenizer_config.json`` (``config.json``, ``vocab.json``, ``merges.txt``), as
    ``kz-transformers/kaz-roberta-conversational`` ships it. Trained on the synthetic training
    sentences as running text, so words after the first exist with the leading-space marker."""
    from tokenizers import ByteLevelBPETokenizer

    from kazner.data import read_iob2

    sentences = [
        " ".join(words) for words in read_iob2(DATA / "kaznerd_tiny" / "IOB2_train.txt")[0]
    ]
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        sentences,
        vocab_size=400,
        min_frequency=1,
        show_progress=False,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    path = tmp_path_factory.mktemp("bytelevel")
    tokenizer.save_model(str(path))
    (path / "config.json").write_text(json.dumps({"model_type": "roberta"}), encoding="utf-8")
    return path
