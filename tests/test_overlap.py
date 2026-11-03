import json
from pathlib import Path

import pytest

from kazner.cli import main
from kazner.data import read_iob2
from kazner.overlap import format_overlap, piece_overlap

TINY = Path(__file__).parent / "data" / "kaznerd_tiny"


class PairTokenizer:
    """Pieces of two characters; a word with '?' is unknown."""

    unk_token = "[UNK]"

    def tokenize(self, word):
        if "?" in word:
            return [self.unk_token]
        return [word[i : i + 2] for i in range(0, len(word), 2)]


def test_overlap_by_hand():
    source = [["abcd", "ef"]]  # pieces ab, cd, ef
    target = [["abef", "gh", "ef"], ["ab?x"]]  # pieces ab ef | gh | ef || [UNK]
    labels = [["B-X", "O", "O"], ["B-Y"]]
    report = piece_overlap(PairTokenizer(), source, target, labels)

    assert report["source"] == {"sentences": 1, "words": 2, "word_types": 2, "piece_types": 3}
    assert report["all_words"] == {
        "words": 4,
        "word_types": 4,
        "piece_tokens": 5,
        "piece_types": 4,  # ab, ef, gh, [UNK]
        "piece_types_shared": 2 / 4,  # ab, ef; the unknown token is never shared
        "piece_tokens_covered": 3 / 5,
        "words_fully_covered": 2 / 4,  # abef, ef
        "word_types_in_source": 1 / 4,  # ef
    }
    assert report["entity_words"] == {
        "words": 2,
        "word_types": 2,
        "piece_tokens": 3,
        "piece_types": 3,
        "piece_types_shared": 2 / 3,
        "piece_tokens_covered": 2 / 3,
        "words_fully_covered": 1 / 2,
        "word_types_in_source": 0.0,
    }
    assert "entity_words" not in piece_overlap(PairTokenizer(), source, target)
    text = format_overlap(report)
    assert "shared piece types 50.0%" in text and "target entity words: 2 words" in text


def test_identical_corpora_overlap_completely():
    words = [["abcd", "ef"], ["ab"]]
    r = piece_overlap(PairTokenizer(), words, words)["all_words"]
    assert r["piece_types_shared"] == r["piece_tokens_covered"] == 1.0
    assert r["words_fully_covered"] == r["word_types_in_source"] == 1.0


def test_no_entities_gives_empty_shares():
    r = piece_overlap(PairTokenizer(), [["ab"]], [["ab"]], [["O"]])["entity_words"]
    assert r["words"] == 0 and r["piece_types_shared"] is None


def test_overlap_cli_with_a_real_tokenizer(tiny_tokenizer, tmp_path, capsys):
    tiny_tokenizer.save_pretrained(tmp_path / "tok")
    out = tmp_path / "overlap.json"
    args = ["overlap", "--source", str(TINY / "IOB2_train.txt")]
    args += ["--target", str(TINY / "IOB2_test.txt"), "--tokenizer", str(tmp_path / "tok")]
    assert main([*args, "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    words, labels = read_iob2(TINY / "IOB2_test.txt")
    assert report["all_words"]["words"] == sum(map(len, words))
    entity_words = sum(t != "O" for s in labels for t in s)
    assert report["entity_words"]["words"] == entity_words
    for key in ("piece_types_shared", "piece_tokens_covered", "words_fully_covered"):
        assert 0 <= report["all_words"][key] <= 1
    assert report["tokenizer"]["name"] == str(tmp_path / "tok")
    assert "source: 4 sentences" in capsys.readouterr().out


@pytest.mark.parametrize("pieces", [[], ["a"]])
def test_a_word_without_pieces_counts_as_unknown(pieces):
    class Empty(PairTokenizer):
        def tokenize(self, word):
            return list(pieces) if word == "x" else super().tokenize(word)

    r = piece_overlap(Empty(), [["ab"]], [["x"]])["all_words"]
    assert r["piece_types"] == 1
    assert r["words_fully_covered"] == 0.0
