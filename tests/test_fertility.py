import json
from pathlib import Path

import pytest

from kazner.cli import main
from kazner.data import read_splits
from kazner.fertility import FertilityStats, WordTokenizer, analyse_corpus, word_stats
from reference import pilot_fertility

TINY = Path(__file__).parent / "data" / "kaznerd_tiny"


def test_hand_computed_statistics(tiny_tokenizer):
    # Алматыға -> 3 pieces, барды -> 1, Ақтөбе -> [UNK], Бекұлы -> 2
    words = ["Алматыға", "барды", "Ақтөбе", "Бекұлы"]
    s = word_stats(words, WordTokenizer(tiny_tokenizer)).summary()
    assert (s["words"], s["subword_tokens"]) == (4, 7)
    assert s["fertility"] == pytest.approx(7 / 4)
    assert (s["single_token_words"], s["split_words"]) == (2, 2)
    assert s["single_token_rate"] == s["split_word_rate"] == 0.5
    assert s["mean_subwords_of_split_words"] == pytest.approx(2.5)
    assert s["median_subwords_per_word"] == 1.5
    assert s["std_subwords_per_word"] == pytest.approx((3.75 - 1.75**2) ** 0.5)
    assert s["max_subwords_per_word"] == 3
    assert (s["unknown_words"], s["unk_rate"]) == (1, 0.25)


def test_empty_tokenization_counts_as_one_unknown_token(tiny_tokenizer):
    assert tiny_tokenizer.tokenize("​") == []  # zero-width space is stripped
    assert WordTokenizer(tiny_tokenizer)("​") == (1, True)


def test_lowercase_option(tiny_tokenizer):
    assert WordTokenizer(tiny_tokenizer)("Асан") == (1, False)
    assert WordTokenizer(tiny_tokenizer, lowercase=True)("Асан") == (1, True)  # not in vocab


def test_word_tokenizer_caches(tiny_tokenizer):
    calls = []

    class Spy:
        unk_token = "[UNK]"

        def tokenize(self, word):
            calls.append(word)
            return tiny_tokenizer.tokenize(word)

    wt = WordTokenizer(Spy())
    assert wt("Алматыға") == wt("Алматыға") == (3, False)
    assert calls == ["Алматыға"]


def test_no_split_words_and_empty_stats():
    stats = FertilityStats()
    stats.add(1, False)
    assert stats.summary()["mean_subwords_of_split_words"] == 0.0
    with pytest.raises(ValueError, match="no words"):
        FertilityStats().summary()


def test_corpus_report_overall_is_the_sum_of_splits(tiny_tokenizer):
    report = analyse_corpus(TINY, tiny_tokenizer, tokenizer_name="tiny")
    splits = report["splits"]
    assert set(splits) == {"train", "validation", "test", "overall"}
    for key in ("words", "subword_tokens", "single_token_words", "split_words", "unknown_words"):
        assert splits["overall"][key] == sum(
            splits[s][key] for s in ("train", "validation", "test")
        )
    assert report["tokenizer"] == "tiny"
    assert splits["train"]["words"] == 20


def test_identical_to_pilot(tiny_tokenizer, monkeypatch):
    """Same per-split numbers as the pilot's RunningStats / TokenizerAnalyzer."""
    monkeypatch.setattr(
        pilot_fertility.TokenizerAnalyzer, "_load_tokenizer", lambda self, name: tiny_tokenizer
    )
    analyzer = pilot_fertility.TokenizerAnalyzer("tiny", lowercase=False)
    ours = analyse_corpus(TINY, tiny_tokenizer)["splits"]
    for split, (tokens, _) in read_splits(TINY).items():
        stats = pilot_fertility.RunningStats()
        for word in (w for sentence in tokens for w in sentence):
            count, _pieces, unknown = analyzer.analyze_word(word)
            stats.update(count, unknown)
        pilot = stats.to_result_row(split, "tiny")
        mine = ours[split]
        assert mine["words"] == pilot["total_words"]
        assert mine["subword_tokens"] == pilot["total_subword_tokens"]
        assert mine["fertility"] == pytest.approx(pilot["fertility"])
        assert mine["split_word_rate"] == pytest.approx(pilot["split_word_rate"])
        assert mine["mean_subwords_of_split_words"] == pytest.approx(
            pilot["mean_subwords_for_split_words"]
        )
        assert mine["median_subwords_per_word"] == pilot["median_subwords_per_word"]
        assert mine["std_subwords_per_word"] == pytest.approx(pilot["std_subwords_per_word"])
        assert mine["max_subwords_per_word"] == pilot["maximum_subwords_per_word"]
        assert mine["unk_rate"] == pytest.approx(pilot["unknown_token_rate"])


def test_cli_fertility(tiny_tokenizer, tmp_path, monkeypatch, capsys):
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda name: tiny_tokenizer)
    out = tmp_path / "fertility.json"
    assert main(["fertility", "--data", str(TINY), "--tokenizer", "tiny", "--out", str(out)]) == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved == json.loads(capsys.readouterr().out)
    assert saved["tokenizer"] == "tiny"


@pytest.mark.network
def test_mbert_fertility_on_fixtures():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    overall = analyse_corpus(TINY, tokenizer)["splits"]["overall"]
    assert overall["words"] == 39  # 20 + 8 + 11
    assert overall["fertility"] > 1.5  # mBERT fragments Kazakh words
    assert overall["split_words"] > overall["words"] / 2
