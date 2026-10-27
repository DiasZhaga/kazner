import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import kazner.fragmentation as frag
from kazner.cli import main
from kazner.config import data_fingerprint
from kazner.contract import Predictions, write_predictions
from kazner.data import read_splits

TINY = Path(__file__).parent / "data" / "kaznerd_tiny"


def predictions(words, gold, pred=None):
    return Predictions(list(range(len(words))), words, gold, pred or gold)


# --- pieces and tables -------------------------------------------------------------------


def test_piece_counter_counts_pieces_and_unknowns(tiny_tokenizer):
    count = frag.piece_counter(tiny_tokenizer)
    assert count("Алматыға") == (3, False)  # Алма ##ты ##ға
    assert count("Асан") == (1, False)
    assert count("Париж") == (1, True)  # not in the vocabulary
    assert count("") == (1, True)  # empty tokenization: one unknown piece, as in fertility


def test_word_table_counts_training_frequency(tiny_tokenizer):
    test = predictions(
        [["Асан", "Бекұлы", "Алматыға", "барды", "."]],
        [["B-PERSON", "I-PERSON", "B-LOCATION", "O", "O"]],
    )
    words = frag.word_table(
        test, [["Асан", "барды", "барды", "."]], frag.piece_counter(tiny_tokenizer)
    )
    assert words["pieces"].tolist() == [1, 2, 3, 1, 1]
    assert words["freq"].tolist() == [1, 0, 0, 2, 1]
    assert words["freq_bin"].tolist() == ["1-4", "unseen", "unseen", "1-4", "1-4"]
    assert words["type"].tolist() == ["PERSON", "PERSON", "LOCATION", "", ""]
    assert words["is_begin"].tolist() == [True, False, True, False, False]
    assert words["is_entity"].tolist() == [True, True, True, False, False]


def test_entity_table_marks_seen_entities_and_unknown_words(tiny_tokenizer):
    train = [["Асан", "Бекұлы", "барды"]], [["B-PERSON", "I-PERSON", "O"]]
    test = predictions(
        [["Асан", "Бекұлы", "Париж", "барды"], ["Асан"]],
        [["B-PERSON", "I-PERSON", "B-LOCATION", "O"], ["B-LOCATION"]],
    )
    entities = frag.entity_table(test, *train, frag.piece_counter(tiny_tokenizer))
    assert entities[["sentence_id", "start", "end", "type"]].values.tolist() == [
        [0, 0, 2, "PERSON"],
        [0, 2, 3, "LOCATION"],
        [1, 0, 1, "LOCATION"],
    ]
    assert entities["mean_pieces"].tolist() == [1.5, 1.0, 1.0]
    assert entities["fertility_bin"].tolist() == ["(1,2]", "1.0", "1.0"]
    assert entities["seen"].tolist() == [True, False, False]  # same text, other type: unseen
    assert entities["has_unk"].tolist() == [False, True, False]


def test_entity_hits_need_the_exact_span_and_type(tiny_tokenizer):
    gold = [["B-PERSON", "I-PERSON", "B-LOCATION", "O"]]
    words = [["Асан", "Бекұлы", "Алматыға", "барды"]]
    entities = frag.entity_table(
        predictions(words, gold), [], [], frag.piece_counter(tiny_tokenizer)
    )
    cases = {
        ("B-PERSON", "I-PERSON", "B-ORGANISATION", "O"): [True, False],  # wrong type
        ("B-PERSON", "O", "B-LOCATION", "O"): [False, True],  # span too short
        ("I-PERSON", "I-PERSON", "I-LOCATION", "O"): [True, True],  # conlleval-style starts
        ("B-PERSON", "I-PERSON", "I-PERSON", "O"): [False, False],  # span too long
    }
    for pred, expected in cases.items():
        hits = frag.entity_hits(entities, predictions(words, gold, [list(pred)]))
        assert hits.tolist() == expected, pred


def test_bins():
    assert [frag.piece_bin(n) for n in (1, 4, 5, 9)] == ["1", "4", "5+", "5+"]
    assert [frag.frequency_bin(f) for f in (0, 1, 4, 5, 10**6)] == [
        "unseen",
        "1-4",
        "1-4",
        "5+",
        "5+",
    ]
    assert [frag.fertility_bin(m) for m in (1.0, 1.5, 2.0, 2.5, 3.0, 3.01)] == [
        "1.0", "(1,2]", "(1,2]", "(2,3]", "(2,3]", ">3",
    ]  # fmt: skip


def test_error_and_recall_tables_by_hand(tiny_tokenizer, monkeypatch):
    monkeypatch.setattr(frag, "MIN_CELL", 1)
    test = predictions(
        [["Асан", "Бекұлы", "Алматыға", "барды"]], [["B-PERSON", "I-PERSON", "B-LOCATION", "O"]]
    )
    pieces = frag.piece_counter(tiny_tokenizer)
    words = frag.word_table(test, [], pieces)  # entity words: 1, 2, 3 pieces
    errors = np.array([[1, 0, 1, 1], [0, 0, 1, 0]], dtype=bool)  # runs x words
    table = {r["pieces"]: r for r in frag.error_by_pieces(words, errors)}
    assert sorted(table) == ["1", "2", "3"]  # the O word is not an entity word
    assert table["1"]["error_mean"] == 0.5
    assert table["1"]["error_sd"] == pytest.approx(math.sqrt(0.5))
    assert table["3"] == {
        "pieces": "3",
        "n": 1,
        "share_unseen": 1.0,
        "error_mean": 1.0,
        "error_sd": 0.0,
    }
    strata = frag.error_by_pieces_within_frequency(words, errors)
    assert {r["train_freq"] for r in strata} == {"unseen"}

    entities = frag.entity_table(test, [], [], pieces)
    hits = np.array([[True, False], [True, True]])
    recall = frag.recall_by_fertility(entities, hits)
    assert [(r["pieces_per_word"], r["recall_mean"]) for r in recall] == [
        ("(1,2]", 1.0),  # Асан Бекұлы: (1 + 2) / 2 pieces per word
        ("(2,3]", 0.5),  # Алматыға: 3 pieces
    ]
    assert all(r["seen_in_train"] is False for r in recall)


# --- statistics --------------------------------------------------------------------------


def test_logistic_fit_equals_the_cross_product_odds_ratio():
    # Intercept + one binary predictor: the maximum-likelihood odds ratio is (a d) / (b c).
    a, b, c, d = 40, 60, 25, 75  # errors / correct with z = 1, then with z = 0
    z = np.r_[np.ones(a + b), np.zeros(c + d)]
    y = np.r_[np.ones(a), np.zeros(b), np.ones(c), np.zeros(d)]
    beta = frag.fit_logistic(np.column_stack([np.ones_like(z), z]), y)
    assert math.exp(beta[1]) == pytest.approx((a * d) / (b * c), rel=1e-6)


def test_logistic_fit_stays_finite_under_perfect_separation():
    z = np.r_[np.ones(20), np.zeros(20)]
    y = np.r_[np.zeros(20), np.r_[np.ones(10), np.zeros(10)]]  # z = 1 never errs
    beta = frag.fit_logistic(np.column_stack([np.ones_like(z), z]), y)
    assert np.isfinite(beta).all() and beta[1] < -5


def test_bootstrap_is_reproducible():
    rng = np.random.default_rng(1)
    n = 400
    pieces = rng.integers(1, 6, n).astype(float)
    x = np.column_stack([np.ones(n), pieces])
    errors = (rng.random((2, n)) < 1 / (1 + np.exp(2.5 - 0.4 * pieces))).astype(float)
    sentence_ids = np.repeat(np.arange(n // 4), 4)
    names = ["intercept", "pieces"]
    first = frag.bootstrap_piece_or(x, errors, sentence_ids, names, replicates=50, seed=7)
    assert first == frag.bootstrap_piece_or(x, errors, sentence_ids, names, replicates=50, seed=7)
    assert first[0] < math.exp(0.4) < first[1]


# --- runs and the command ----------------------------------------------------------------


def make_run(root, run_id, *, seed, method, pred=None, **data):
    test_words, test_gold = read_splits(TINY)["test"]
    config = {
        "run_id": run_id,
        "data_fingerprint": data_fingerprint(TINY),
        "seed": seed,
        "data": {"train_fraction": 1.0, "subsample_seed": 42, "max_train_samples": None, **data},
        "model": {"name": "tiny"},
        "method": {"name": method},
    }
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    p = Predictions(list(range(len(test_words))), test_words, test_gold, pred or test_gold)
    write_predictions(run_dir / "predictions.csv", p)
    return run_dir


def test_cli_scores_every_group_on_the_same_entities(tiny_tokenizer, tmp_path, monkeypatch, capsys):
    import transformers

    seen_kwargs = []

    def fake(name, **kwargs):
        seen_kwargs.append(kwargs)
        return tiny_tokenizer

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake)
    wrong = [["B-PERSON", "I-PERSON", "O", "O", "O", "O"], ["B-DATE", "O", "O", "O", "O"]]
    runs = [
        make_run(tmp_path / "runs", "a1", seed=1, method="full"),
        make_run(tmp_path / "runs", "a2", seed=2, method="full", pred=wrong),
        make_run(tmp_path / "runs", "b1", seed=1, method="lora", pred=wrong),
    ]
    out = tmp_path / "out"
    args = ["fragmentation", "--runs", *map(str, runs), "--data", str(TINY), "--tokenizer", "tiny"]
    assert (
        main([*args, "--by", "method", "--bootstrap", "3", "--revision", "abc", "--out", str(out)])
        == 0
    )
    assert seen_kwargs == [{"revision": "abc"}]

    report = json.loads((out / "fragmentation.json").read_text(encoding="utf-8"))
    assert report["reference_tokenizer"] == {
        "name": "tiny",
        "revision": "abc",
        "add_prefix_space": False,
    }
    assert report["data"]["train_sentences"] == 4
    assert (report["n_test_words"], report["n_entity_words"], report["n_gold_entities"]) == (
        11,
        4,
        3,
    )
    assert set(report["groups"]) == {"method=full", "method=lora"}
    full = report["groups"]["method=full"]
    assert full["run_ids"] == ["a1", "a2"] and full["seeds"] == [1, 2]
    regression = full["logistic_regression"]
    assert regression["per_run"][0]["OR_per_piece"] is None  # a1 makes no error: undefined
    assert regression["per_run"][1]["OR_per_piece"] > 0
    assert regression["OR_per_piece_geomean"] is None and regression["OR_per_piece_CI95"] is None
    lora = report["groups"]["method=lora"]["logistic_regression"]
    assert lora["OR_per_piece_CI95"][0] <= lora["OR_per_piece_geomean"]

    entities = pd.read_csv(out / "entities.csv")
    assert entities["text"].tolist() == ["Марат Сейітов", "Ертең", "Шымкентте"]
    assert entities["mean_pieces"].tolist() == [1.0, 1.0, 2.0]  # Шымкент ##те
    assert entities["hit_a1"].tolist() == [True, True, True]
    assert entities["hit_a2"].tolist() == entities["hit_b1"].tolist() == [True, True, False]
    assert "method=lora" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"subsample_seed": 7}, "different training subsamples"),
        ({"fingerprint": "0" * 64}, "trained on other data"),
        ({"words": True}, "another test set"),
    ],
)
def test_runs_must_share_data_subsample_and_test_set(tiny_tokenizer, tmp_path, change, message):
    first = frag.load_run(make_run(tmp_path, "a", seed=1, method="full"))
    second = frag.load_run(make_run(tmp_path, "b", seed=2, method="full"))
    if "subsample_seed" in change:
        second.config["data"]["subsample_seed"] = change["subsample_seed"]
    if "fingerprint" in change:
        second.config["data_fingerprint"] = change["fingerprint"]
    if "words" in change:
        second.predictions.words[0][0] = "Асан"
    with pytest.raises(ValueError, match=message):
        frag.analyse([first, second], TINY, tiny_tokenizer, bootstrap=1)
