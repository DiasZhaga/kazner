"""Paired arm comparison (#37) on hand-made entity tables."""

import json

import pandas as pd
import pytest

from kazner.cli import main
from kazner.compare import Pair, compare, holm, subset_masks

# sentence, seen, pieces per word, [UNK], hits of runs a1 a2 (arm A) and b1 (arm B)
ROWS = [
    (0, False, 4.0, False, 1, 0, 1),  # primary
    (0, False, 3.5, False, 0, 0, 1),  # primary
    (1, False, 5.0, False, 1, 1, 1),  # primary
    (1, False, 4.0, False, 0, 1, 0),  # primary
    (2, False, 1.0, False, 1, 1, 1),  # specificity
    (2, False, 2.0, False, 1, 0, 1),  # specificity
    (3, True, 1.0, False, 1, 1, 1),  # negative control
    (3, True, 4.0, False, 1, 1, 0),  # negative control
    (4, False, 4.0, True, 0, 1, 1),  # unseen with [UNK]
    (4, False, 2.5, False, 1, 1, 1),  # in no subset (unseen, between 2 and 3 pieces)
]
COLUMNS = ["sentence_id", "seen", "mean_pieces", "has_unk", "hit_a1", "hit_a2", "hit_b1"]


@pytest.fixture
def entities():
    return pd.DataFrame(ROWS, columns=COLUMNS)


def test_subsets_follow_the_preregistered_definitions(entities):
    sizes = {name: int(mask.sum()) for name, mask in subset_masks(entities).items()}
    assert sizes == {"primary": 4, "specificity": 2, "negative_control": 2, "unseen_with_unk": 1}


def test_point_estimates_by_hand(entities):
    report = compare(entities, [Pair("full", ("a1", "a2"), ("b1",))], replicates=200)
    s = report["pairs"]["full"]["subsets"]
    assert s["primary"]["recall_a_runs"] == {"a1": 0.5, "a2": 0.5}
    assert (s["primary"]["recall_a"], s["primary"]["recall_b"]) == (0.5, 0.75)
    assert s["primary"]["delta"] == pytest.approx(0.25)
    assert s["specificity"]["delta"] == pytest.approx(0.25)  # A (1.0 + 0.5) / 2, B 1.0
    assert s["negative_control"]["delta"] == pytest.approx(-0.5)
    assert s["unseen_with_unk"]["delta"] == pytest.approx(0.5)
    assert report["pairs"]["full"]["did_primary_minus_specificity"]["delta"] == pytest.approx(0)
    assert report["bootstrap"] == {
        "unit": "test sentence",
        "replicates": 200,
        "seed": 0,
        "sentences": 5,
    }


def test_bootstrap_is_reproducible_and_brackets_the_estimate(entities):
    pairs = [Pair("full", ("a1", "a2"), ("b1",))]
    first = compare(entities, pairs, replicates=400, seed=3)
    assert first == compare(entities, pairs, replicates=400, seed=3)
    assert first != compare(entities, pairs, replicates=400, seed=4)
    primary = first["pairs"]["full"]["subsets"]["primary"]
    assert primary["ci95"][0] <= primary["delta"] <= primary["ci95"][1]
    assert 0 < primary["p_two_sided"] <= 1


def test_identical_arms_give_no_difference(entities):
    entities["hit_c1"], entities["hit_c2"] = entities["hit_a1"], entities["hit_a2"]
    report = compare(entities, [Pair("same", ("a1", "a2"), ("c1", "c2"))], replicates=300)
    pair = report["pairs"]["same"]
    for s in pair["subsets"].values():
        assert s["delta"] == 0 and s["ci95"] == [0.0, 0.0] and s["p_two_sided"] == 1.0
    assert pair["h1_supported"] is False and pair["h2_supported"] is False


def test_a_clear_effect_is_detected_and_holm_corrected():
    rows = []
    for sid in range(200):  # arm B finds every primary entity, arm A none
        rows.append((sid, False, 4.0, False, 0, 1))
        rows.append((sid, False, 1.0, False, 1, 1))
        rows.append((sid, True, 1.0, False, 1, 1))
    rows.append((200, False, 4.0, True, 1, 1))
    frame = pd.DataFrame(
        rows, columns=["sentence_id", "seen", "mean_pieces", "has_unk", "hit_a", "hit_b"]
    )
    frame["hit_c"], frame["hit_d"] = frame["hit_a"], frame["hit_b"]
    pairs = [Pair("full", ("a",), ("b",)), Pair("lora", ("c",), ("d",))]
    report = compare(frame, pairs, replicates=999)
    for name in ("full", "lora"):
        pair = report["pairs"][name]
        primary = pair["subsets"]["primary"]
        assert primary["delta"] == 1.0 and primary["p_two_sided"] == pytest.approx(2 / 1000)
        assert primary["p_holm"] == pytest.approx(2 * 2 / 1000)
        assert pair["h1_supported"] is True and pair["h2_supported"] is True  # DiD = 1 - 0


def test_holm_by_hand():
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})  # monotone step-down
    assert holm({"a": 0.6, "b": 0.7}) == {"a": 1.0, "b": 1.0}  # capped at 1


@pytest.mark.parametrize(
    ("pairs", "message"),
    [
        ([], "no pairs"),
        ([Pair("x", ("a1",), ("zz",))], r"no hit column for runs \['zz'\]"),
        ([Pair("x", ("a1",), ("a1", "b1"))], "cannot be in both arms"),
        ([Pair("x", (), ("b1",))], "at least one run"),
        ([Pair("x", ("a1",), ("b1",)), Pair("x", ("a2",), ("b1",))], "must be unique"),
    ],
)
def test_invalid_pairs_are_rejected(entities, pairs, message):
    with pytest.raises(ValueError, match=message):
        compare(entities, pairs, replicates=10)


def test_an_empty_subset_is_rejected(entities):
    with pytest.raises(ValueError, match="subset 'unseen_with_unk' is empty"):
        compare(entities[~entities["has_unk"]], [Pair("x", ("a1",), ("b1",))], replicates=10)


def test_cli_compare(entities, tmp_path, capsys):
    entities["hit_c1"] = entities["hit_b1"]
    path = tmp_path / "entities.csv"
    entities.to_csv(path, index=False)
    out = tmp_path / "comparison.json"
    args = ["compare", "--entities", str(path), "--bootstrap", "100", "--out", str(out)]
    assert main([*args, "--pair", "full", "a1,a2", "b1", "--pair", "lora", "a2", "c1"]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert set(report["pairs"]) == {"full", "lora"} and report["pairs"]["lora"]["a"] == ["a2"]
    assert "p_holm" in report["pairs"]["full"]["subsets"]["primary"]
    printed = capsys.readouterr().out
    assert "=== full" in printed and "H1 supported" in printed
