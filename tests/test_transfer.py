import csv
import json
from pathlib import Path

import pytest

from kazner.cli import main
from kazner.config import ConfigError
from kazner.contract import Predictions, write_predictions
from kazner.evaluate import compute_metrics
from kazner.transfer import (
    analyse,
    bootstrap_mean_f1,
    entity_counts,
    load_mapping,
    load_transfer_run,
    parse_mapping,
    recall_by_pieces,
    score_run,
)

ROOT = Path(__file__).resolve().parent.parent
#: KazNERD's 25 entity types (Yeshpanov et al., 2022).
KAZNERD_TYPES = {
    "ADAGE", "ART", "CARDINAL", "CONTACT", "DATE", "DISEASE", "EVENT", "FACILITY", "GPE",
    "LANGUAGE", "LAW", "LOCATION", "MISCELLANEOUS", "MONEY", "NON_HUMAN", "NORP", "ORDINAL",
    "ORGANISATION", "PERCENTAGE", "PERSON", "POSITION", "PRODUCT", "PROJECT", "QUANTITY", "TIME",
}  # fmt: skip
RAW = {
    "source": "KazNERD",
    "target": "WikiANN",
    "target_types": ["PER", "LOC", "ORG"],
    "mapping": {
        "PERSON": "PER",
        "GPE": "LOC",
        "LOCATION": "LOC",
        "ORGANISATION": "ORG",
        "DATE": "O",
    },
}
MAPPING = parse_mapping(RAW)
#: Hand-made KazNERD test: gold PERSON, GPE, DATE; predicted PERSON and LOCATION.
SOURCE = Predictions(
    sentence_ids=[0],
    words=[["Асан", "кеше", "Астанада", "болды"]],
    gold=[["B-PERSON", "O", "B-GPE", "B-DATE"]],
    pred=[["B-PERSON", "O", "B-LOCATION", "O"]],
)
#: Hand-made extra set: gold PER (2 words) and ORG; predicted PERSON and GPE.
EXTRA = Predictions(
    sentence_ids=[0, 1],
    words=[["Айгүл", "Сейітова", "университет"], ["Марат", "тұрады"]],
    gold=[["B-PER", "I-PER", "B-ORG"], ["B-PER", "O"]],
    pred=[["B-PERSON", "I-PERSON", "B-GPE"], ["O", "O"]],
)


def make_run(path: Path, seed: int = 42, model: str = "m", source=SOURCE, extra=EXTRA) -> Path:
    path.mkdir(parents=True)
    config = {"name": "t", "seed": seed, "model": {"name": model}, "method": {"name": "full"}}
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    write_predictions(path / "predictions.csv", source)
    write_predictions(path / "predictions_wiki.csv", extra)
    return path


# --- mapping ---------------------------------------------------------------------------


def test_mapping_keeps_prefixes_and_sends_other_types_to_o():
    assert MAPPING.map_tags(["B-GPE", "I-GPE", "O"]) == ["B-LOC", "I-LOC", "O"]
    assert MAPPING.map_tags(["B-DATE", "I-DATE"]) == ["O", "O"]
    adjacent = MAPPING.map_tags(["B-GPE", "B-LOCATION"])
    assert adjacent == ["B-LOC", "B-LOC"]  # two entities, not one
    assert compute_metrics([adjacent], [adjacent])["n_gold_entities"] == 2


def test_unlisted_types_are_errors():
    with pytest.raises(ConfigError, match="'NORP' is not listed"):
        MAPPING.map_tags(["B-NORP"])
    with pytest.raises(ConfigError, match=r"contains types it does not list: \['NORP'\]"):
        MAPPING.check_covers({"PERSON", "NORP"}, "the predictions")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"mapping": {"PERSON": "PERSON"}}, "unknown target 'PERSON'"),
        ({"target_types": ["PER", "O"]}, "target_types"),
        ({"target_types": ["PER", "PER"]}, "target_types"),
        ({"mapping": {"B-PERSON": "PER"}}, "invalid source type"),
        ({"labels": []}, "unknown key"),
        ({"mapping": {}}, "non-empty mapping"),
    ],
)
def test_invalid_mappings_are_rejected(change, message):
    with pytest.raises(ConfigError, match=message):
        parse_mapping({**RAW, **change})
    without_target = {k: v for k, v in RAW.items() if k != "target"}
    with pytest.raises(ConfigError, match="missing key"):
        parse_mapping(without_target)


def test_shipped_mappings_list_every_kaznerd_type():
    main_map = load_mapping(ROOT / "configs" / "mappings" / "kaznerd_to_wikiann.yaml")
    variant = load_mapping(ROOT / "configs" / "mappings" / "kaznerd_to_wikiann_facility_loc.yaml")
    for mapping in (main_map, variant):
        assert set(mapping.types) == KAZNERD_TYPES
        assert mapping.target_types == ("PER", "LOC", "ORG")
    kept = {t: m for t, m in main_map.types.items() if m != "O"}
    assert kept == {"PERSON": "PER", "GPE": "LOC", "LOCATION": "LOC", "ORGANISATION": "ORG"}
    assert (main_map.types["FACILITY"], variant.types["FACILITY"]) == ("O", "LOC")
    assert {**variant.types, "FACILITY": "O"} == dict(main_map.types)


# --- scores ----------------------------------------------------------------------------


def test_three_scores_of_a_hand_made_run(tmp_path):
    run = load_transfer_run(make_run(tmp_path / "r1"), "wiki")
    scores = score_run(run, MAPPING, "wiki")
    # 25 types: gold PERSON, GPE, DATE; predicted PERSON (correct) and LOCATION (wrong type).
    assert scores["kaznerd"]["precision"] == pytest.approx(1 / 2)
    assert scores["kaznerd"]["recall"] == pytest.approx(1 / 3)
    # Mapped: GPE and LOCATION are both LOC, DATE disappears: both entities are correct.
    assert scores["kaznerd_mapped"]["f1"] == pytest.approx(1.0)
    # Extra set: PER is found with both words, ORG is predicted as GPE -> LOC; one PER missed.
    w = scores["wiki"]
    assert (w["precision"], w["recall"]) == (pytest.approx(1 / 2), pytest.approx(1 / 3))
    assert set(w["per_type"]) == {"PER", "LOC", "ORG"}


def test_entity_counts_reproduce_seqeval_micro_f1():
    gold, pred = EXTRA.gold, [MAPPING.map_tags(p) for p in EXTRA.pred]
    tp, n_pred, n_gold = entity_counts(gold, pred).sum(axis=0)
    assert 2 * tp / (n_pred + n_gold) == pytest.approx(compute_metrics(gold, pred)["f1"])


def test_bootstrap_ci_is_deterministic_and_contains_the_estimate():
    counts = entity_counts(EXTRA.gold, [MAPPING.map_tags(p) for p in EXTRA.pred])
    low, high = bootstrap_mean_f1([counts, counts], replicates=600, seed=3)
    assert low <= 0.4 <= high  # F1 = 2·1 / (2 + 3)
    assert bootstrap_mean_f1([counts, counts], replicates=600, seed=3) == [low, high]


def test_recall_by_pieces_bins_the_extra_entities():
    run = type("R", (), {"extra": EXTRA})()
    rows = recall_by_pieces([run], MAPPING, lambda word: (len(word) // 3, False))
    by_bin = {r["bin"]: r for r in rows}
    # PER "Айгүл Сейітова": (1 + 2) / 2 = 1.5 pieces -> (1,2], found; ORG "университет": 3 -> (2,3],
    # missed; PER "Марат": 1 -> 1.0, missed.
    assert (by_bin["(1,2]"]["n"], by_bin["(1,2]"]["recall"]) == (1, 1.0)
    assert (by_bin["(2,3]"]["n"], by_bin["(2,3]"]["recall"]) == (1, 0.0)
    assert (by_bin["1.0"]["n"], by_bin["1.0"]["recall"]) == (1, 0.0)
    assert by_bin[">3"] == {"bin": ">3", "n": 0, "recall": None, "sd": None}


def test_analyse_groups_runs_and_averages_seeds(tmp_path):
    runs = [
        load_transfer_run(make_run(tmp_path / f"r{seed}{model}", seed, model), "wiki")
        for model in ("a", "b")
        for seed in (43, 42)
    ]
    report = analyse(runs, MAPPING, "wiki", bootstrap=200)
    assert [g["group"] for g in report["groups"]] == [{"model": "a"}, {"model": "b"}]
    first = report["groups"][0]
    assert first["seeds"] == [42, 43]
    assert first["steps"]["kaznerd_mapped"]["f1"] == pytest.approx(1.0)
    assert first["steps"]["kaznerd_mapped"]["f1_sd"] == pytest.approx(0.0)
    assert len(first["steps"]["wiki"]["f1_ci95"]) == 2
    assert "f1_ci95" not in first["steps"]["kaznerd"]
    assert report["steps"] == ["kaznerd", "kaznerd_mapped", "wiki"]


def test_analyse_refuses_runs_on_different_test_sets(tmp_path):
    other = Predictions([0, 1], [["x", "y", "z"], ["u", "v"]], EXTRA.gold, EXTRA.pred)
    runs = [
        load_transfer_run(make_run(tmp_path / "a"), "wiki"),
        load_transfer_run(make_run(tmp_path / "b", 43, extra=other), "wiki"),
    ]
    with pytest.raises(ValueError, match="different wiki sets"):
        analyse(runs, MAPPING, "wiki")


def test_foreign_gold_types_and_missing_files_are_reported(tmp_path):
    misc = Predictions([0], [["a"]], [["B-MISC"]], [["O"]])
    run = load_transfer_run(make_run(tmp_path / "a", extra=misc), "wiki")
    with pytest.raises(ConfigError, match=r"outside target_types"):
        score_run(run, MAPPING, "wiki")
    with pytest.raises(FileNotFoundError, match="extra_tests 'other'"):
        load_transfer_run(tmp_path / "a", "other")


def test_cli_writes_json_and_csv(tmp_path, capsys):
    runs = [str(make_run(tmp_path / f"r{seed}", seed)) for seed in (42, 43)]
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text(json.dumps(RAW), encoding="utf-8")
    out = tmp_path / "out"
    args = ["transfer", "--runs", *runs, "--mapping", str(mapping), "--extra", "wiki"]
    assert main([*args, "--tokenizer", "none", "--bootstrap", "100", "--out", str(out)]) == 0
    assert "Transfer to wiki (KazNERD -> WikiANN)" in capsys.readouterr().out
    report = json.loads((out / "transfer.json").read_text(encoding="utf-8"))
    assert report["reference_tokenizer"] is None
    with (out / "transfer.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["step"] for r in rows] == ["kaznerd", "kaznerd_mapped", "wiki"]
    assert rows[2]["n_runs"] == "2" and rows[2]["f1_ci_low"] != ""
