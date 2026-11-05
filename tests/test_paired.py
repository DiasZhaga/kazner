import json

import numpy as np
import pytest

from kazner.cli import main
from kazner.contract import Predictions, write_predictions
from kazner.evaluate import compute_metrics
from kazner.paired import paired, restrict_types, select_runs, sign_flip_p

WORDS = [["Асан", "Алматыға", "барды"], ["Кеше", "банк", "ашылды"], ["Марат", "Сейітов"]]
GOLD = [["B-PERSON", "B-GPE", "O"], ["B-DATE", "B-ORGANISATION", "O"], ["B-PERSON", "I-PERSON"]]
#: Predictions per arm and seed: arm B finds more entities than arm A.
PRED = {
    ("a", 1): [["B-PERSON", "O", "O"], ["O", "O", "O"], ["B-PERSON", "I-PERSON"]],
    ("a", 2): [["B-PERSON", "B-LOCATION", "O"], ["B-DATE", "O", "O"], ["O", "O"]],
    ("a", 3): [["O", "B-GPE", "O"], ["B-DATE", "O", "O"], ["B-PERSON", "O"]],
    ("b", 1): [["B-PERSON", "B-GPE", "O"], ["B-DATE", "O", "O"], ["B-PERSON", "I-PERSON"]],
    ("b", 2): [["B-PERSON", "B-GPE", "O"], ["B-DATE", "B-ORGANISATION", "O"], ["O", "O"]],
    ("b", 3): [["B-PERSON", "B-GPE", "O"], ["O", "O", "O"], ["B-PERSON", "I-PERSON"]],
}


def write_run(root, run_id, name, seed, pred, gold=GOLD, complete=True):
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    config = {"run_id": run_id, "name": name, "seed": seed}
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    write_predictions(run_dir / "predictions.csv", Predictions([0, 1, 2], WORDS, gold, pred))
    if complete:
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")


@pytest.fixture
def runs(tmp_path):
    for (arm, seed), pred in PRED.items():
        write_run(tmp_path, f"{arm}{seed:011d}", f"arm_{arm}", seed, pred)
    write_run(tmp_path, "incomplete00", "arm_a", 4, PRED["a", 1], complete=False)
    return tmp_path


def f1(arm, seed, types=None):
    gold = restrict_types(GOLD, types)
    return compute_metrics(gold, restrict_types(PRED[arm, seed], types))["f1"]


def test_select_runs_takes_completed_runs_of_one_name_by_seed(runs):
    selected = select_runs(runs, "arm_a")
    assert [(r.seed, r.run_id) for r in selected] == [
        (1, "a00000000001"),
        (2, "a00000000002"),
        (3, "a00000000003"),
    ]  # the incomplete run of seed 4 is ignored
    assert selected[0].predictions.pred == PRED["a", 1]
    with pytest.raises(ValueError, match="no completed runs named 'arm_c'"):
        select_runs(runs, "arm_c")
    write_run(runs, "again0000001", "arm_a", 1, PRED["a", 1])
    with pytest.raises(ValueError, match=r"several runs of 'arm_a' for one seed: \{1: "):
        select_runs(runs, "arm_a")


def test_scores_and_deltas_equal_seqeval_per_seed(runs):
    report = paired(select_runs(runs, "arm_a"), select_runs(runs, "arm_b"), replicates=200)
    for arm in ("a", "b"):
        for seed in (1, 2, 3):
            assert report[arm]["runs"][str(seed)]["f1"] == pytest.approx(f1(arm, seed))
    deltas = [f1("b", s) - f1("a", s) for s in (1, 2, 3)]
    assert report["delta"] == pytest.approx(np.mean(deltas))
    assert report["per_seed_delta"] == {
        str(s): pytest.approx(d) for s, d in zip((1, 2, 3), deltas, strict=True)
    }
    assert report["b_better_seeds"] == sum(d > 0 for d in deltas) == 3
    assert report["a"]["f1_mean"] == pytest.approx(np.mean([f1("a", s) for s in (1, 2, 3)]))
    assert report["seeds"] == [1, 2, 3] and report["sign_flip"]["patterns"] == 8
    assert report["sign_flip"]["p_two_sided"] == pytest.approx(2 / 8)  # all deltas positive


def test_bootstrap_is_reproducible_and_brackets_the_estimate(runs):
    a, b = select_runs(runs, "arm_a"), select_runs(runs, "arm_b")
    first = paired(a, b, replicates=500, seed=7)["bootstrap"]
    assert first == paired(a, b, replicates=500, seed=7)["bootstrap"]
    assert first["sentences"] == 3 and first["replicates"] == 500
    low, high = first["ci95"]
    assert low <= paired(a, b, replicates=10)["delta"] <= high
    assert 0 < first["p_two_sided"] <= 1


def test_identical_arms_give_a_zero_delta(runs):
    a = select_runs(runs, "arm_a")
    report = paired(a, a, replicates=200)
    assert report["delta"] == 0.0
    assert report["bootstrap"]["ci95"] == [0.0, 0.0]
    assert report["bootstrap"]["p_two_sided"] == 1.0
    assert report["sign_flip"]["p_two_sided"] == 1.0


def test_type_restrictions(runs):
    a, b = select_runs(runs, "arm_a"), select_runs(runs, "arm_b")
    person = paired(a, b, types=["PERSON"], replicates=50)
    assert person["a"]["runs"]["2"]["f1"] == pytest.approx(f1("a", 2, ["PERSON"]))
    full = compute_metrics(GOLD, PRED["a", 2])["per_type"]["PERSON"]["f1"]
    assert person["a"]["runs"]["2"]["f1"] == pytest.approx(full)  # one type: its own F1
    rest = paired(a, b, exclude_types=["PERSON"], replicates=50)
    kept = ["DATE", "GPE", "ORGANISATION"]
    assert rest["b"]["runs"]["2"]["f1"] == pytest.approx(f1("b", 2, kept))
    assert (rest["types"], rest["exclude_types"]) == (None, ["PERSON"])
    assert restrict_types([["B-DATE", "I-PERSON", "O"]], exclude_types={"DATE"}) == [
        ["O", "I-PERSON", "O"]
    ]
    with pytest.raises(ValueError, match="not both"):
        paired(a, b, types=["PERSON"], exclude_types=["GPE"])
    with pytest.raises(ValueError, match=r"types \['NORP'\] do not occur"):
        paired(a, b, types=["NORP"])


def test_arms_must_match_in_seeds_and_test_set(runs, tmp_path):
    a, b = select_runs(runs, "arm_a"), select_runs(runs, "arm_b")
    with pytest.raises(ValueError, match="different seeds"):
        paired(a[:2], b)
    other = tmp_path / "other"
    gold = [list(s) for s in GOLD]
    gold[1][0] = "O"
    write_run(other, "c00000000001", "arm_c", 1, PRED["a", 1], gold=gold)
    with pytest.raises(ValueError, match="another test set"):
        paired(a[:1], select_runs(other, "arm_c"))


def test_sign_flip_p_is_exact():
    assert sign_flip_p([1.0, -1.0]) == 1.0
    assert sign_flip_p([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2 / 16)
    assert sign_flip_p([0.5, 0.5, -0.1]) == pytest.approx(4 / 8)
    with pytest.raises(ValueError, match="at most 16 seeds"):
        sign_flip_p([1.0] * 17)


def test_paired_cli(runs, tmp_path, capsys):
    out = tmp_path / "paired.json"
    args = ["paired", "--runs", str(runs), "--a", "arm_a", "--b", "arm_b"]
    assert main([*args, "--bootstrap", "200", "--types", "PERSON,GPE", "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert (report["a_name"], report["b_name"]) == ("arm_a", "arm_b")
    assert report["types"] == ["GPE", "PERSON"]
    printed = capsys.readouterr().out
    assert "arm_b - arm_a (types: GPE, PERSON), seeds 1-3 (n=3)" in printed
    assert "sign-flip test over seeds" in printed
