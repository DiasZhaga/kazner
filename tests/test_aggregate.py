import csv
import json
from pathlib import Path

import pytest

from kazner.aggregate import aggregate, config_field, format_table, load_runs, write_csv
from kazner.cli import main


def make_run(root: Path, rid: str, method: str, seed: int, f1: float, code="tree-a", done=True):
    d = root / rid
    d.mkdir(parents=True)
    config = {
        "run_id": rid,
        "seed": seed,
        "method": {"name": method, "lora": {"r": 8}},
        "data": {"train_fraction": 0.1},
        "model": {"name": "mbert"},
        "training": {"learning_rate": 5e-5},
    }
    (d / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (d / "environment.json").write_text(json.dumps({"code": {"code_ref": code}}), encoding="utf-8")
    if done:
        block = {"precision": f1, "recall": f1, "f1": f1, "accuracy": 0.9}
        metrics = {"test_contract": block, "test_trainer": {**block, "f1": f1 + 0.01}}
        (d / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


@pytest.fixture
def runs(tmp_path):
    for i, (seed, f1) in enumerate([(42, 0.90), (43, 0.92), (44, 0.91)]):
        make_run(tmp_path, f"full{i}", "full", seed, f1)
    make_run(tmp_path, "lora0", "lora", 42, 0.68)
    make_run(tmp_path, "lora1", "lora", 43, 0.70, done=False)  # interrupted
    (tmp_path / "not-a-run").mkdir()
    return tmp_path


def test_mean_sd_and_n_per_group(runs):
    with pytest.warns(UserWarning, match="lora1: no metrics.json"):
        loaded = load_runs(runs)
    rows = {r["method"]: r for r in aggregate(loaded, ["method", "fraction"])}

    full = rows["full"]
    assert (full["n"], full["fraction"]) == (3, 0.1)
    assert full["f1_mean"] == pytest.approx(0.91)
    assert full["f1_sd"] == pytest.approx(0.01)  # sample sd of 0.90, 0.92, 0.91
    assert full["seeds"] == "42 43 44"

    lora = rows["lora"]
    assert lora["n"] == 1 and lora["f1_sd"] is None  # n = 1: no sd


def test_trainer_block_and_dotted_paths(runs):
    with pytest.warns(UserWarning):
        loaded = load_runs(runs)
    rows = aggregate(loaded, ["method.lora.r", "lr"], block="test_trainer")
    assert len(rows) == 1 and rows[0]["n"] == 4
    assert rows[0]["f1_mean"] == pytest.approx((0.91 + 0.93 + 0.92 + 0.69) / 4)


def test_mixed_code_versions_are_warned_about(tmp_path):
    make_run(tmp_path, "a", "full", 42, 0.9, code="tree-a")
    make_run(tmp_path, "b", "full", 43, 0.9, code="tree-b")
    with pytest.warns(UserWarning, match="mixes runs from different code versions"):
        aggregate(load_runs(tmp_path), ["method"])


def test_errors_for_unknown_key_and_block(runs):
    with pytest.warns(UserWarning):
        loaded = load_runs(runs)
    with pytest.raises(KeyError, match="no field 'training.warmup'"):
        aggregate(loaded, ["training.warmup"])
    with pytest.raises(ValueError, match="unknown metric block"):
        aggregate(loaded, ["method"], block="validation")


def test_csv_and_table(runs, tmp_path):
    with pytest.warns(UserWarning):
        rows = aggregate(load_runs(runs), ["method"])
    path = write_csv(tmp_path / "out" / "summary.csv", rows)
    with path.open(encoding="utf-8") as f:
        saved = list(csv.DictReader(f))
    assert [r["method"] for r in saved] == ["full", "lora"]
    assert saved[1]["f1_sd"] == ""  # n = 1 written as an empty cell
    table = format_table(rows, ["method"])
    assert "method=full  n=3  f1=0.9100 ± 0.0100" in table
    assert "method=lora  n=1  f1=0.6800  " in table


def test_cli_aggregate(runs, tmp_path, capsys):
    out = tmp_path / "summary.csv"
    with pytest.warns(UserWarning):
        code = main(["aggregate", "--runs", str(runs), "--by", "method", "--out", str(out)])
    assert code == 0 and out.exists()
    assert "method=full  n=3" in capsys.readouterr().out


def test_config_field_resolves_aliases_and_dotted_paths():
    config = {"method": {"name": "lora", "lora": {"r": 16}}, "training": {"learning_rate": 1e-3}}
    assert config_field(config, "method") == "lora"
    assert config_field(config, "lr") == 1e-3
    assert config_field(config, "method.lora.r") == 16
    with pytest.raises(KeyError, match="model.name"):
        config_field(config, "model")
