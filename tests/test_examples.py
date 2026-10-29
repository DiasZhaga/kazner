"""The synthetic example shipped in the container image (used by its smoke test)."""

import json
from pathlib import Path

import pytest

from kazner.cli import main
from kazner.data import read_iob2
from kazner.evaluate import evaluate_file

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "contract_example.csv"
TINY_TEST = ROOT / "tests" / "data" / "kaznerd_tiny" / "IOB2_test.txt"


def test_example_is_the_synthetic_test_split_with_one_missed_entity():
    tokens, labels = read_iob2(TINY_TEST)
    rows = [line.split(",") for line in EXAMPLE.read_text(encoding="utf-8").splitlines()[1:]]
    assert [r[2] for r in rows] == [w for sentence in tokens for w in sentence]
    assert [r[3] for r in rows] == [label for sentence in labels for label in sentence]


def test_example_scores_are_hand_computed():
    # Gold: PERSON, DATE, LOCATION; LOCATION is missed. Tokens: 10 of 11 correct.
    m = evaluate_file(EXAMPLE)
    assert (m["precision"], m["recall"]) == (1.0, pytest.approx(2 / 3))
    assert m["f1"] == pytest.approx(0.8)
    assert m["accuracy"] == pytest.approx(10 / 11)


def test_example_through_the_cli(capsys):
    assert main(["evaluate", "--predictions", str(EXAMPLE)]) == 0
    assert json.loads(capsys.readouterr().out)["f1"] == pytest.approx(0.8)


def test_dockerfile_ships_the_example():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY examples/contract_example.csv /opt/kazner/examples/" in dockerfile
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").split()
    assert "examples" not in ignored and "tests" not in ignored
