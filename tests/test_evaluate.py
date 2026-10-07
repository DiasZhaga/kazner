import json
from pathlib import Path

import pytest

from kazner.cli import main
from kazner.contract import Predictions, write_predictions
from kazner.evaluate import compute_metrics, evaluate_file, evaluate_predictions

TINY = Path(__file__).parent / "data" / "kaznerd_tiny"

# Gold entities: PER (s1 0-1), LOC (s1 3), ORG (s2 0)      -> 3
# Pred entities: PER (s1 0-1), ORG (s2 0), LOC (s2 1)      -> 3
# Correct: PER and ORG -> P = R = F1 = 2/3. Tokens: 4 of 6 correct.
GOLD = [["B-PER", "I-PER", "O", "B-LOC"], ["B-ORG", "O"]]
PRED = [["B-PER", "I-PER", "O", "O"], ["B-ORG", "B-LOC"]]


def test_hand_computed_example():
    m = compute_metrics(GOLD, PRED)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert (m["n_gold_entities"], m["n_pred_entities"]) == (3, 3)
    assert m["per_type"]["PER"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1}
    assert m["per_type"]["LOC"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1}


def test_entity_boundaries_must_match_exactly():
    m = compute_metrics([["B-PER", "I-PER", "O"]], [["B-PER", "O", "O"]])
    assert (m["precision"], m["recall"], m["f1"]) == (0.0, 0.0, 0.0)
    assert m["accuracy"] == pytest.approx(2 / 3)


def test_default_mode_accepts_entity_starting_with_i():
    """seqeval default (conlleval) mode, as in the pilot: I-X after O starts an entity."""
    m = compute_metrics([["O", "B-LOC"]], [["O", "I-LOC"]])
    assert m["f1"] == 1.0


def test_no_entities_gives_zero_without_warnings(recwarn):
    m = compute_metrics([["O", "O"]], [["O", "O"]])
    assert (m["f1"], m["accuracy"]) == (0.0, 1.0)
    assert not [w for w in recwarn if "zero" in str(w.message).lower()]


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="same sentences and lengths"):
        compute_metrics([["O", "O"]], [["O"]])


def test_evaluate_predictions_adds_counts():
    truncated = [[False, False, False, True], [False, False]]
    m = evaluate_predictions(Predictions([0, 1], [list("abcd"), list("ef")], GOLD, PRED, truncated))
    assert (m["n_sentences"], m["n_words"], m["n_truncated_words"]) == (2, 6, 1)
    assert m["seqeval_mode"] == "default"


def test_evaluate_file_writes_metrics_json(tmp_path):
    path = write_predictions(
        tmp_path / "p.csv", Predictions([0, 1], [list("abcd"), list("ef")], GOLD, PRED)
    )
    metrics = evaluate_file(path, out_path=tmp_path / "out" / "metrics.json")
    saved = json.loads((tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))
    assert saved == metrics
    assert saved["f1"] == pytest.approx(2 / 3)


def test_cli_evaluate(tmp_path, capsys):
    gold = [["B-PERSON", "O"]]
    path = write_predictions(tmp_path / "p.csv", Predictions([0], [["Асан", "."]], gold, gold))
    out = tmp_path / "metrics.json"

    code = main(
        ["evaluate", "--predictions", str(path), "--out", str(out),
         "--labels-from", str(TINY / "IOB2_train.txt")]
    )  # fmt: skip

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["f1"] == 1.0
    assert "per_type" not in printed  # summary only on stdout
    assert "PERSON" in json.loads(out.read_text(encoding="utf-8"))["per_type"]


def test_cli_evaluate_rejects_labels_outside_train_set(tmp_path):
    gold = [["B-GPE"]]
    path = write_predictions(tmp_path / "p.csv", Predictions([0], [["x"]], gold, gold))
    with pytest.raises(ValueError, match="unknown gold label 'B-GPE'"):
        main(
            ["evaluate", "--predictions", str(path), "--labels-from", str(TINY / "IOB2_train.txt")]
        )
