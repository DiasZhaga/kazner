from pathlib import Path

import pytest

from kazner.contract import (
    ContractError,
    Predictions,
    convert_pilot_predictions,
    read_predictions,
    write_predictions,
)

TINY = Path(__file__).parent / "data" / "kaznerd_tiny"
HEADER = "sentence_id,word_idx,word,gold,pred\n"


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8", newline="")
    return path


@pytest.fixture
def sample() -> Predictions:
    return Predictions(
        sentence_ids=[0, 1],
        words=[["Асан", "NA", "барды"], ["null", ","]],
        gold=[["B-PERSON", "I-PERSON", "O"], ["O", "O"]],
        pred=[["B-PERSON", "O", "O"], ["O", "O"]],
        truncated=[[False, False, False], [False, True]],
    )


def test_round_trip_keeps_every_value_as_a_string(tmp_path, sample):
    path = write_predictions(tmp_path / "p.csv", sample)
    assert path.read_text(encoding="utf-8").splitlines()[0] == (
        "sentence_id,word_idx,word,gold,pred,truncated"
    )
    loaded = read_predictions(path)
    assert loaded == sample  # "NA", "null" and "," survive as tokens
    assert loaded.n_words == 5
    assert loaded.n_truncated_words == 1


def test_truncated_column_is_optional(tmp_path):
    path = write(tmp_path / "p.csv", HEADER + "0,0,a,O,O\n0,1,b,B-X,B-X\n")
    loaded = read_predictions(path)
    assert loaded.gold == [["O", "B-X"]]
    assert loaded.n_truncated_words == 0


def test_rows_may_come_in_any_order(tmp_path):
    path = write(tmp_path / "p.csv", HEADER + "5,1,b,O,O\n2,0,x,O,O\n5,0,a,O,O\n")
    loaded = read_predictions(path)
    assert loaded.sentence_ids == [5, 2]
    assert loaded.words == [["a", "b"], ["x"]]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("sentence_id,word_idx,word,gold\n0,0,a,O\n", "missing column.*pred"),
        (HEADER.strip() + ",score\n0,0,a,O,O,1\n", "unknown column.*score"),
        (HEADER + "0,0,a,O,O\n0,2,b,O,O\n", r"word_idx must run 0\.\.1"),
        (HEADER + "0,0,a,O,O\n0,0,b,O,O\n", "duplicate word_idx 0"),
        (HEADER + "0,0,a,O,\n", "empty value"),
        (HEADER + "0,0,a,O,O,extra\n", "wrong number of fields"),
        (HEADER + "x,0,a,O,O\n", "must be integers"),
        (HEADER + "0,0,a,O,PERSON\n", "unknown pred label 'PERSON'"),
        (HEADER.strip() + ",truncated\n0,0,a,O,O,yes\n", "truncated must be 0 or 1"),
    ],
)
def test_invalid_files_are_rejected(tmp_path, text, message):
    with pytest.raises(ContractError, match=message):
        read_predictions(write(tmp_path / "bad.csv", text))


def test_known_label_set_is_enforced(tmp_path):
    path = write(tmp_path / "p.csv", HEADER + "0,0,a,B-GPE,O\n")
    assert read_predictions(path, labels={"O", "B-GPE"}).gold == [["B-GPE"]]
    with pytest.raises(ContractError, match="unknown gold label 'B-GPE'"):
        read_predictions(path, labels={"O", "B-PERSON"})


def test_predictions_object_validates_lengths():
    with pytest.raises(ContractError, match="differ in length"):
        Predictions([0], [["a", "b"]], [["O"]], [["O", "O"]])
    with pytest.raises(ContractError, match="differ in length"):
        Predictions([0, 1], [["a"]], [["O"]], [["O"]])


# --- pilot formats ---------------------------------------------------------------------

CROSS = "sentence_id,token,true_label,pred_label\n"
MBERT = "sent_id,token_id,token,gold,pred\n"


def test_convert_cross_lingual_format_derives_word_idx(tmp_path):
    path = write(tmp_path / "x.csv", CROSS + "0,Айгүл,B-PERSON,B-PERSON\n0,NA,O,O\n1,b,O,B-DATE\n")
    p = convert_pilot_predictions(path)
    assert p.words == [["Айгүл", "NA"], ["b"]]
    assert p.pred == [["B-PERSON", "O"], ["B-DATE"]]
    assert p.n_truncated_words == 0


def test_convert_mbert_format(tmp_path):
    path = write(tmp_path / "m.csv", MBERT + "0,0,a,B-X,B-X\n0,1,b,O,O\n1,0,c,O,O\n")
    assert convert_pilot_predictions(path).gold == [["B-X", "O"], ["O"]]


def test_convert_mbert_format_checks_token_id(tmp_path):
    path = write(tmp_path / "m.csv", MBERT + "0,1,a,O,O\n")
    with pytest.raises(ContractError, match="unexpected token_id"):
        convert_pilot_predictions(path)


def test_gold_file_restores_truncated_words(tmp_path):
    # Test split of the tiny corpus: sentence 0 has 6 words; pretend the model saw 4.
    text = CROSS + "".join(
        f"0,{w},{g},{g}\n"
        for w, g in [
            ("Марат", "B-PERSON"),
            ("Сейітов", "I-PERSON"),
            ("зауытта", "O"),
            ("жұмыс", "O"),
        ]
    )
    text += "".join(
        f"1,{w},{g},O\n"
        for w, g in [("Ертең", "B-DATE"), ("Шымкентте", "B-LOCATION"), ("концерт", "O"),
                     ("болады", "O"), (".", "O")]
    )  # fmt: skip
    p = convert_pilot_predictions(write(tmp_path / "x.csv", text), gold_path=TINY / "IOB2_test.txt")

    assert p.words[0][-2:] == ["істейді", "."]
    assert p.pred[0][-2:] == ["O", "O"]
    assert p.truncated[0] == [False] * 4 + [True] * 2
    assert p.n_truncated_words == 2


def test_gold_file_must_match(tmp_path):
    gold = TINY / "IOB2_test.txt"
    one_sentence = write(tmp_path / "a.csv", CROSS + "0,Марат,B-PERSON,O\n")
    with pytest.raises(ContractError, match="1 sentences, gold file has 2"):
        convert_pilot_predictions(one_sentence, gold_path=gold)
    wrong_word = write(tmp_path / "b.csv", CROSS + "0,Асан,B-PERSON,O\n1,Ертең,B-DATE,O\n")
    with pytest.raises(ContractError, match="sentence 0: predictions do not match"):
        convert_pilot_predictions(wrong_word, gold_path=gold)


def test_unknown_pilot_format(tmp_path):
    with pytest.raises(ContractError, match="not a known pilot prediction format"):
        convert_pilot_predictions(write(tmp_path / "x.csv", "a,b\n1,2\n"))
