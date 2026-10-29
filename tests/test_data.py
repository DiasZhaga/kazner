import hashlib
import io
import json
from pathlib import Path

import pytest

from kazner.data import (
    KAZNERD_SPLITS,
    WIKIANN_LABELS,
    WIKIANN_SHA256,
    IOB2FormatError,
    build_label2id,
    download_kaznerd,
    download_wikiann_kk,
    read_iob2,
    read_splits,
    sample_fraction,
    sample_indices,
    training_subsample,
    wikiann_to_iob2,
    write_iob2,
)
from reference import pilot_data

TINY = Path(__file__).parent / "data" / "kaznerd_tiny"


def write(path: Path, text: str, newline: str = "\n") -> Path:
    path.write_bytes(text.replace("\n", newline).encode("utf-8"))
    return path


# --- read_iob2 -------------------------------------------------------------------------


def test_read_iob2_sentences_and_labels():
    tokens, labels = read_iob2(TINY / "IOB2_train.txt")
    assert len(tokens) == len(labels) == 4
    assert tokens[0] == ["Асан", "Бекұлы", "Алматыға", "барды", "."]
    assert labels[0] == ["B-PERSON", "I-PERSON", "B-LOCATION", "O", "O"]
    assert all(len(t) == len(lab) for t, lab in zip(tokens, labels, strict=True))


def test_label_zero_is_normalised_to_o():
    _, labels = read_iob2(TINY / "IOB2_train.txt")
    assert labels[1] == ["B-DATE", "B-ORGANISATION", "I-ORGANISATION", "O", "O"]
    assert "0" not in {label for sentence in labels for label in sentence}


def test_repeated_and_trailing_blank_lines_create_no_empty_sentences(tmp_path):
    path = write(tmp_path / "x.txt", "\n\na O\n\n\n\nb B-PERSON\n\n\n")
    assert read_iob2(path) == ([["a"], ["b"]], [["O"], ["B-PERSON"]])


def test_file_without_trailing_newline(tmp_path):
    path = write(tmp_path / "x.txt", "a O\nb O")
    assert read_iob2(path) == ([["a", "b"]], [["O", "O"]])


def test_token_is_first_column_and_label_last(tmp_path):
    path = write(tmp_path / "x.txt", "Астана NNP B-GPE\n")
    assert read_iob2(path) == ([["Астана"]], [["B-GPE"]])


def test_crlf_line_endings(tmp_path):
    lf = write(tmp_path / "lf.txt", "a B-PERSON\nb O\n\nc O\n")
    crlf = write(tmp_path / "crlf.txt", "a B-PERSON\nb O\n\nc O\n", newline="\r\n")
    assert read_iob2(crlf) == read_iob2(lf)


def test_malformed_row_raises_with_location(tmp_path):
    path = write(tmp_path / "bad.txt", "a O\nlonely\nb O\n")
    with pytest.raises(IOB2FormatError, match=r"bad\.txt:2: .*'lonely'"):
        read_iob2(path)


@pytest.mark.parametrize("name", list(KAZNERD_SPLITS.values()))
def test_read_iob2_matches_pilot_on_well_formed_files(name):
    assert read_iob2(TINY / name) == pilot_data.read_iob2(TINY / name)


def test_pilot_silently_skipped_malformed_rows(tmp_path):
    """Documents the intended deviation: the pilot dropped the row, kazner raises."""
    path = write(tmp_path / "bad.txt", "a O\nlonely\nb O\n")
    assert pilot_data.read_iob2(path) == ([["a", "b"]], [["O", "O"]])
    with pytest.raises(IOB2FormatError):
        read_iob2(path)


def test_read_splits_reads_every_kaznerd_file():
    splits = read_splits(TINY)
    assert set(splits) == {"train", "validation", "test"}
    assert [len(splits[s][0]) for s in ("train", "validation", "test")] == [4, 2, 2]


def test_read_splits_reports_missing_files(tmp_path):
    write(tmp_path / "IOB2_train.txt", "a O\n")
    with pytest.raises(FileNotFoundError, match="IOB2_valid.txt, IOB2_test.txt"):
        read_splits(tmp_path)


# --- build_label2id --------------------------------------------------------------------


def test_build_label2id_is_sorted_and_invertible():
    _, labels = read_iob2(TINY / "IOB2_train.txt")
    label2id, id2label = build_label2id(labels)
    assert list(label2id) == sorted(label2id)
    assert label2id["B-DATE"] == 0
    assert {id2label[i]: i for i in id2label} == label2id


def test_build_label2id_matches_pilot():
    _, labels = read_iob2(TINY / "IOB2_train.txt")
    assert build_label2id(labels) == pilot_data.build_label_maps(labels)


def test_subsampling_does_not_change_the_mapping():
    tokens, labels = read_iob2(TINY / "IOB2_train.txt")
    full_mapping = build_label2id(labels)
    for seed in range(20):
        sub_tokens, sub_labels = sample_fraction(tokens, labels, 0.25, seed)
        # The mapping is always built from the full split, so it never depends on the sample ...
        assert build_label2id(labels) == full_mapping
        # ... and it covers every label the sample can contain.
        assert {lab for s in sub_labels for lab in s} <= set(full_mapping[0])
    # A mapping built from a one-sentence sample would miss labels and shift ids.
    _, one = sample_fraction(tokens, labels, 0.25, seed=0)
    assert build_label2id(one)[0] != full_mapping[0]


def test_training_subsample_is_the_sample_cut_to_max_samples():
    tokens, labels = read_iob2(TINY / "IOB2_train.txt")
    sample = sample_fraction(tokens, labels, 1.0, 42)
    assert training_subsample(tokens, labels, 1.0, 42) == sample
    cut = training_subsample(tokens, labels, 1.0, 42, max_samples=2)
    assert cut == (sample[0][:2], sample[1][:2])


# --- sampling --------------------------------------------------------------------------


def test_ten_percent_of_kaznerd_train_is_9023_sentences():
    indices = sample_indices(90_228, 0.10, seed=42)
    assert len(indices) == 9_023
    assert indices == sorted(set(indices))
    assert indices == sample_indices(90_228, 0.10, seed=42)


@pytest.mark.parametrize("fraction", [0.01, 0.1, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("seed", [0, 42, 1234])
def test_sample_fraction_matches_pilot(fraction, seed):
    tokens = [[f"w{i}", "."] for i in range(257)]
    labels = [["O", "O"] for _ in range(257)]
    tokens[3] = ["Астана", "."]
    sub_tokens, sub_labels = sample_fraction(tokens, labels, fraction, seed)
    oracle = pilot_data.sample_train_subset(tokens, labels, fraction, seed)
    assert sub_tokens == oracle["tokens"]
    assert sub_labels == oracle["labels_str"]


def test_tiny_fraction_keeps_at_least_one_sentence():
    assert len(sample_indices(10, 0.001, seed=1)) == 1


@pytest.mark.parametrize("fraction", [0, -0.1, 1.5])
def test_invalid_fraction_is_rejected(fraction):
    with pytest.raises(ValueError, match="fraction"):
        sample_indices(10, fraction, seed=1)


def test_sample_fraction_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="2 token sentences but 1 label"):
        sample_fraction([["a"], ["b"]], [["O"]], 0.5, seed=1)


# --- download_kaznerd ------------------------------------------------------------------


def fake_hub(contents: dict[str, bytes]):
    calls: list[str] = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        return contents[url.rsplit("/", 1)[1]]

    return fetch, calls


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_download_writes_verified_files_and_is_idempotent(tmp_path):
    contents = {"IOB2_train.txt": b"a O\r\n", "IOB2_test.txt": b"b O\n"}
    files = {name: sha(data) for name, data in contents.items()}
    fetch, calls = fake_hub(contents)

    paths = download_kaznerd(tmp_path / "kaznerd", files=files, commit="abc", fetch=fetch)

    assert [p.read_bytes() for p in paths] == list(contents.values())  # bytes unchanged
    assert calls[0].endswith("/IS2AI/KazNERD/abc/KazNERD/IOB2_train.txt")
    download_kaznerd(tmp_path / "kaznerd", files=files, fetch=fetch)
    assert len(calls) == 2  # second call found valid files and fetched nothing
    download_kaznerd(tmp_path / "kaznerd", files=files, fetch=fetch, force=True)
    assert len(calls) == 4


def test_download_replaces_a_corrupted_file(tmp_path):
    data = b"a O\n"
    (tmp_path / "IOB2_test.txt").write_bytes(b"corrupted")
    fetch, calls = fake_hub({"IOB2_test.txt": data})
    download_kaznerd(tmp_path, files={"IOB2_test.txt": sha(data)}, fetch=fetch)
    assert (tmp_path / "IOB2_test.txt").read_bytes() == data
    assert len(calls) == 1


def test_download_rejects_checksum_mismatch(tmp_path):
    fetch, _ = fake_hub({"IOB2_test.txt": b"tampered"})
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download_kaznerd(tmp_path, files={"IOB2_test.txt": sha(b"original")}, fetch=fetch)
    assert not (tmp_path / "IOB2_test.txt").exists()


# --- WikiANN (#51) ---------------------------------------------------------------------

#: Synthetic WikiANN-style rows: tokens and ner_tags ids (1 B-PER, 3 B-ORG, 4 I-ORG, 5 B-LOC).
ROWS = [(["Айгүл", "Астанада", "тұрады"], [1, 5, 0]), (["Бек", "университет", "кеңесі"], [1, 3, 4])]
WIKI_PARQUET = "test-00000-of-00001.parquet"


def wikiann_parquet(rows=ROWS, names=WIKIANN_LABELS, metadata=True) -> bytes:
    """A parquet file laid out like the Hub's WikiANN files, with Hugging Face metadata."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table({"tokens": [r[0] for r in rows], "ner_tags": [r[1] for r in rows]})
    if metadata:
        features = {
            "tokens": {"feature": {"dtype": "string", "_type": "Value"}, "_type": "Sequence"},
            "ner_tags": {
                "feature": {"names": list(names), "_type": "ClassLabel"},
                "_type": "Sequence",
            },
        }
        info = json.dumps({"info": {"features": features}})
        table = table.replace_schema_metadata({"huggingface": info})
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


def test_write_iob2_round_trips_through_read_iob2(tmp_path):
    tokens, labels = [["Айгүл", "тұрады"], ["Бек"]], [["B-PER", "O"], ["B-PER"]]
    path = write_iob2(tmp_path / "x.txt", tokens, labels)
    assert read_iob2(path) == (tokens, labels)
    assert b"\r" not in path.read_bytes()


@pytest.mark.parametrize("word", ["", "a b", "a\tb"])
def test_write_iob2_rejects_tokens_that_would_break_the_format(tmp_path, word):
    with pytest.raises(IOB2FormatError, match="empty or has whitespace"):
        write_iob2(tmp_path / "x.txt", [[word]], [["O"]])


def test_wikiann_tags_are_named_by_the_class_names_in_the_file():
    tokens, labels = wikiann_to_iob2(wikiann_parquet())
    assert tokens == [r[0] for r in ROWS]
    assert labels == [["B-PER", "B-LOC", "O"], ["B-PER", "B-ORG", "I-ORG"]]


def test_wikiann_with_other_or_missing_class_names_is_rejected():
    shuffled = ("O", "B-LOC", "I-LOC", "B-PER", "I-PER", "B-ORG", "I-ORG")
    with pytest.raises(ValueError, match="unexpected ner_tags class names"):
        wikiann_to_iob2(wikiann_parquet(names=shuffled))
    with pytest.raises(ValueError, match="no ner_tags class names"):
        wikiann_to_iob2(wikiann_parquet(metadata=False))


def test_download_wikiann_writes_splits_and_a_manifest(tmp_path):
    data = wikiann_parquet()
    fetch, calls = fake_hub({WIKI_PARQUET: data})
    files = {"test": sha(data)}

    manifest = download_wikiann_kk(tmp_path, files=files, commit="abc", fetch=fetch)

    assert calls == [
        f"https://huggingface.co/datasets/unimelb-nlp/wikiann/resolve/abc/kk/{WIKI_PARQUET}"
    ]
    assert read_iob2(tmp_path / "IOB2_test.txt")[1] == [
        ["B-PER", "B-LOC", "O"],
        ["B-PER", "B-ORG", "I-ORG"],
    ]
    test = manifest["splits"]["test"]
    assert (test["sentences"], test["tokens"]) == (2, 6)
    assert test["entities"] == {"LOC": 1, "ORG": 1, "PER": 2}
    assert test["parquet_sha256"] == sha(data)
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == manifest
    download_wikiann_kk(tmp_path, files=files, fetch=fetch)
    assert len(calls) == 1  # the verified parquet file is reused


def test_download_wikiann_rejects_checksum_mismatch(tmp_path):
    fetch, _ = fake_hub({WIKI_PARQUET: b"tampered"})
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download_wikiann_kk(tmp_path, files={"test": sha(b"original")}, fetch=fetch)
    assert not (tmp_path / "IOB2_test.txt").exists()


def test_pinned_wikiann_files_are_the_three_splits():
    assert set(WIKIANN_SHA256) == set(KAZNERD_SPLITS)  # written under KazNERD's file names
