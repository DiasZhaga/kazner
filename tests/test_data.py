import hashlib
from pathlib import Path

import pytest

from kazner.data import (
    KAZNERD_SPLITS,
    IOB2FormatError,
    build_label2id,
    download_kaznerd,
    read_iob2,
    read_splits,
    sample_fraction,
    sample_indices,
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
