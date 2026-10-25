"""Corpus sampling and cleaning (#34). All text is SYNTHETIC, written for these tests."""

import hashlib
import importlib.util
import io
import json
import unicodedata
from pathlib import Path

import pytest

from kazner.corpus import (
    STAGES,
    CorpusBuilder,
    SentenceIndex,
    build_corpus,
    is_kazakh,
    iter_documents,
    match_words,
    normalize_line,
    seeded_row_groups,
)

ROOT = Path(__file__).resolve().parent.parent
TINY = Path(__file__).parent / "data" / "kaznerd_tiny"

KEPT = "Біз бүгін жаңа кітапхананың ашылуына қатыстық."
RUSSIAN = "Мы сегодня были на открытии новой библиотеки города."
ENGLISH = "Today we visited the opening of the new city library."
SHORT = "Сәлем бәріне!"
OVERLAP = "Кеше ауыл әкімі жаңа мектептің құрылысын бастады, деп хабарлады."
KAZNERD_SENTENCE = ["Ауыл", "әкімі", "жаңа", "мектептің", "құрылысын", "бастады", "."]


# --- line rules --------------------------------------------------------------------------


def test_normalize_line_composes_letters_and_collapses_whitespace():
    decomposed = unicodedata.normalize("NFD", "  Қазақстан   мен\tРесей ")
    assert decomposed != unicodedata.normalize("NFC", decomposed)
    assert normalize_line(decomposed) == "Қазақстан мен Ресей"


@pytest.mark.parametrize(
    ("line", "kazakh"),
    [
        (KEPT, True),
        ("Apple компаниясы жаңа телефонын көрсетті.", True),  # a Latin name is fine
        (RUSSIAN, False),  # Cyrillic, but no letter specific to Kazakh
        (ENGLISH, False),
        ("This is a short test of the new system қ", False),  # mostly Latin
        ("2020 2021 2022", False),  # no letters
    ],
)
def test_is_kazakh(line, kazakh):
    assert is_kazakh(line) is kazakh


def test_match_words_ignore_case_and_punctuation():
    assert match_words("Барды. «Астана»-ға 2020-жылы!") == ("барды", "астана", "ға", "2020", "жылы")


def test_sentence_index_finds_long_sentences_inside_lines():
    short = ["Рахмет", "."]
    index = SentenceIndex([KAZNERD_SENTENCE, short])
    assert index.size == 1  # the two-word sentence is not indexed
    assert index.contains(match_words(OVERLAP))  # other case and punctuation, inside a line
    assert not index.contains(match_words("Ауыл әкімі жаңа мектептің құрылысын тоқтатты."))
    assert not index.contains(match_words("Рахмет, достар, бәріне үлкен рахмет."))


# --- builder -----------------------------------------------------------------------------


def build(document, target_words=10**6, index=None):
    out = io.BytesIO()
    builder = CorpusBuilder(index or SentenceIndex([KAZNERD_SENTENCE]), target_words, out)
    builder.add_document(document)
    return builder, out.getvalue()


def test_every_line_is_counted_once_at_the_first_rule_it_fails():
    long_line = " ".join(["сөз"] * 501)
    document = "\n".join([KEPT, SHORT, RUSSIAN, ENGLISH, "", KEPT, OVERLAP, long_line, "  "])
    builder, written = build(document)
    assert dict(builder.counts) == {
        "lines": 7, "too_short": 1, "too_long": 1, "not_kazakh": 2, "duplicate": 1,
        "kaznerd": 1, "kept": 1,
    }  # fmt: skip
    assert written == (KEPT + "\n").encode("utf-8")
    assert builder.sha256 == hashlib.sha256(written).hexdigest()
    assert builder.words == 6 and builder.documents == 1


def test_builder_stops_at_the_target():
    lines = [f"Біз {n}-сыныпта жаңа кітап оқыдық." for n in range(1, 6)]  # 5 words each
    builder, written = build("\n".join(lines), target_words=12)
    assert builder.done and builder.words == 15  # stops after the line that crosses 12
    assert written.decode("utf-8").splitlines() == lines[:3]


# --- parquet sampling --------------------------------------------------------------------


@pytest.fixture
def parquet_dir(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    for f in range(2):
        texts = [f"{KEPT} Файл {f}, құжат {i}." for i in range(6)]  # unique, 10 words each
        table = pa.table({"text": texts, "id": [f"{f}-{i}" for i in range(6)]})
        pq.write_table(table, tmp_path / f"part{f}.parquet", row_group_size=2)
    return tmp_path


def open_parquet(directory):
    import pyarrow.parquet as pq

    paths = sorted(directory.glob("*.parquet"))
    return [pq.ParquetFile(p) for p in paths], [p.name for p in paths]


def test_row_groups_are_visited_in_a_seeded_order(parquet_dir):
    files, _ = open_parquet(parquet_dir)
    order = seeded_row_groups(files, seed=1)
    assert sorted(order) == [(f, g) for f in range(2) for g in range(3)]  # each group once
    assert order == seeded_row_groups(files, seed=1)
    assert order != seeded_row_groups(files, seed=2)
    docs = list(iter_documents(files, seed=1))
    assert [(f, g) for f, g, _ in docs[::2]] == order  # two documents per row group
    first_file, first_group = order[0]
    assert docs[0][2].endswith(f"Файл {first_file}, құжат {2 * first_group}.")


def test_build_corpus_reads_only_the_row_groups_it_needs(parquet_dir):
    files, names = open_parquet(parquet_dir)
    out = io.BytesIO()
    sample = build_corpus(files, names, SentenceIndex([]), out, target_words=30, seed=1)
    assert sample["complete"] and sample["words"] == 30  # 3 documents of 10 words
    assert [g["documents"] for g in sample["row_groups_used"]] == [2, 1]
    assert sample["row_groups_total"] == 6
    counts = sample["line_counts"]
    assert list(counts) == list(STAGES)
    assert counts["lines"] == sum(v for k, v in counts.items() if k != "lines")
    assert sample["sha256"] == hashlib.sha256(out.getvalue()).hexdigest()


# --- the download script, offline --------------------------------------------------------


def load_script():
    spec = importlib.util.spec_from_file_location(
        "download_kk_corpus", ROOT / "scripts" / "download_kk_corpus.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_pins_the_train_split_of_a_fixed_revision():
    script = load_script()
    assert len(script.REVISION) == 40
    assert all("/train/" in path for path in script.FILES)


def test_script_writes_corpus_and_manifest_from_local_parquet(parquet_dir, tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pq.write_table(pa.table({"text": [OVERLAP]}), parquet_dir / "part2.parquet")
    kaznerd = tmp_path / "kaznerd"
    kaznerd.mkdir()
    rows = "\n".join(f"{w} O" for w in KAZNERD_SENTENCE) + "\n"
    for name in ("IOB2_train.txt", "IOB2_valid.txt", "IOB2_test.txt"):
        (kaznerd / name).write_text(rows, encoding="utf-8")

    out = tmp_path / "corpus" / "kk.txt"
    args = ["--local", str(parquet_dir), "--kaznerd", str(kaznerd), "--out", str(out)]
    assert load_script().main([*args, "--words", "1000", "--seed", "3"]) == 1  # ran out of text
    manifest = json.loads((out.parent / "kk.manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is False and manifest["source"]["repo"] is None
    assert manifest["line_counts"]["kaznerd"] == 1  # the overlapping line was removed
    assert manifest["line_counts"]["kept"] == 12 == manifest["lines"]
    assert manifest["rules"]["kaznerd_sentences_indexed"] == 1
    assert manifest["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert b"\r\n" not in out.read_bytes()
    assert not out.with_name("kk.txt.part").exists()


def test_dry_run_reads_only_metadata(parquet_dir, tmp_path, capsys):
    out = tmp_path / "never.txt"
    args = ["--local", str(parquet_dir), "--out", str(out), "--kaznerd", str(TINY), "--dry-run"]
    assert load_script().main(args) == 0
    printed = capsys.readouterr().out
    assert "part0.parquet: 6 documents, 3 row groups" in printed
    assert not out.exists()
