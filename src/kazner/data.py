"""Word-level IOB2 data: reading, label maps, deterministic subsampling and KazNERD download.

Behaviour follows the pilot (``ner-project/src/train_ner_mbert.py`` and
``run_peft_comparison.py``) with one intended change: malformed rows raise
:class:`IOB2FormatError` instead of being skipped silently.
"""

from __future__ import annotations

import hashlib
import math
import random
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path

Sentences = list[list[str]]

#: File names of the official KazNERD splits, keyed by split name.
KAZNERD_SPLITS: dict[str, str] = {
    "train": "IOB2_train.txt",
    "validation": "IOB2_valid.txt",
    "test": "IOB2_test.txt",
}

#: Upstream commit of https://github.com/IS2AI/KazNERD that the checksums refer to.
KAZNERD_COMMIT = "bd4333d0f5952b9fafb2ef2ac2fefa0ad3c0333f"
KAZNERD_URL = "https://raw.githubusercontent.com/IS2AI/KazNERD/{commit}/KazNERD/{name}"
#: SHA-256 of the files exactly as published (LF line endings).
KAZNERD_SHA256: dict[str, str] = {
    "IOB2_train.txt": "e5a80f6d4bbff499a4bf7cbf96a169feb761411151fb45abebeaae06995ff647",
    "IOB2_valid.txt": "29d584d1a0d9aca6d58b39eb8a93e154791d303ecff56ef93f12b3c889239bf1",
    "IOB2_test.txt": "359a1f61f6a21280c6c47b4fde9a5adcd07411d62015aa6e32d940c51f163b7a",
}


class IOB2FormatError(ValueError):
    """A row of an IOB2 file does not have the form ``<token> ... <label>``."""


def read_iob2(path: str | Path) -> tuple[Sentences, Sentences]:
    """Read a whitespace-separated IOB2 file into word-level sentences.

    The token is the first column and the label the last one; blank lines separate
    sentences. The label ``0`` is normalised to ``O``. Returns ``(tokens, labels)``.
    """
    path = Path(path)
    s_tokens: Sentences = []
    s_labels: Sentences = []
    tokens: list[str] = []
    labels: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                if tokens:
                    s_tokens.append(tokens)
                    s_labels.append(labels)
                    tokens, labels = [], []
                continue
            parts = line.split()
            if len(parts) < 2:
                raise IOB2FormatError(
                    f"{path}:{line_no}: expected '<token> ... <label>', got {line.rstrip()!r}"
                )
            label = parts[-1]
            tokens.append(parts[0])
            labels.append("O" if label == "0" else label)
    if tokens:
        s_tokens.append(tokens)
        s_labels.append(labels)
    return s_tokens, s_labels


def read_splits(
    data_dir: str | Path, splits: Mapping[str, str] = KAZNERD_SPLITS
) -> dict[str, tuple[Sentences, Sentences]]:
    """Read every split in ``splits`` (split name -> file name) from ``data_dir``."""
    data_dir = Path(data_dir)
    missing = [name for name in splits.values() if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{data_dir}: missing split file(s): {', '.join(missing)}")
    return {split: read_iob2(data_dir / name) for split, name in splits.items()}


def build_label2id(train_labels: Sentences) -> tuple[dict[str, int], dict[int, str]]:
    """Map the sorted unique labels of the FULL train split to ids (pilot ordering).

    Build the mapping from the full split, never from a subsample: a subsample may miss
    rare labels and would then shift every id after them.
    """
    unique = sorted({label for sentence in train_labels for label in sentence})
    label2id = {label: i for i, label in enumerate(unique)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def sample_indices(n_sentences: int, fraction: float, seed: int) -> list[int]:
    """Sorted indices of a deterministic subsample of ``max(1, ceil(n * fraction))`` sentences."""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    sample_size = max(1, math.ceil(n_sentences * fraction))
    rng = random.Random(seed)
    indices = list(range(n_sentences))
    rng.shuffle(indices)
    return sorted(indices[:sample_size])


def sample_fraction(
    tokens: Sentences, labels: Sentences, fraction: float, seed: int
) -> tuple[Sentences, Sentences]:
    """Deterministic sentence-level subsample, identical to the pilot's ``sample_train_subset``."""
    if len(tokens) != len(labels):
        raise ValueError(f"{len(tokens)} token sentences but {len(labels)} label sentences")
    selected = sample_indices(len(tokens), fraction, seed)
    return [tokens[i] for i in selected], [labels[i] for i in selected]


def training_subsample(
    tokens: Sentences,
    labels: Sentences,
    fraction: float,
    seed: int,
    max_samples: int | None = None,
) -> tuple[Sentences, Sentences]:
    """The sentences a run is trained on: :func:`sample_fraction`, then the first
    ``max_samples`` (``data.max_train_samples``) if set."""
    sub_tokens, sub_labels = sample_fraction(tokens, labels, fraction, seed)
    if max_samples is not None:
        sub_tokens, sub_labels = sub_tokens[:max_samples], sub_labels[:max_samples]
    return sub_tokens, sub_labels


def _fetch_url(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_kaznerd(
    dest_dir: str | Path,
    *,
    force: bool = False,
    files: Mapping[str, str] = KAZNERD_SHA256,
    commit: str = KAZNERD_COMMIT,
    fetch: Callable[[str], bytes] = _fetch_url,
) -> list[Path]:
    """Download the KazNERD IOB2 splits into ``dest_dir`` and verify their SHA-256.

    Files that already exist with the expected checksum are not downloaded again unless
    ``force`` is set. Bytes are written unchanged, so line endings stay as published.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, expected in files.items():
        target = dest_dir / name
        paths.append(target)
        if not force and target.is_file() and _sha256(target.read_bytes()) == expected:
            continue
        data = fetch(KAZNERD_URL.format(commit=commit, name=name))
        actual = _sha256(data)
        if actual != expected:
            raise ValueError(f"{name}: SHA-256 mismatch (expected {expected}, got {actual})")
        partial = target.with_name(target.name + ".part")
        partial.write_bytes(data)
        partial.replace(target)
    return paths


# --------------------------------------------------------------------------- WikiANN (#51)

#: Kazakh part of WikiANN on the Hugging Face Hub, at the commit the checksums refer to.
WIKIANN_REPO = "unimelb-nlp/wikiann"
WIKIANN_COMMIT = "f0a3be6dc5564c0cc4150bb660144800a1f539d4"
WIKIANN_URL = (
    "https://huggingface.co/datasets/{repo}/resolve/{commit}/kk/{split}-00000-of-00001.parquet"
)
#: SHA-256 of the parquet files as published (the Hub's LFS metadata), keyed by split.
WIKIANN_SHA256: dict[str, str] = {
    "train": "0360ddaebb431285a74241db0f00a3ed0c7605ef290b5669f340a334408ceecd",
    "validation": "e7f860c1c616942e17879bf536827d4960b3f09d00f9ad0e833d4d184c981c19",
    "test": "176f3870d82b91e52c813b35744850fc041cead3769e976210a42d9d846cc3e5",
}
#: Class names of WikiANN's ``ner_tags``, in id order.
WIKIANN_LABELS = ("O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC")


def write_iob2(path: str | Path, tokens: Sentences, labels: Sentences) -> Path:
    """Write sentences as ``<token> <label>`` lines with a blank line after each (LF)."""
    lines: list[str] = []
    for i, (words, tags) in enumerate(zip(tokens, labels, strict=True)):
        if not words or len(words) != len(tags):
            raise IOB2FormatError(f"sentence {i}: {len(words)} tokens and {len(tags)} labels")
        for word, tag in zip(words, tags, strict=True):
            if not word or any(ch.isspace() for ch in word):
                raise IOB2FormatError(f"sentence {i}: token {word!r} is empty or has whitespace")
            lines.append(f"{word} {tag}\n")
        lines.append("\n")
    path = Path(path)
    path.write_text("".join(lines), encoding="utf-8", newline="\n")
    return path


def wikiann_to_iob2(parquet: bytes) -> tuple[Sentences, Sentences]:
    """Tokens and IOB2 labels of a WikiANN parquet file.

    The label names are checked against the class names stored in the file's Hugging Face
    metadata, so a changed label order cannot pass silently.
    """
    import io
    import json

    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(parquet))
    try:
        features = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]
        names = tuple(features["ner_tags"]["feature"]["names"])
    except (TypeError, KeyError, ValueError) as exc:
        raise ValueError(f"no ner_tags class names in the parquet metadata ({exc!r})") from None
    if names != WIKIANN_LABELS:
        raise ValueError(f"unexpected ner_tags class names {names}, expected {WIKIANN_LABELS}")
    tokens = [list(words) for words in table.column("tokens").to_pylist()]
    labels = [[names[i] for i in tags] for tags in table.column("ner_tags").to_pylist()]
    return tokens, labels


def download_wikiann_kk(
    dest_dir: str | Path,
    *,
    force: bool = False,
    files: Mapping[str, str] = WIKIANN_SHA256,
    commit: str = WIKIANN_COMMIT,
    fetch: Callable[[str], bytes] = _fetch_url,
) -> dict[str, object]:
    """Download WikiANN ``kk``, verify the parquet files and write IOB2 splits.

    The parquet files are kept in ``dest_dir/parquet``. Valid files are not downloaded again
    unless ``force`` is set. The splits get KazNERD's file names
    (:data:`KAZNERD_SPLITS`), and ``manifest.json`` records the source and checksums.
    """
    import json

    dest_dir = Path(dest_dir)
    raw_dir = dest_dir / "parquet"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "source": f"https://huggingface.co/datasets/{WIKIANN_REPO}",
        "config": "kk",
        "commit": commit,
        "licence": "unknown on the dataset card; the text comes from Wikipedia (CC BY-SA)",
        "labels": list(WIKIANN_LABELS),
    }
    splits: dict[str, dict[str, object]] = {}
    for split, expected in files.items():
        raw = raw_dir / f"{split}.parquet"
        if force or not raw.is_file() or _sha256(raw.read_bytes()) != expected:
            data = fetch(WIKIANN_URL.format(repo=WIKIANN_REPO, commit=commit, split=split))
            actual = _sha256(data)
            if actual != expected:
                raise ValueError(f"{split}: SHA-256 mismatch (expected {expected}, got {actual})")
            partial = raw.with_name(raw.name + ".part")
            partial.write_bytes(data)
            partial.replace(raw)
        data = raw.read_bytes()
        tokens, labels = wikiann_to_iob2(data)
        target = write_iob2(dest_dir / KAZNERD_SPLITS[split], tokens, labels)
        entities: dict[str, int] = {}
        for tag in (tag for sentence in labels for tag in sentence if tag.startswith("B-")):
            entities[tag[2:]] = entities.get(tag[2:], 0) + 1
        splits[split] = {
            "parquet": raw.name,
            "parquet_bytes": len(data),
            "parquet_sha256": _sha256(data),
            "iob2": target.name,
            "iob2_sha256": _sha256(target.read_bytes()),
            "sentences": len(tokens),
            "tokens": sum(len(s) for s in tokens),
            "entities": dict(sorted(entities.items())),
        }
    manifest["splits"] = splits
    text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    (dest_dir / "manifest.json").write_text(text, encoding="utf-8", newline="\n")
    return manifest
