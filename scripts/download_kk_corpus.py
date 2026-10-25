"""Download a seeded sample of Kazakh web text (FineWeb-2, ODC-By 1.0) for continued pre-training.

Usage:
    python scripts/download_kk_corpus.py [--words N] [--seed S] [--out FILE] [--kaznerd DIR]
                                         [--local DIR]

Row groups of the pinned FineWeb-2 ``kaz_Cyrl`` train files are read in an order fixed by
``--seed`` until the cleaned text reaches ``--words`` words, so only the row groups needed
are downloaded (rules: ``kazner.corpus``). Lines containing a KazNERD sentence are removed
(``--kaznerd``, default ``data/kaznerd``). Writes the text, one paragraph per line (UTF-8,
LF), and ``<out stem>.manifest.json`` with the source, revision, rules, counts and SHA-256.
``--local DIR`` reads ``*.parquet`` files from a directory instead of the Hub. ``data/`` is
git-ignored: the corpus is never committed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from kazner.config import data_fingerprint
from kazner.corpus import CountingReader, SentenceIndex, build_corpus
from kazner.data import read_splits

ROOT = Path(__file__).resolve().parent.parent
REPO = "HuggingFaceFW/fineweb-2"
CONFIG = "kaz_Cyrl"
#: Pinned commit of the dataset repository (last modified 2025-10-27).
REVISION = "af9c13333eb981300149d5ca60a8e9d659b276b9"
#: Train split only: the dataset card says the small ``test`` split should not be trained on.
FILES = ("data/kaz_Cyrl/train/000_00000.parquet", "data/kaz_Cyrl/train/001_00000.parquet")
DEFAULT_OUT = ROOT / "data" / "kk_corpus" / "fineweb2_kaz_Cyrl.txt"


def open_files(
    local: Path | None, revision: str
) -> tuple[list[Any], list[str], list[CountingReader]]:
    """Parquet files to sample from, their names, and the readers counting bytes read.

    Hub files are opened with ``cache_type="none"``: fsspec would otherwise read
    ahead in 5 MiB blocks and fetch several times the bytes the parquet reader
    asks for (#42).
    """
    import pyarrow.parquet as pq

    if local is not None:
        paths = sorted(local.glob("*.parquet"))
        if not paths:
            raise SystemExit(f"no .parquet files in {local}")
        readers = [CountingReader(path.open("rb")) for path in paths]
        return [pq.ParquetFile(r) for r in readers], [p.name for p in paths], readers
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    readers = [
        CountingReader(
            fs.open(f"datasets/{REPO}@{revision}/{path}", "rb", cache_type="none")
        )
        for path in FILES
    ]
    return [pq.ParquetFile(r) for r in readers], list(FILES), readers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--words", type=int, default=20_000_000, help="target number of words")
    parser.add_argument("--seed", type=int, default=42, help="seed of the row-group order")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output text file")
    parser.add_argument(
        "--kaznerd", type=Path, default=ROOT / "data" / "kaznerd", help="KazNERD IOB2 directory"
    )
    parser.add_argument("--revision", default=REVISION, help="dataset commit (default: pinned)")
    parser.add_argument("--local", type=Path, help="read *.parquet from here instead of the Hub")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="read only the parquet metadata (row groups, sizes) and stop",
    )
    args = parser.parse_args(argv)

    files, names, readers = open_files(args.local, args.revision)
    source = "local " + str(args.local) if args.local else f"{REPO}@{args.revision[:7]}"
    print(
        f"{source}: {sum(f.num_row_groups for f in files)} row groups; target {args.words:,} words"
    )
    if args.dry_run:
        for name, f in zip(names, files, strict=True):
            meta = f.metadata
            chunks = [
                meta.row_group(i).column(c)
                for i in range(meta.num_row_groups)
                for c in range(meta.num_columns)
            ]
            text = sum(c.total_compressed_size for c in chunks if c.path_in_schema == "text")
            read_mib = sum(r.bytes_read for r in readers) / 2**20
            print(
                f"  {name}: {meta.num_rows:,} documents, {meta.num_row_groups} row groups, "
                f"{text / max(1, meta.num_row_groups) / 2**20:.2f} MiB of compressed text "
                f"per row group (the only column read); {read_mib:.1f} MiB read"
            )
        return 0

    splits = read_splits(args.kaznerd)
    index = SentenceIndex(sentence for tokens, _ in splits.values() for sentence in tokens)

    start = time.perf_counter()

    def progress(groups: int, words: int) -> None:
        read_mib = sum(r.bytes_read for r in readers) / 2**20
        print(
            f"  row group {groups}: {words:,} words, {read_mib:.1f} MiB read, "
            f"{time.perf_counter() - start:.0f}s",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    part = args.out.with_name(args.out.name + ".part")
    with part.open("wb") as out:
        sample = build_corpus(
            files, names, index, out, target_words=args.words, seed=args.seed, progress=progress
        )
    part.replace(args.out)

    manifest = {
        "output": args.out.name,
        "source": {
            "repo": None if args.local else REPO,
            "config": CONFIG,
            "revision": None if args.local else args.revision,
            "local_dir": str(args.local) if args.local else None,
            "license": "ODC-By 1.0 (FineWeb-2)",
        },
        "kaznerd": {"dir": str(args.kaznerd), "fingerprint": data_fingerprint(args.kaznerd)},
        **sample,
        "bytes_read": sum(r.bytes_read for r in readers),
        "seconds": round(time.perf_counter() - start, 1),
    }
    manifest_path = args.out.with_name(args.out.stem + ".manifest.json")
    text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    manifest_path.write_text(text, encoding="utf-8", newline="\n")
    counts = sample["line_counts"]
    print(
        f"{sample['words']:,} words in {sample['lines']:,} lines from "
        f"{len(sample['row_groups_used'])} row groups -> {args.out}\n"
        f"lines {counts['lines']:,}: "
        + ", ".join(f"{k} {v:,}" for k, v in counts.items() if k != "lines")
    )
    if not sample["complete"]:
        print(f"WARNING: the source ran out before {args.words:,} words")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
