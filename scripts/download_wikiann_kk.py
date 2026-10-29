"""Download the Kazakh part of WikiANN (Hugging Face Hub, pinned commit) and write IOB2 splits.

Usage:
    python scripts/download_wikiann_kk.py [--dest DIR] [--force]

The three parquet files (about 218 kB in total) are verified against their SHA-256 and
converted to ``IOB2_train.txt``, ``IOB2_valid.txt`` and ``IOB2_test.txt`` with labels O, B/I-PER,
B/I-ORG and B/I-LOC; ``manifest.json`` records the source and the checksums. The default
destination is ``data/wikiann_kk`` under the repository root. ``data/`` is git-ignored: the
dataset is never committed. Its licence is not stated on the dataset card; the text comes
from Wikipedia (CC BY-SA).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kazner.data import WIKIANN_COMMIT, WIKIANN_REPO, download_wikiann_kk

DEFAULT_DEST = Path(__file__).resolve().parent.parent / "data" / "wikiann_kk"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="target directory")
    parser.add_argument("--force", action="store_true", help="download even if files are valid")
    args = parser.parse_args()

    print(f"WikiANN kk @ {WIKIANN_REPO} {WIKIANN_COMMIT[:7]} -> {args.dest}")
    manifest = download_wikiann_kk(args.dest, force=args.force)
    for info in manifest["splits"].values():
        print(f"  ok  {info['iob2']}: {info['sentences']} sentences, entities {info['entities']}")


if __name__ == "__main__":
    main()
