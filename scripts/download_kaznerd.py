"""Download the KazNERD IOB2 splits (IS2AI, CC BY 4.0) and verify their checksums.

Usage:
    python scripts/download_kaznerd.py [--dest DIR] [--force]

The default destination is ``data/kaznerd`` under the repository root, independent of the
current working directory. ``data/`` is git-ignored: the dataset is never committed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kazner.data import KAZNERD_COMMIT, download_kaznerd

DEFAULT_DEST = Path(__file__).resolve().parent.parent / "data" / "kaznerd"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="target directory")
    parser.add_argument("--force", action="store_true", help="download even if files are valid")
    args = parser.parse_args()

    print(f"KazNERD @ IS2AI/KazNERD {KAZNERD_COMMIT[:7]} -> {args.dest}")
    for path in download_kaznerd(args.dest, force=args.force):
        print(f"  ok  {path.name}")


if __name__ == "__main__":
    main()
