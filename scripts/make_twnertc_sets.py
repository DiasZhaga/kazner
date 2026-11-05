"""Write the Turkish training subset and a held-out set from the pilot's TWNERTC file (#59).

Usage:
    python scripts/make_twnertc_sets.py --source <coarse_train.txt> [--dest DIR]

``--source`` is the pilot's ``data/coarse_train.txt``: TWNERTC's coarse-grained,
domain-dependent noise-reduced version in IOB2, written by the pilot's
``src/prepare_twnertc_coarse.py``. The pilot trained its Turkish model on 10% of it (seed 42);
``train_subset.txt`` repeats that subsample exactly. ``heldout.txt`` holds 5,000 other
sentences, none equal to a training sentence (:func:`kazner.data.heldout_indices`). The pilot's
other two files are not usable as held-out sets: they are other versions of the same corpus.
``manifest.json`` records the settings and SHA-256 checksums. The default destination is
``data/twnertc`` under the repository root; ``data/`` is git-ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from kazner.data import heldout_indices, read_iob2, write_iob2

DEFAULT_DEST = Path(__file__).resolve().parent.parent / "data" / "twnertc"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="the pilot's coarse_train.txt")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="target directory")
    parser.add_argument("--fraction", type=float, default=0.1, help="pilot's Turkish fraction")
    parser.add_argument("--seed", type=int, default=42, help="pilot's subsample seed")
    parser.add_argument("--heldout", type=int, default=5000, help="held-out sentences")
    parser.add_argument("--heldout-seed", type=int, default=42, help="held-out sample seed")
    args = parser.parse_args()

    tokens, labels = read_iob2(args.source)
    train, heldout = heldout_indices(
        tokens, args.fraction, args.seed, args.heldout, args.heldout_seed
    )
    args.dest.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, indices in (("train_subset.txt", train), ("heldout.txt", heldout)):
        path = write_iob2(
            args.dest / name, [tokens[i] for i in indices], [labels[i] for i in indices]
        )
        files[name] = {"sentences": len(indices), "sha256": sha256(path)}
        print(f"  ok  {path}: {len(indices)} sentences")
    manifest = {
        "source": {
            "file": args.source.name,
            "sentences": len(tokens),
            "sha256": sha256(args.source),
        },
        "fraction": args.fraction,
        "seed": args.seed,
        "heldout": args.heldout,
        "heldout_seed": args.heldout_seed,
        "files": files,
    }
    text = json.dumps(manifest, indent=2) + "\n"
    (args.dest / "manifest.json").write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
