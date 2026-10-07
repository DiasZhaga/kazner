"""Convert a pilot ``test_predictions*.csv`` into the kazner prediction contract.

Usage:
    python scripts/convert_pilot_predictions.py PILOT_CSV OUT_CSV [--gold IOB2_test.txt]

With ``--gold`` the words the pilot model never saw (truncated at ``max_len``) are appended
with ``pred = O`` and ``truncated = 1``, giving a complete contract file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kazner.contract import convert_pilot_predictions, write_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pilot_csv", type=Path)
    parser.add_argument("out_csv", type=Path)
    parser.add_argument("--gold", type=Path, help="IOB2 test split used by the pilot run")
    args = parser.parse_args()
    predictions = convert_pilot_predictions(args.pilot_csv, gold_path=args.gold)
    write_predictions(args.out_csv, predictions)
    print(
        f"{args.out_csv}: {len(predictions.words)} sentences, {predictions.n_words} words, "
        f"{predictions.n_truncated_words} truncated"
    )


if __name__ == "__main__":
    main()
