"""Command-line interface: ``kazner <command>``.

Subcommands (``fertility``, ``evaluate``, ...) are added by later issues.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from kazner import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kazner",
        description="Kazakh NER toolkit: data, evaluation and tokenizer analysis.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_subparsers(dest="command", metavar="<command>")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
    return 0
