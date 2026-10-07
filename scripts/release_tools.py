"""Release helpers used by ``.github/workflows/release.yml``.

Usage:
    python scripts/release_tools.py check TAG          # tag must match pyproject version
    python scripts/release_tools.py notes TAG OUT.md   # CHANGELOG section for the release

``check`` prints ``prerelease=true|false`` in the ``$GITHUB_OUTPUT`` format. Versions are
compared with ``packaging.version.Version`` (PEP 440), so ``v0.1.0rc1`` and ``v0.1.0-rc.1``
both match version ``0.1.0rc1``.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parent.parent


def project_version(pyproject: Path = ROOT / "pyproject.toml") -> Version:
    with pyproject.open("rb") as f:
        return Version(tomllib.load(f)["project"]["version"])


def check_tag(tag: str, version: Version) -> Version:
    """Return the tag's version if it equals ``version``; raise ``ValueError`` otherwise."""
    if not tag.startswith("v"):
        raise ValueError(f"release tags start with 'v', got {tag!r}")
    try:
        tagged = Version(tag[1:])
    except InvalidVersion:
        raise ValueError(f"tag {tag!r} is not a PEP 440 version") from None
    if tagged != version:
        raise ValueError(f"tag {tag} is version {tagged}, but pyproject.toml says {version}")
    return tagged


def changelog_section(changelog: str, version: Version) -> str:
    """Body of the ``## [<version>]`` section; pre-releases fall back to ``[Unreleased]``."""
    sections: dict[str, str] = {}
    for match in re.finditer(r"^## \[([^\]]+)\][^\n]*\n(.*?)(?=^## \[|\Z)", changelog, re.M | re.S):
        body = re.sub(r"^\[[^\]]+\]: \S+\s*$", "", match.group(2), flags=re.M)  # link refs
        sections[match.group(1)] = body.strip()
    for name, body in sections.items():
        try:
            if Version(name) == version:
                return body
        except InvalidVersion:
            continue
    if version.is_prerelease and sections.get("Unreleased"):
        return sections["Unreleased"]
    raise ValueError(f"CHANGELOG.md has no section for {version}")


def main(argv: list[str]) -> int:
    command, tag, *rest = argv
    version = check_tag(tag, project_version())
    if command == "check":
        print(f"prerelease={'true' if version.is_prerelease else 'false'}")
    elif command == "notes":
        notes = changelog_section((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), version)
        Path(rest[0]).write_text(notes + "\n", encoding="utf-8", newline="\n")
    else:
        raise SystemExit(f"unknown command {command!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
