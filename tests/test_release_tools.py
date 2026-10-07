import importlib.util
from pathlib import Path

import pytest
from packaging.version import Version

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "release_tools", ROOT / "scripts" / "release_tools.py"
)
release_tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_tools)

CHANGELOG = """# Changelog

## [Unreleased]

### Added
- New thing.

## [0.1.0] - 2026-10-08

### Added
- First release.

## [0.0.9] - 2026-01-01
- Old.

[Unreleased]: https://github.com/o/r/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/o/r/releases/tag/v0.1.0
"""


@pytest.mark.parametrize("tag", ["v0.1.0rc1", "v0.1.0-rc.1", "v0.1.0.rc1"])
def test_pep440_normalisation_matches_rc_tags(tag):
    assert release_tools.check_tag(tag, Version("0.1.0rc1")).is_prerelease


def test_final_tag_is_not_a_prerelease():
    assert not release_tools.check_tag("v0.1.0", Version("0.1.0")).is_prerelease


@pytest.mark.parametrize(
    ("tag", "message"),
    [
        ("v0.1.1", "pyproject.toml says 0.1.0"),
        ("0.1.0", "start with 'v'"),
        ("vbanana", "not a PEP 440 version"),
    ],
)
def test_bad_tags_are_rejected(tag, message):
    with pytest.raises(ValueError, match=message):
        release_tools.check_tag(tag, Version("0.1.0"))


def test_release_notes_come_from_the_version_section():
    assert release_tools.changelog_section(CHANGELOG, Version("0.1.0")) == (
        "### Added\n- First release."
    )
    assert release_tools.changelog_section(CHANGELOG, Version("0.0.9")) == "- Old."


def test_prerelease_falls_back_to_unreleased():
    notes = release_tools.changelog_section(CHANGELOG, Version("0.2.0rc1"))
    assert notes == "### Added\n- New thing."


def test_missing_section_is_an_error():
    with pytest.raises(ValueError, match="no section for 0.3.0"):
        release_tools.changelog_section(CHANGELOG, Version("0.3.0"))


def test_repository_version_and_changelog_are_consistent():
    version = release_tools.project_version()
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert release_tools.changelog_section(changelog, version)


def test_main_check_and_notes(tmp_path, capsys):
    tag = f"v{release_tools.project_version()}"
    assert release_tools.main(["check", tag]) == 0
    assert capsys.readouterr().out.strip() in ("prerelease=true", "prerelease=false")
    out = tmp_path / "notes.md"
    assert release_tools.main(["notes", tag, str(out)]) == 0
    assert out.read_text(encoding="utf-8").strip()
