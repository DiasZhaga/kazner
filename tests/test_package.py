from importlib.metadata import version

import pytest

import kazner
from kazner.cli import main


def test_version_matches_installed_metadata():
    assert kazner.__version__ == version("kazner")


def test_cli_version_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == f"kazner {kazner.__version__}"


def test_cli_without_command_prints_help(capsys):
    assert main([]) == 0
    assert "usage: kazner" in capsys.readouterr().out
