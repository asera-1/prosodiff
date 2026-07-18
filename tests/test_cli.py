from __future__ import annotations

from typer.testing import CliRunner

from prosodiff import __version__
from prosodiff.cli import app


runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"prosodiff {__version__}"


def test_compare_requires_at_least_two_files() -> None:
    result = runner.invoke(app, ["compare", "one.wav"])
    assert result.exit_code == 2
    assert "Expected 2 to 4" in result.output


def test_ui_help_does_not_start_server() -> None:
    result = runner.invoke(app, ["ui", "--help"])
    assert result.exit_code == 0
    assert "local browser interface" in result.output


def test_voicing_threshold_help_is_diagnostic_only() -> None:
    result = runner.invoke(app, ["compare", "--help"])
    assert result.exit_code == 0
    assert "threshold used only to" in result.output
    assert "emphasize contour" in result.output
