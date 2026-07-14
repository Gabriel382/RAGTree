"""Smoke tests for the lightweight CLI (no extras required)."""

from typer.testing import CliRunner

from ragtree.cli.main import app

runner = CliRunner()


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "RAGTree" in result.output


def test_doctor_command():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0


def test_addons_command_lists_extras():
    result = runner.invoke(app, ["addons"])
    assert result.exit_code == 0
