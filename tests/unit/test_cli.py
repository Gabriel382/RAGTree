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


# ---------------------------------------------------------------------------
# Sprint-3 commands
# ---------------------------------------------------------------------------

import json
from pathlib import Path

from ragtree.core.schemas import RAGResult

REPO_ROOT = Path(__file__).parents[2]


def test_demo_commands_run_offline():
    for which in ("semantic-rag", "relation-extraction"):
        result = runner.invoke(app, ["demo", which])
        assert result.exit_code == 0, result.output
    assert runner.invoke(app, ["demo", "nope"]).exit_code == 1


def test_run_command_executes_example_config(tmp_path):
    config = REPO_ROOT / "examples" / "configs" / "semantic_rag_demo.yaml"
    result = runner.invoke(
        app, ["run", "--config", str(config), "--output", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "results.jsonl").is_file()
    assert (tmp_path / "manifest.json").is_file()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["versions"]["ragtree"]
    assert manifest["finished_at"]


def test_export_command_roundtrip(tmp_path):
    source = tmp_path / "result.json"
    result_obj = RAGResult(
        task_type="relation_extraction",
        output={"CAUSES": [["E1", "E2"]]},
        metadata={"document_id": "d1"},
    )
    source.write_text(result_obj.model_dump_json(), encoding="utf-8")

    edges = tmp_path / "graph.csv"
    outcome = runner.invoke(
        app,
        ["export", "--input", str(source), "--format", "graph-csv", "--output", str(edges)],
    )
    assert outcome.exit_code == 0, outcome.output
    assert edges.is_file()
    assert (tmp_path / "graph_nodes.csv").is_file()
    assert "E1" in edges.read_text(encoding="utf-8")

    assert (
        runner.invoke(
            app,
            ["export", "--input", str(source), "--format", "nope", "--output", str(edges)],
        ).exit_code
        == 1
    )


def test_serve_and_workbench_are_registered():
    assert runner.invoke(app, ["serve", "--help"]).exit_code == 0
    assert runner.invoke(app, ["workbench", "--help"]).exit_code == 0
