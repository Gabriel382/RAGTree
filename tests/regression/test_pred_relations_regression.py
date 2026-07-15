"""Regression: the historical pred_relations output format keeps evaluating
identically (compatibility contract, design doc section 11.1).

The fixture predictions file uses the exact legacy shape produced by the
benchmark scripts: one JSON object per document with ``document_id``,
``type`` and ``pred_relations``. Golden values were computed once and are
asserted exactly; any drift in metrics, IO or normalization fails here.
"""

import json
from pathlib import Path

import pytest

from ragtree.core.schemas import RAGResult
from ragtree.evaluation.relation_evaluator import RelationEvaluator
from ragtree.evaluation.relations.runner import evaluate_relations

FIXTURES = Path(__file__).parents[1] / "fixtures" / "regression"
GOLD = FIXTURES / "gold_documents.jsonl"
PRED = FIXTURES / "predictions_legacy_format.jsonl"

GOLDEN_FULL = {"precision": 0.9696969696969697, "recall": 0.5245901639344263, "f1": 0.6808510638297871}
GOLDEN_FULL_COUNTS = {"tp": 32, "fp": 1, "fn": 29, "num_docs_eval": 5, "num_docs_missing_gold": 0}
GOLDEN_NO_NULL = {"precision": 0.96875, "recall": 0.5166666666666667, "f1": 0.673913043478261}
GOLDEN_NO_NULL_COUNTS = {"tp": 31, "fp": 1, "fn": 29}


def test_legacy_runner_reproduces_golden_metrics():
    metrics = evaluate_relations(gold_path=GOLD, pred_path=PRED)
    for key, value in GOLDEN_FULL.items():
        assert metrics["micro"][key] == pytest.approx(value, abs=1e-12), key
    for key, value in GOLDEN_FULL_COUNTS.items():
        assert metrics["counts"][key] == value, key


def test_legacy_runner_with_ignored_null_label():
    metrics = evaluate_relations(gold_path=GOLD, pred_path=PRED, ignore_labels=["null"])
    for key, value in GOLDEN_NO_NULL.items():
        assert metrics["micro"][key] == pytest.approx(value, abs=1e-12), key
    for key, value in GOLDEN_NO_NULL_COUNTS.items():
        assert metrics["counts"][key] == value, key


def test_doc_type_filter_still_supported():
    assert evaluate_relations(gold_path=GOLD, pred_path=PRED, doc_type_filter="test")[
        "counts"
    ]["num_docs_eval"] == 5
    assert evaluate_relations(gold_path=GOLD, pred_path=PRED, doc_type_filter="train")[
        "counts"
    ]["num_docs_eval"] == 0


def test_new_evaluator_agrees_with_legacy_runner():
    """Old outputs evaluate identically through the protocol-level evaluator."""
    gold_by_id = {
        json.loads(line)["document_id"]: json.loads(line)["relations"]
        for line in GOLD.read_text(encoding="utf-8").splitlines()
    }
    evaluator = RelationEvaluator()
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for line in PRED.read_text(encoding="utf-8").splitlines():
        doc = json.loads(line)
        report = evaluator.evaluate(
            RAGResult(task_type="relation_extraction", output=doc["pred_relations"]),
            reference=gold_by_id[doc["document_id"]],
        )
        for key in totals:
            totals[key] += report.counts[key]
    assert totals == {k: GOLDEN_FULL_COUNTS[k] for k in ("tp", "fp", "fn")}


def test_cli_evaluate_command_on_legacy_outputs(tmp_path):
    from typer.testing import CliRunner

    from ragtree.cli.main import app

    out = tmp_path / "metrics.json"
    result = CliRunner().invoke(
        app,
        ["evaluate", "--gold", str(GOLD), "--pred", str(PRED), "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "0.6809" in result.output or "0.680851" in result.output
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["counts"]["tp"] == GOLDEN_FULL_COUNTS["tp"]
