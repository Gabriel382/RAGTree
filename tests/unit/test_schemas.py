"""Unit tests for the core schemas (design doc, section 6.1)."""

import pytest
from pydantic import ValidationError

import ragtree
from ragtree.core.schemas import (
    Chunk,
    Document,
    EvaluationResult,
    EvidenceSpan,
    RAGResult,
    RAGTask,
    RelationPrediction,
    RunManifest,
)


def test_document_minimal():
    doc = Document(id="d1", text="hello world")
    assert doc.title is None
    assert doc.source is None
    assert doc.metadata == {}


def test_document_requires_id_and_text():
    with pytest.raises(ValidationError):
        Document(id="d1")  # type: ignore[call-arg]


def test_chunk_roundtrip():
    chunk = Chunk(id="c1", document_id="d1", text="part", index=2, metadata={"page": 1})
    assert Chunk.model_validate(chunk.model_dump()) == chunk


def test_evidence_span_serializes_span():
    span = EvidenceSpan(document_id="d1", text="proof", span=(0, 5), score=0.9)
    restored = EvidenceSpan.model_validate_json(span.model_dump_json())
    assert restored.span == (0, 5)
    assert restored.score == pytest.approx(0.9)


def test_ragtask_accepts_output_schema():
    task = RAGTask(
        task_type="relation_extraction",
        output_schema={"CAUSES": "list of [head_id, tail_id] pairs"},
        constraints={"only_provided_entities": True},
    )
    assert task.query is None
    assert "CAUSES" in task.output_schema


def test_ragresult_default_lists_are_isolated():
    first = RAGResult(task_type="qa")
    second = RAGResult(task_type="qa")
    first.evidence.append(EvidenceSpan(document_id="d1", text="x"))
    assert second.evidence == []


def test_relation_prediction_keeps_legacy_relations_format():
    pred = RelationPrediction(
        document_id="doc0",
        relations={"CAUSES": [["EVENT_1", "EVENT_2"]], "PRECONDITION": []},
        method="baseline",
    )
    dumped = pred.model_dump()
    assert dumped["relations"] == {"CAUSES": [["EVENT_1", "EVENT_2"]], "PRECONDITION": []}


def test_run_manifest_timestamps_are_timezone_aware():
    manifest = RunManifest(run_id="run-001")
    assert manifest.started_at.tzinfo is not None
    assert manifest.finished_at is None


def test_evaluation_result_defaults():
    report = EvaluationResult(metrics={"f1": 0.5})
    assert report.counts == {}
    assert report.dataset is None


def test_package_exports_version_and_schemas():
    assert isinstance(ragtree.__version__, str) and ragtree.__version__
    assert ragtree.Document is Document
