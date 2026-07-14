"""Unit tests for the task layer."""

import json

from ragtree.core.schemas import Document, EvidenceSpan
from ragtree.tasks import (
    ClaimVerificationTask,
    QuestionAnsweringTask,
    RelationExtractionTask,
    SummarizationTask,
)
from ragtree.tasks.relation_extraction import format_entities

EVIDENCE = [
    EvidenceSpan(document_id="d1", chunk_id="c1", text="The seal wore out."),
    EvidenceSpan(document_id="d2", text="Maintenance was done in June."),
]


def test_qa_task_messages_contain_question_and_evidence():
    task = QuestionAnsweringTask("Why did the pump fail?")
    messages = task.build_messages(EVIDENCE)
    assert messages[0]["role"] == "system"
    user = messages[1]["content"]
    assert "Why did the pump fail?" in user
    assert "[d1/c1] The seal wore out." in user
    assert "[d2] Maintenance was done in June." in user


def test_qa_task_to_ragtask_roundtrip():
    task = QuestionAnsweringTask("Q?", metadata={"run": 1})
    rag_task = task.to_ragtask()
    assert rag_task.task_type == "question_answering"
    assert rag_task.query == "Q?"
    assert rag_task.constraints["require_evidence"] is True
    assert rag_task.metadata == {"run": 1}


def test_summarization_task_instructions():
    task = SummarizationTask(focus="failures", max_sentences=3)
    text = task.instructions()
    assert "3 sentences" in text
    assert "failures" in text
    assert task.parse_output("  a summary \n") == "a summary"


def test_claim_verification_parses_labels():
    task = ClaimVerificationTask("The pump failed because of the seal.")
    parsed = task.parse_output("LABEL: SUPPORTS\nRATIONALE: [d1/c1] shows seal wear.")
    assert parsed == {"label": "SUPPORTS", "rationale": "[d1/c1] shows seal wear."}
    assert task.parse_output("nonsense")["label"] == "NOT_ENOUGH_INFO"
    assert task.parse_output("It REFUTES the claim")["label"] == "REFUTES"


def test_relation_task_prompt_includes_doc_entities_and_schema():
    doc = {
        "document_id": "doc-1",
        "title": "T",
        "text": "A causes B.",
        "entities": {"E1": {"type": "EVENT", "mentions": [{"trigger_word": "A"}]}},
    }
    task = RelationExtractionTask(["CAUSES", "PRECONDITION"], document=doc)
    prompt = task.instructions()
    assert "doc-1" in prompt
    assert "A causes B." in prompt
    assert '- E1: "A" [EVENT]' in prompt
    assert "- CAUSES" in prompt and "- PRECONDITION" in prompt
    assert task.document_id == "doc-1"


def test_relation_task_accepts_core_document():
    doc = Document(id="d9", text="X because Y.")
    task = RelationExtractionTask(["BECAUSE"], document=doc, entities=[{"id": "E1", "text": "X"}])
    assert task.document_id == "d9"
    assert '- E1: "X"' in task.instructions()


def test_relation_task_parses_and_normalizes_output():
    task = RelationExtractionTask(["CAUSES", "PRECONDITION"])
    raw = 'Here you go:\n```json\n{"CAUSES": [["E1", "E2"]], "EXTRA": [["E9", "E8"]]}\n```'
    assert task.parse_output(raw) == {"CAUSES": [["E1", "E2"]], "PRECONDITION": []}
    assert task.parse_output("no json at all") == {"CAUSES": [], "PRECONDITION": []}


def test_relation_task_make_prediction_keeps_legacy_format():
    doc = {"document_id": "doc-7", "text": "t", "entities": {}}
    task = RelationExtractionTask(["CAUSES"], document=doc)
    relations = task.parse_output(json.dumps({"CAUSES": [["E1", "E2"]]}))
    prediction = task.make_prediction(relations, method="unit", model="mock")
    assert prediction.document_id == "doc-7"
    assert prediction.relations == {"CAUSES": [["E1", "E2"]]}


def test_format_entities_handles_empty_and_list_forms():
    assert format_entities(None) == "(no entities provided)"
    assert format_entities([{"id": "E1"}]) == '- E1: "E1"'
