"""Unit tests for RAGTreePipeline wiring."""

import json

import pytest

from ragtree.core.errors import RagtreeError
from ragtree.core.pipeline import RAGTreePipeline
from ragtree.core.schemas import Chunk, RAGTask
from ragtree.integrations.exporters import JsonExporter
from ragtree.integrations.llms import MockLLMProvider
from ragtree.integrations.vectorstores import InMemoryVectorStore
from ragtree.retrieval import DenseRetriever
from ragtree.tasks import QuestionAnsweringTask
from tests.contract.fakes import ExactMatchEvaluator

CHUNKS = [
    Chunk(id="c1", document_id="d1", text="The pump failed because the seal wore out."),
    Chunk(id="c2", document_id="d2", text="Routine maintenance was performed in June."),
]


def _retriever():
    store = InMemoryVectorStore()
    store.add_chunks(list(CHUNKS))
    return DenseRetriever(store, top_k=1)


def test_run_wires_retrieve_generate_evaluate_export(tmp_path):
    out_file = tmp_path / "result.json"
    pipeline = RAGTreePipeline(
        retriever=_retriever(),
        generator=MockLLMProvider(reply="seal wear [d1/c1]"),
        evaluator=ExactMatchEvaluator(),
        exporter=JsonExporter(),
    )
    task = QuestionAnsweringTask("Why did the pump fail?")
    result = pipeline.run(task, reference="seal wear [d1/c1]", output_path=str(out_file))

    assert result.task_type == "question_answering"
    assert result.output == "seal wear [d1/c1]"
    assert result.evidence and result.evidence[0].document_id == "d1"
    assert result.metadata["raw_output"] == "seal wear [d1/c1]"
    assert result.metrics["exact_match"] == 1.0
    assert result.artifacts["evaluation"]["method"] == "exact_match"
    assert result.artifacts["exported_to"] == str(out_file)
    exported = json.loads(out_file.read_text(encoding="utf-8"))
    assert exported["task_type"] == "question_answering"


def test_run_accepts_plain_ragtask_without_task_object_helpers():
    pipeline = RAGTreePipeline(generator=MockLLMProvider())
    task = RAGTask(task_type="generic", query="hello there")
    result = pipeline.run(task)
    assert result.output.startswith("echo:")
    assert "hello there" in result.output
    assert result.evidence == []


def test_evidence_reaches_the_prompt():
    provider = MockLLMProvider(reply="ok")
    pipeline = RAGTreePipeline(retriever=_retriever(), generator=provider)
    pipeline.run(QuestionAnsweringTask("Why did the pump fail?"))
    prompt = provider.calls[0][1]["content"]
    assert "seal wore out" in prompt
    assert "[d1/c1]" in prompt


def test_run_without_generator_raises():
    pipeline = RAGTreePipeline()
    with pytest.raises(RagtreeError):
        pipeline.run(QuestionAnsweringTask("anything"))


def test_run_rejects_non_task_objects():
    pipeline = RAGTreePipeline(generator=MockLLMProvider())
    with pytest.raises(RagtreeError):
        pipeline.run("just a string")
