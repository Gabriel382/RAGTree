"""End-to-end semantic RAG over the tiny QA corpus — no extras, no network.

This is the design doc's section 17.3 example with the mock stack: the same
pipeline runs unchanged with LiteLLM + Chroma/Qdrant once extras are
installed.
"""

import json
from pathlib import Path

from ragtree.core.pipeline import RAGTreePipeline
from ragtree.core.schemas import Chunk
from ragtree.integrations.exporters import JsonlExporter
from ragtree.integrations.llms import MockLLMProvider
from ragtree.integrations.vectorstores import InMemoryVectorStore
from ragtree.retrieval import DenseRetriever
from ragtree.tasks import ClaimVerificationTask, QuestionAnsweringTask, SummarizationTask

FIXTURES = Path(__file__).parents[1] / "fixtures" / "qa" / "tiny_documents.jsonl"


def _chunks() -> list[Chunk]:
    chunks = []
    for line in FIXTURES.read_text(encoding="utf-8").splitlines():
        doc = json.loads(line)
        chunks.append(
            Chunk(id=f"{doc['id']}-c0", document_id=doc["id"], text=doc["text"])
        )
    return chunks


def _retriever(top_k: int = 3) -> DenseRetriever:
    store = InMemoryVectorStore()
    store.add_chunks(_chunks())
    return DenseRetriever(store, top_k=top_k)


def test_question_answering_demo(tmp_path):
    out_path = tmp_path / "qa_results.jsonl"
    pipeline = RAGTreePipeline(
        retriever=_retriever(),
        generator=MockLLMProvider(
            reply="The seal wore out after dry running [maint-001/maint-001-c0]."
        ),
        exporter=JsonlExporter(),
    )
    result = pipeline.run(
        QuestionAnsweringTask("Why did pump P-102 fail?"), output_path=str(out_path)
    )

    retrieved_docs = {span.document_id for span in result.evidence}
    assert "maint-001" in retrieved_docs, "failure report must be retrieved"
    assert result.evidence[0].document_id == "maint-001", "failure report should rank first"
    assert "maint-001" in result.output

    exported = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert len(exported) == 1
    assert exported[0]["task_type"] == "question_answering"
    assert exported[0]["evidence"]


def test_claim_verification_demo():
    pipeline = RAGTreePipeline(
        retriever=_retriever(),
        generator=MockLLMProvider(
            reply="LABEL: SUPPORTS\nRATIONALE: [maint-001/maint-001-c0] states the seal wore out."
        ),
    )
    result = pipeline.run(
        ClaimVerificationTask("Pump P-102 failed because its mechanical seal wore out.")
    )
    assert result.output["label"] == "SUPPORTS"
    assert "maint-001" in result.output["rationale"]


def test_summarization_demo():
    provider = MockLLMProvider(
        reply="Pump P-102 failed due to seal wear [maint-001/maint-001-c0]."
    )
    pipeline = RAGTreePipeline(retriever=_retriever(top_k=4), generator=provider)
    result = pipeline.run(SummarizationTask(focus="pump P-102 seal failure"))
    assert isinstance(result.output, str) and result.output
    prompt = provider.calls[0][1]["content"]
    assert "seal wore out" in prompt, "summary prompt must carry retrieved evidence"
