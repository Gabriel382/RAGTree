"""Reusable protocol contract test bases.

Every adapter must pass the contract tests for the protocol it implements
(design doc, section 7.2). Adapters bind a fixture and inherit the checks:

    from tests.contract.bases import VectorStoreContractTests

    class TestChromaVectorStore(VectorStoreContractTests):
        @pytest.fixture
        def vector_store(self):
            return ChromaVectorStore(collection_name="contract-test")

Stores with a non-trivial query language (e.g. Neo4j) override the
``list_nodes_query`` / ``list_edges_query`` class attributes.
"""

from __future__ import annotations

import pytest

from ragtree.core.protocols import (
    Embedder,
    Evaluator,
    Exporter,
    GraphStore,
    LLMProvider,
    Retriever,
    VectorStore,
)
from ragtree.core.schemas import Chunk, EvaluationResult, EvidenceSpan, RAGResult, RAGTask

SAMPLE_CHUNKS = [
    Chunk(id="c1", document_id="d1", text="The pump failed because the seal wore out.", index=0),
    Chunk(id="c2", document_id="d1", text="Routine maintenance was performed in June.", index=1),
    Chunk(id="c3", document_id="d2", text="A pressure spike triggered the alarm.", index=0),
]


class LLMProviderContractTests:
    def test_satisfies_protocol(self, llm_provider):
        assert isinstance(llm_provider, LLMProvider)

    def test_complete_returns_nonempty_text(self, llm_provider):
        out = llm_provider.complete([{"role": "user", "content": "Say hello."}])
        assert isinstance(out, str)
        assert out.strip()


class EmbedderContractTests:
    def test_satisfies_protocol(self, embedder):
        assert isinstance(embedder, Embedder)

    def test_embed_returns_one_fixed_size_vector_per_text(self, embedder):
        vectors = embedder.embed(["pump seal wear", "pressure alarm"])
        assert len(vectors) == 2
        assert len({len(v) for v in vectors}) == 1
        assert all(isinstance(x, float) for vector in vectors for x in vector)


class VectorStoreContractTests:
    @pytest.fixture
    def loaded_store(self, vector_store):
        vector_store.add_chunks(list(SAMPLE_CHUNKS))
        return vector_store

    def test_satisfies_protocol(self, vector_store):
        assert isinstance(vector_store, VectorStore)

    def test_search_returns_evidence_spans(self, loaded_store):
        hits = loaded_store.search("why did the pump fail", top_k=2)
        assert 1 <= len(hits) <= 2
        assert all(isinstance(hit, EvidenceSpan) for hit in hits)

    def test_search_respects_top_k(self, loaded_store):
        assert len(loaded_store.search("maintenance", top_k=1)) == 1

    def test_search_finds_the_relevant_chunk(self, loaded_store):
        hits = loaded_store.search("pump seal failure", top_k=2)
        assert "c1" in {hit.chunk_id for hit in hits}

    def test_scores_sorted_descending_when_present(self, loaded_store):
        hits = loaded_store.search("alarm pressure spike", top_k=3)
        scores = [hit.score for hit in hits if hit.score is not None]
        assert scores == sorted(scores, reverse=True)


class RetrieverContractTests:
    def test_satisfies_protocol(self, retriever):
        assert isinstance(retriever, Retriever)

    def test_retrieve_returns_evidence(self, retriever):
        task = RAGTask(task_type="question_answering", query="why did the pump fail")
        evidence = retriever.retrieve(task)
        assert isinstance(evidence, list)
        assert evidence, "retriever fixture must be preloaded with searchable content"
        assert all(isinstance(item, EvidenceSpan) for item in evidence)


class GraphStoreContractTests:
    list_nodes_query = "nodes"
    list_edges_query = "edges"

    def test_satisfies_protocol(self, graph_store):
        assert isinstance(graph_store, GraphStore)

    def test_upsert_and_query_roundtrip(self, graph_store):
        graph_store.upsert_nodes(
            [{"id": "n1", "label": "Event"}, {"id": "n2", "label": "Event"}]
        )
        graph_store.upsert_edges([{"source": "n1", "target": "n2", "type": "CAUSES"}])
        nodes = graph_store.query(self.list_nodes_query)
        assert isinstance(nodes, list)
        assert {node["id"] for node in nodes} >= {"n1", "n2"}
        edges = graph_store.query(self.list_edges_query)
        assert any(edge.get("type") == "CAUSES" for edge in edges)

    def test_upsert_nodes_is_idempotent(self, graph_store):
        node = {"id": "n1", "label": "Event"}
        graph_store.upsert_nodes([node])
        graph_store.upsert_nodes([node])
        matching = [n for n in graph_store.query(self.list_nodes_query) if n["id"] == "n1"]
        assert len(matching) == 1


class EvaluatorContractTests:
    def test_satisfies_protocol(self, evaluator):
        assert isinstance(evaluator, Evaluator)

    def test_evaluate_returns_evaluation_result(self, evaluator):
        result = RAGResult(task_type="question_answering", output="seal wear")
        report = evaluator.evaluate(result, reference="seal wear")
        assert isinstance(report, EvaluationResult)
        assert all(isinstance(value, float) for value in report.metrics.values())


class ExporterContractTests:
    def test_satisfies_protocol(self, exporter):
        assert isinstance(exporter, Exporter)

    def test_export_writes_nonempty_file(self, exporter, tmp_path):
        result = RAGResult(task_type="question_answering", output="answer")
        target = tmp_path / "result.json"
        exporter.export(result, str(target))
        assert target.is_file()
        assert target.stat().st_size > 0
