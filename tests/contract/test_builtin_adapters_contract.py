"""Contract conformance of the shipped no-dependency adapters."""

import pytest

from ragtree.core.schemas import RAGTask
from ragtree.evaluation.relation_evaluator import RelationEvaluator
from ragtree.integrations.exporters import (
    CsvExporter,
    GraphCsvExporter,
    JsonExporter,
    JsonlExporter,
)
from ragtree.integrations.graphstores import LocalGraphStore
from ragtree.integrations.llms import MockLLMProvider
from ragtree.integrations.vectorstores import InMemoryVectorStore
from ragtree.retrieval import (
    DenseRetriever,
    HybridRetriever,
    KGGuidedRetriever,
    OntologyGuidedRetriever,
)
from tests.contract.bases import (
    SAMPLE_CHUNKS,
    EvaluatorContractTests,
    ExporterContractTests,
    GraphStoreContractTests,
    LLMProviderContractTests,
    RetrieverContractTests,
    VectorStoreContractTests,
)


class TestMockLLMProvider(LLMProviderContractTests):
    @pytest.fixture
    def llm_provider(self):
        return MockLLMProvider(reply="a deterministic reply")


class TestPackageInMemoryVectorStore(VectorStoreContractTests):
    @pytest.fixture
    def vector_store(self):
        return InMemoryVectorStore()


class TestDenseRetriever(RetrieverContractTests):
    @pytest.fixture
    def retriever(self):
        store = InMemoryVectorStore()
        store.add_chunks(list(SAMPLE_CHUNKS))
        return DenseRetriever(store, top_k=3)


class TestHybridRetriever(RetrieverContractTests):
    @pytest.fixture
    def retriever(self):
        store = InMemoryVectorStore()
        store.add_chunks(list(SAMPLE_CHUNKS))
        return HybridRetriever(
            [DenseRetriever(store, top_k=3), DenseRetriever(store, top_k=1)], top_k=3
        )


class _StubOntologyStore:
    def load(self, source: str) -> None:  # pragma: no cover - protocol filler
        pass

    def search_concepts(self, text: str, top_k: int = 5):
        return [
            {"uri": "onto:Pump", "label": "Pump", "score": 0.9, "description": "moves fluid"},
            {"uri": "onto:Seal", "label": "Seal", "score": 0.7, "description": None},
        ][:top_k]


class TestOntologyGuidedRetriever(RetrieverContractTests):
    @pytest.fixture
    def retriever(self):
        return OntologyGuidedRetriever(_StubOntologyStore(), top_k=2)


class TestKGGuidedRetriever(RetrieverContractTests):
    @pytest.fixture
    def retriever(self):
        store = LocalGraphStore()
        store.upsert_edges(
            [
                {"source": "pump_P102", "type": "CAUSED_BY", "target": "seal_wear"},
                {"source": "alarm_7741", "type": "TRIGGERED_BY", "target": "pressure_spike"},
            ]
        )
        return KGGuidedRetriever(store, top_k=5)


class TestLocalGraphStoreAdapter(GraphStoreContractTests):
    @pytest.fixture
    def graph_store(self):
        return LocalGraphStore()


class TestRelationEvaluatorContract(EvaluatorContractTests):
    @pytest.fixture
    def evaluator(self):
        return RelationEvaluator()


class TestPackageJsonExporter(ExporterContractTests):
    @pytest.fixture
    def exporter(self):
        return JsonExporter()


class TestJsonlExporter(ExporterContractTests):
    @pytest.fixture
    def exporter(self):
        return JsonlExporter()


class TestCsvExporter(ExporterContractTests):
    @pytest.fixture
    def exporter(self):
        return CsvExporter()


class TestGraphCsvExporter(ExporterContractTests):
    @pytest.fixture
    def exporter(self):
        return GraphCsvExporter()


def test_local_graph_store_neighbors_query():
    store = LocalGraphStore()
    store.upsert_edges([{"source": "a", "type": "CAUSES", "target": "b"}])
    assert store.query("neighbors:a") == [{"source": "a", "type": "CAUSES", "target": "b"}]
    with pytest.raises(ValueError):
        store.query("MATCH (n) RETURN n")


def test_hybrid_retriever_prefers_items_ranked_by_multiple_retrievers():
    store = InMemoryVectorStore()
    store.add_chunks(list(SAMPLE_CHUNKS))
    hybrid = HybridRetriever(
        [DenseRetriever(store, top_k=3), DenseRetriever(store, top_k=3)], top_k=3
    )
    spans = hybrid.retrieve(RAGTask(task_type="qa", query="pump seal failure"))
    assert spans
    assert spans[0].metadata.get("fusion") == "rrf"
    scores = [span.score for span in spans]
    assert scores == sorted(scores, reverse=True)
