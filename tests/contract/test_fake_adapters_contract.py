"""Contract conformance of the in-memory reference implementations.

One concrete class per protocol; sprint-2 adapters add parallel files
(test_chroma_vectorstore.py, test_litellm_provider.py, ...) under
tests/integration reusing the same bases.
"""

import pytest

from tests.contract import fakes
from tests.contract.bases import (
    SAMPLE_CHUNKS,
    EmbedderContractTests,
    EvaluatorContractTests,
    ExporterContractTests,
    GraphStoreContractTests,
    LLMProviderContractTests,
    RetrieverContractTests,
    VectorStoreContractTests,
)


class TestFakeLLMProvider(LLMProviderContractTests):
    @pytest.fixture
    def llm_provider(self):
        return fakes.FakeLLMProvider()


class TestHashingEmbedder(EmbedderContractTests):
    @pytest.fixture
    def embedder(self):
        return fakes.HashingEmbedder()


class TestInMemoryVectorStore(VectorStoreContractTests):
    @pytest.fixture
    def vector_store(self):
        return fakes.InMemoryVectorStore()


class TestSimpleRetriever(RetrieverContractTests):
    @pytest.fixture
    def retriever(self):
        store = fakes.InMemoryVectorStore()
        store.add_chunks(list(SAMPLE_CHUNKS))
        return fakes.SimpleRetriever(store)


class TestInMemoryGraphStore(GraphStoreContractTests):
    @pytest.fixture
    def graph_store(self):
        return fakes.InMemoryGraphStore()


class TestExactMatchEvaluator(EvaluatorContractTests):
    @pytest.fixture
    def evaluator(self):
        return fakes.ExactMatchEvaluator()


class TestJsonExporter(ExporterContractTests):
    @pytest.fixture
    def exporter(self):
        return fakes.JsonExporter()
