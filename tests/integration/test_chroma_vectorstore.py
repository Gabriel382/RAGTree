"""Chroma adapter contract test (ephemeral client, RAGTree-side embeddings)."""

import uuid

import pytest

pytest.importorskip("chromadb")

from ragtree.integrations.embedders import HashingEmbedder
from ragtree.integrations.vectorstores import ChromaVectorStore
from tests.contract.bases import VectorStoreContractTests

pytestmark = [pytest.mark.integration, pytest.mark.chroma]


class TestChromaVectorStore(VectorStoreContractTests):
    @pytest.fixture
    def vector_store(self):
        return ChromaVectorStore(
            collection_name=f"contract-{uuid.uuid4().hex[:8]}",
            embedder=HashingEmbedder(),
        )
