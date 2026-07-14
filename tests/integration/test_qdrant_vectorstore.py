"""Qdrant adapter contract test (in-process :memory: mode, no server)."""

import uuid

import pytest

pytest.importorskip("qdrant_client")

from ragtree.integrations.embedders import HashingEmbedder
from ragtree.integrations.vectorstores import QdrantVectorStore
from tests.contract.bases import VectorStoreContractTests

pytestmark = [pytest.mark.integration, pytest.mark.qdrant]


class TestQdrantVectorStore(VectorStoreContractTests):
    @pytest.fixture
    def vector_store(self):
        return QdrantVectorStore(
            embedder=HashingEmbedder(),
            collection_name=f"contract-{uuid.uuid4().hex[:8]}",
            location=":memory:",
        )
