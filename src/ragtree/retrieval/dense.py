# ragtree/retrieval/dense.py
"""Dense retrieval over any VectorStore."""

from __future__ import annotations

from ragtree.core.protocols import VectorStore
from ragtree.core.schemas import Document, EvidenceSpan, RAGTask

__all__ = ["DenseRetriever"]


class DenseRetriever:
    """Retriever protocol implementation delegating to a VectorStore."""

    def __init__(self, vector_store: VectorStore, top_k: int = 5) -> None:
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(
        self, task: RAGTask, documents: list[Document] | None = None
    ) -> list[EvidenceSpan]:
        return self.vector_store.search(task.query or "", top_k=self.top_k)
