# ragtree/integrations/vectorstores/memory.py
"""In-memory vector store: exact cosine search, no services required.

The reference VectorStore implementation (design backlog P1): used by the
demos, the e2e suite and as executable documentation for adapter authors.
"""

from __future__ import annotations

from typing import Any

from ragtree.core.protocols import Embedder
from ragtree.core.schemas import Chunk, EvidenceSpan
from ragtree.integrations.embedders.hashing import HashingEmbedder

__all__ = ["InMemoryVectorStore"]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class InMemoryVectorStore:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    def add_chunks(
        self, chunks: list[Chunk], embeddings: list[list[float]] | None = None
    ) -> None:
        vectors = (
            embeddings
            if embeddings is not None
            else self.embedder.embed([chunk.text for chunk in chunks])
        )
        self._chunks.extend(chunks)
        self._vectors.extend(vectors)

    def search(self, query: str, top_k: int = 5, **filters: Any) -> list[EvidenceSpan]:
        if not self._chunks:
            return []
        [query_vec] = self.embedder.embed([query])
        ranked = sorted(
            zip(self._chunks, self._vectors),
            key=lambda pair: _dot(query_vec, pair[1]),
            reverse=True,
        )[:top_k]
        return [
            EvidenceSpan(
                document_id=chunk.document_id,
                chunk_id=chunk.id,
                text=chunk.text,
                score=_dot(query_vec, vector),
                metadata=dict(chunk.metadata),
            )
            for chunk, vector in ranked
        ]
