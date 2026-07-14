# ragtree/integrations/vectorstores/qdrant.py
"""Qdrant vector store adapter (supports server, local path and :memory:)."""

from __future__ import annotations

import uuid
from typing import Any

from ragtree.core.errors import require_extra
from ragtree.core.protocols import Embedder
from ragtree.core.schemas import Chunk, EvidenceSpan

__all__ = ["QdrantVectorStore"]

_NAMESPACE = uuid.UUID("2f1c43fa-93a2-4f0d-9a6f-ragtree000000".replace("ragtree000000", "0d0d0d0d0d0d"))


class QdrantVectorStore:
    """VectorStore over Qdrant (extra: ``vector-qdrant``).

    ``location=":memory:"`` runs fully in-process — no server needed, which
    is how the integration tests exercise this adapter.
    """

    def __init__(
        self,
        embedder: Embedder,
        collection_name: str = "ragtree",
        location: str | None = ":memory:",
        url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        require_extra("qdrant_client", "vector-qdrant")
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.embedder = embedder
        self.collection_name = collection_name
        self._client = (
            QdrantClient(url=url, api_key=api_key) if url else QdrantClient(location=location)
        )
        self._dim = len(self.embedder.embed(["dimension probe"])[0])
        if not self._client.collection_exists(collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )

    def add_chunks(
        self, chunks: list[Chunk], embeddings: list[list[float]] | None = None
    ) -> None:
        from qdrant_client.models import PointStruct

        vectors = (
            embeddings
            if embeddings is not None
            else self.embedder.embed([chunk.text for chunk in chunks])
        )
        points = [
            PointStruct(
                id=str(uuid.uuid5(_NAMESPACE, chunk.id)),
                vector=vector,
                payload={
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "index": chunk.index,
                    "metadata": chunk.metadata,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self._client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, top_k: int = 5, **filters: Any) -> list[EvidenceSpan]:
        [query_vec] = self.embedder.embed([query])
        try:
            hits = self._client.query_points(
                collection_name=self.collection_name, query=query_vec, limit=top_k
            ).points
        except AttributeError:  # older qdrant-client
            hits = self._client.search(
                collection_name=self.collection_name, query_vector=query_vec, limit=top_k
            )
        spans: list[EvidenceSpan] = []
        for hit in hits:
            payload = hit.payload or {}
            spans.append(
                EvidenceSpan(
                    document_id=str(payload.get("document_id", "")),
                    chunk_id=payload.get("chunk_id"),
                    text=str(payload.get("text", "")),
                    score=float(hit.score) if hit.score is not None else None,
                    metadata=dict(payload.get("metadata") or {}),
                )
            )
        return spans
