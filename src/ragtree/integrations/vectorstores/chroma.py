# ragtree/integrations/vectorstores/chroma.py
"""Chroma vector store adapter."""

from __future__ import annotations

from typing import Any

from ragtree.core.errors import require_extra
from ragtree.core.protocols import Embedder
from ragtree.core.schemas import Chunk, EvidenceSpan

__all__ = ["ChromaVectorStore"]


class ChromaVectorStore:
    """VectorStore over Chroma (extra: ``vector-chroma``).

    With an ``embedder`` given, embeddings are computed by RAGTree and Chroma
    only indexes them; without one, Chroma's default embedding function is
    used (may download a model on first run).
    """

    def __init__(
        self,
        collection_name: str = "ragtree",
        persist_directory: str | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        require_extra("chromadb", "vector-chroma")
        import chromadb

        self.embedder = embedder
        client = (
            chromadb.PersistentClient(path=persist_directory)
            if persist_directory
            else chromadb.EphemeralClient()
        )
        self._collection = client.get_or_create_collection(collection_name)

    def add_chunks(
        self, chunks: list[Chunk], embeddings: list[list[float]] | None = None
    ) -> None:
        if embeddings is None and self.embedder is not None:
            embeddings = self.embedder.embed([chunk.text for chunk in chunks])
        kwargs: dict[str, Any] = {
            "ids": [chunk.id for chunk in chunks],
            "documents": [chunk.text for chunk in chunks],
            "metadatas": [
                {"document_id": chunk.document_id, "index": chunk.index} for chunk in chunks
            ],
        }
        if embeddings is not None:
            kwargs["embeddings"] = embeddings
        self._collection.upsert(**kwargs)

    def search(self, query: str, top_k: int = 5, **filters: Any) -> list[EvidenceSpan]:
        kwargs: dict[str, Any] = {"n_results": top_k}
        if self.embedder is not None:
            kwargs["query_embeddings"] = self.embedder.embed([query])
        else:
            kwargs["query_texts"] = [query]
        response = self._collection.query(**kwargs)

        spans: list[EvidenceSpan] = []
        ids = (response.get("ids") or [[]])[0]
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        for i, chunk_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else None
            spans.append(
                EvidenceSpan(
                    document_id=str((metadata or {}).get("document_id", "")),
                    chunk_id=str(chunk_id),
                    text=str(documents[i]) if i < len(documents) else "",
                    score=1.0 / (1.0 + float(distance)) if distance is not None else None,
                    metadata=dict(metadata or {}),
                )
            )
        return spans
