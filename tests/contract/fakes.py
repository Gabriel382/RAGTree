"""In-memory reference implementations of the core protocols.

These fakes serve two roles: they validate the contract-test machinery
without external services, and they are executable documentation of the
minimum an adapter must do. The vector store and retriever graduate into
``ragtree.integrations`` when the adapter layer lands (sprint 2).
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from ragtree.core.schemas import (
    Chunk,
    Document,
    EvaluationResult,
    EvidenceSpan,
    RAGResult,
    RAGTask,
)

__all__ = [
    "FakeLLMProvider",
    "HashingEmbedder",
    "InMemoryVectorStore",
    "SimpleRetriever",
    "InMemoryGraphStore",
    "ExactMatchEvaluator",
    "JsonExporter",
]


class FakeLLMProvider:
    """Returns a canned reply, or echoes the last user message."""

    def __init__(self, reply: str | None = None) -> None:
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.calls.append(messages)
        if self.reply is not None:
            return self.reply
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        return f"echo: {last_user}"


class HashingEmbedder:
    """Deterministic bag-of-tokens embedding. No ML dependencies."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in text.lower().split():
                digest = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                vec[digest % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            vectors.append([x / norm for x in vec])
        return vectors


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class InMemoryVectorStore:
    """Exact cosine search over normalized embeddings."""

    def __init__(self, embedder: HashingEmbedder | None = None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    def add_chunks(
        self, chunks: list[Chunk], embeddings: list[list[float]] | None = None
    ) -> None:
        vectors = (
            embeddings
            if embeddings is not None
            else self.embedder.embed([c.text for c in chunks])
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
            )
            for chunk, vector in ranked
        ]


class SimpleRetriever:
    """Retriever protocol implementation over any VectorStore."""

    def __init__(self, vector_store: InMemoryVectorStore, top_k: int = 5) -> None:
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(
        self, task: RAGTask, documents: list[Document] | None = None
    ) -> list[EvidenceSpan]:
        return self.vector_store.search(task.query or "", top_k=self.top_k)


class InMemoryGraphStore:
    """Dict-backed graph store; supports the 'nodes' and 'edges' queries."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[tuple, dict[str, Any]] = {}

    def upsert_nodes(self, nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            self.nodes[node["id"]] = {**self.nodes.get(node["id"], {}), **node}

    def upsert_edges(self, edges: list[dict[str, Any]]) -> None:
        for edge in edges:
            key = (edge.get("source"), edge.get("target"), edge.get("type"))
            self.edges[key] = {**self.edges.get(key, {}), **edge}

    def query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        if query == "nodes":
            return list(self.nodes.values())
        if query == "edges":
            return list(self.edges.values())
        raise ValueError(
            f"Unsupported query {query!r}: the fake graph store answers 'nodes' and 'edges'."
        )


class ExactMatchEvaluator:
    def evaluate(
        self, result: RAGResult, reference: Any | None = None
    ) -> EvaluationResult:
        match = float(reference is not None and result.output == reference)
        return EvaluationResult(
            metrics={"exact_match": match}, counts={"n": 1}, method="exact_match"
        )


class JsonExporter:
    def export(self, result: RAGResult, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result.model_dump(mode="json"), handle, ensure_ascii=False, indent=2)
