# ragtree/core/protocols.py
"""Protocol interfaces that make RAGTree bring-your-own-stack.

The core never imports provider SDKs. External systems join a pipeline by
implementing one of these structural protocols; any object with matching
methods is accepted — no inheritance or registration required.

Design reference: BYOS architecture document, section 6.2.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .schemas import (
    Chunk,
    Document,
    EvaluationResult,
    EvidenceSpan,
    RAGResult,
    RAGTask,
)

__all__ = [
    "LLMProvider",
    "Embedder",
    "VectorStore",
    "Retriever",
    "GraphStore",
    "OntologyStore",
    "Evaluator",
    "Exporter",
]


@runtime_checkable
class LLMProvider(Protocol):
    """Chat-completion backend (LiteLLM, Ollama, OpenAI-compatible, ...)."""

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...


@runtime_checkable
class Embedder(Protocol):
    """Text embedding backend."""

    def embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Chunk index with similarity search (in-memory, Chroma, Qdrant, ...)."""

    def add_chunks(
        self, chunks: list[Chunk], embeddings: list[list[float]] | None = None
    ) -> None: ...

    def search(self, query: str, top_k: int = 5, **filters: Any) -> list[EvidenceSpan]: ...


@runtime_checkable
class Retriever(Protocol):
    """Task-aware evidence retrieval (dense, hybrid, ontology- or KG-guided)."""

    def retrieve(
        self, task: RAGTask, documents: list[Document] | None = None
    ) -> list[EvidenceSpan]: ...


@runtime_checkable
class GraphStore(Protocol):
    """Graph backend (Neo4j, local graph, CSV export target)."""

    def upsert_nodes(self, nodes: list[dict[str, Any]]) -> None: ...

    def upsert_edges(self, edges: list[dict[str, Any]]) -> None: ...

    def query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]: ...


@runtime_checkable
class OntologyStore(Protocol):
    """Ontology backend (rdflib / owlready adapters land with the adapter layer).

    Provisional contract: aligned with the existing ``OntologyIndex`` loader,
    it may grow methods when the ontology adapters are ported.
    """

    def load(self, source: str) -> None: ...

    def search_concepts(self, text: str, top_k: int = 5) -> list[dict[str, Any]]: ...


@runtime_checkable
class Evaluator(Protocol):
    """Metric computation over results, optionally against a reference."""

    def evaluate(
        self, result: RAGResult, reference: Any | None = None
    ) -> EvaluationResult: ...


@runtime_checkable
class Exporter(Protocol):
    """Serialization of results to files or external sinks."""

    def export(self, result: RAGResult, output_path: str) -> None: ...
