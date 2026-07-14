# ragtree/retrieval/hybrid.py
"""Hybrid retrieval: reciprocal-rank fusion over multiple retrievers."""

from __future__ import annotations

from ragtree.core.protocols import Retriever
from ragtree.core.schemas import Document, EvidenceSpan, RAGTask

__all__ = ["HybridRetriever"]


class HybridRetriever:
    """Fuses ranked lists from several retrievers with RRF.

    score(item) = sum over retrievers of 1 / (k + rank); deterministic and
    scale-free, so heterogeneous retrievers (dense, ontology, KG) can be
    combined without score calibration.
    """

    def __init__(self, retrievers: list[Retriever], top_k: int = 5, k: int = 60) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")
        self.retrievers = list(retrievers)
        self.top_k = top_k
        self.k = k

    def retrieve(
        self, task: RAGTask, documents: list[Document] | None = None
    ) -> list[EvidenceSpan]:
        fused: dict[tuple, dict] = {}
        for retriever in self.retrievers:
            for rank, span in enumerate(retriever.retrieve(task, documents)):
                key = (span.document_id, span.chunk_id, span.text)
                entry = fused.setdefault(key, {"span": span, "score": 0.0})
                entry["score"] += 1.0 / (self.k + rank + 1)

        ranked = sorted(fused.values(), key=lambda e: e["score"], reverse=True)[: self.top_k]
        return [
            EvidenceSpan(
                document_id=e["span"].document_id,
                chunk_id=e["span"].chunk_id,
                text=e["span"].text,
                score=e["score"],
                span=e["span"].span,
                metadata={**e["span"].metadata, "fusion": "rrf"},
            )
            for e in ranked
        ]
