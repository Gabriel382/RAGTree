# ragtree/retrieval/ontology_guided.py
"""Ontology-guided retrieval over any OntologyStore.

Protocol-level counterpart of the research chunk-ORAG / subontology
retrievers: returns matched ontology concepts as evidence spans that ground
the generation step. The index-based research retrievers remain available
under ``ragtree.ontologies.retrieval`` for the benchmark scripts.
"""

from __future__ import annotations

from ragtree.core.protocols import OntologyStore
from ragtree.core.schemas import Document, EvidenceSpan, RAGTask

__all__ = ["OntologyGuidedRetriever"]


class OntologyGuidedRetriever:
    def __init__(self, ontology_store: OntologyStore, top_k: int = 5) -> None:
        self.ontology_store = ontology_store
        self.top_k = top_k

    def retrieve(
        self, task: RAGTask, documents: list[Document] | None = None
    ) -> list[EvidenceSpan]:
        concepts = self.ontology_store.search_concepts(task.query or "", top_k=self.top_k)
        spans: list[EvidenceSpan] = []
        for concept in concepts:
            label = str(concept.get("label", ""))
            description = concept.get("description") or ""
            text = f"{label}: {description}".strip().rstrip(":")
            spans.append(
                EvidenceSpan(
                    document_id=str(concept.get("uri", label)),
                    text=text or label,
                    score=float(concept["score"]) if concept.get("score") is not None else None,
                    metadata={
                        "source": "ontology",
                        "aliases": list(concept.get("aliases") or []),
                    },
                )
            )
        return spans
