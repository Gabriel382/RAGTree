# ragtree/retrieval/kg_guided.py
"""KG-guided retrieval over any GraphStore.

Protocol-level counterpart of the research KG-RAG retrievers: matches query
tokens against triples and verbalizes them as evidence. The heavyweight
community / triple retrievers stay under ``ragtree.kg`` for the scripts.
"""

from __future__ import annotations

from ragtree.core.protocols import GraphStore
from ragtree.core.schemas import Document, EvidenceSpan, RAGTask

__all__ = ["KGGuidedRetriever"]


def _tokens(text: str) -> set[str]:
    return {t for t in text.lower().replace("_", " ").replace("-", " ").split() if t}


class KGGuidedRetriever:
    """Ranks graph edges by token overlap with the task query.

    ``edges_query`` is the GraphStore query returning edge dicts with
    ``source``/``type``/``target`` keys ('edges' for the local store; a
    Cypher query for Neo4j).
    """

    def __init__(
        self, graph_store: GraphStore, top_k: int = 10, edges_query: str = "edges"
    ) -> None:
        self.graph_store = graph_store
        self.top_k = top_k
        self.edges_query = edges_query

    def retrieve(
        self, task: RAGTask, documents: list[Document] | None = None
    ) -> list[EvidenceSpan]:
        query_tokens = _tokens(task.query or "")
        if not query_tokens:
            return []

        scored: list[tuple[float, dict]] = []
        for edge in self.graph_store.query(self.edges_query):
            source = str(edge.get("source", ""))
            target = str(edge.get("target", ""))
            rel = str(edge.get("type", "RELATED"))
            names = " ".join(
                str(edge.get(k, "")) for k in ("source_label", "target_label")
            )
            edge_tokens = _tokens(f"{source} {rel} {target} {names}")
            overlap = len(query_tokens & edge_tokens)
            if overlap:
                scored.append((float(overlap), edge))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        spans: list[EvidenceSpan] = []
        for score, edge in scored[: self.top_k]:
            source = str(edge.get("source", ""))
            target = str(edge.get("target", ""))
            rel = str(edge.get("type", "RELATED"))
            spans.append(
                EvidenceSpan(
                    document_id=str(edge.get("document_id", source)),
                    text=f"{source} —{rel}→ {target}",
                    score=score,
                    metadata={"source": "graph"},
                )
            )
        return spans
