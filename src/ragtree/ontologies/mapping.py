from __future__ import annotations

"""Backward-compatible facade for ontology linking.

Historically, RAGTree exposed `OntologyEntityLinker` here.
We keep that name, but it now produces the v1 `ontology_links` schema by default.

If you need the legacy dict-of-lists output, call `link_document_legacy()`.
"""

from typing import Any, Dict, List

from ragtree.ontologies.linking.embedding import EmbeddingOntologyLinker


class OntologyEntityLinker(EmbeddingOntologyLinker):
    """Alias for backward compatibility."""

    def link_document_legacy(
        self,
        doc: Dict[str, Any],
        *,
        top_k: int = 3,
        min_score: float = 0.3,
    ) -> Dict[str, List[Dict[str, Any]]]:
        links_v1 = super().link_document(doc, top_k=top_k, min_score=min_score)
        legacy: Dict[str, List[Dict[str, Any]]] = {}
        for ent_id, payload in (links_v1.get("by_entity") or {}).items():
            legacy[ent_id] = [
                {
                    "concept_uri": c.get("concept_uri"),
                    "label": c.get("label"),
                    "score": float(c.get("score", 0.0)),
                }
                for c in (payload.get("candidates") or [])
            ]
        return legacy
