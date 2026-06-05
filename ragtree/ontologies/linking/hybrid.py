from __future__ import annotations

from typing import Any, Dict

from ragtree.ontologies.loader import OntologyIndex
from .base import make_links_v1

class HybridOntologyLinker:
    """Placeholder hybrid linker (rules + embeddings)."""

    def __init__(self, *, ontology_index: OntologyIndex, method: str = "hybrid", ontology_key: str = "unknown", **kwargs: Any):
        self.ontology_index = ontology_index
        self.method = method
        self.ontology_key = ontology_key
        self.kwargs = kwargs

    def link_document(self, doc: Dict[str, Any], *, top_k: int = 3, min_score: float = 0.3) -> Dict[str, Any]:
        return make_links_v1(
            method=self.method,
            ontology_key=self.ontology_key,
            params={"note": "not_implemented", "top_k": top_k, "min_score": min_score},
            by_entity={},
        )
