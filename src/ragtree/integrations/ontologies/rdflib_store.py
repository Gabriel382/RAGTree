# ragtree/integrations/ontologies/rdflib_store.py
"""OntologyStore adapter over the existing rdflib-based ontology loader."""

from __future__ import annotations

from typing import Any

from ragtree.core.errors import require_extra

__all__ = ["RdflibOntologyStore"]


class RdflibOntologyStore:
    """OntologyStore over TTL/OWL files (extra: ``rdf``).

    Loading reuses the research ``OntologyIndex`` (rdflib); concept search is
    fuzzy lexical matching over labels and aliases via rapidfuzz (a core
    dependency), which keeps the adapter usable without embedding models.
    """

    def __init__(self, source: str | None = None) -> None:
        self._concepts: list[Any] = []
        if source is not None:
            self.load(source)

    def load(self, source: str) -> None:
        require_extra("rdflib", "rdf")
        from pathlib import Path

        from ragtree.ontologies.loader import OntologyIndex

        self._concepts = list(OntologyIndex.from_turtle(Path(source)).concepts)

    def search_concepts(self, text: str, top_k: int = 5) -> list[dict[str, Any]]:
        from rapidfuzz import fuzz

        scored: list[tuple[float, Any]] = []
        for concept in self._concepts:
            names = [concept.label, *(concept.aliases or [])]
            name_score = max(
                (fuzz.WRatio(text, name) for name in names if name), default=0.0
            )
            desc_score = (
                fuzz.partial_ratio(text, concept.description)
                if concept.description
                else 0.0
            )
            # Label/alias matches outrank mentions inside descriptions.
            score = max(name_score, 0.6 * desc_score)
            scored.append((float(score) / 100.0, concept))

        scored.sort(key=lambda pair: (-pair[0], pair[1].label))
        return [
            {
                "uri": concept.uri,
                "label": concept.label,
                "score": score,
                "description": concept.description,
                "aliases": list(concept.aliases or []),
            }
            for score, concept in scored[:top_k]
        ]
