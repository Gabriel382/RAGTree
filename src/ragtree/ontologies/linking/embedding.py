from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import math

from ragtree.ontologies.loader import OntologyIndex, OntologyConcept
from ragtree.services.llm.llm import embed_text

from .base import OntologyLinkCandidate, make_links_v1

def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

@dataclass
class ConceptEmbedding:
    concept: OntologyConcept
    vector: List[float]

class EmbeddingOntologyLinker:
    """Embedding-based linker (current RAGTree behavior), outputting schema v1."""

    def __init__(
        self,
        *,
        ontology_index: OntologyIndex,
        backend: str = "ollama",
        method: str = "llm_embedding",
        ontology_key: str = "unknown",
    ):
        self.ontology_index = ontology_index
        self.backend = backend
        self.method = method
        self.ontology_key = ontology_key
        self._concept_embs: List[ConceptEmbedding] = []
        self._build_concept_embeddings()

    def _build_concept_embeddings(self) -> None:
        for c in self.ontology_index.concepts:
            txt = c.label
            if c.description:
                txt += f". {c.description}"
            vec = embed_text(txt, backend=self.backend)
            self._concept_embs.append(ConceptEmbedding(concept=c, vector=vec))

    def link_document(
        self,
        doc: Dict[str, Any],
        *,
        top_k: int = 3,
        min_score: float = 0.3,
    ) -> Dict[str, Any]:
        entities = doc.get("entities", {}) or {}
        by_entity: Dict[str, List[OntologyLinkCandidate]] = {}

        for ent_id, ent in entities.items():
            ent_repr = self._build_entity_text_representation(doc, ent_id, ent)
            if not ent_repr.strip():
                continue

            e_vec = embed_text(ent_repr, backend=self.backend)
            candidates: List[Tuple[float, OntologyConcept]] = []

            for ce in self._concept_embs:
                score = cosine(e_vec, ce.vector)
                if score >= min_score:
                    candidates.append((score, ce.concept))

            candidates.sort(key=lambda x: x[0], reverse=True)
            top = candidates[:top_k]

            if top:
                by_entity[ent_id] = [
                    OntologyLinkCandidate(
                        target_kind="class",
                        concept_uri=c.uri,
                        label=c.label,
                        score=float(score),
                        source="embedding",
                        evidence={"entity_repr": ent_repr},
                    )
                    for score, c in top
                ]

        params = {"backend": self.backend, "top_k": top_k, "min_score": min_score}
        return make_links_v1(
            method=self.method,
            ontology_key=self.ontology_key,
            params=params,
            by_entity=by_entity,
            selected_top1=False,
        )

    def _build_entity_text_representation(
        self,
        doc: Dict[str, Any],
        ent_id: str,
        ent: Dict[str, Any],
    ) -> str:
        etype = ent.get("type") or ""
        mentions = ent.get("mentions") or []

        trigger = ""
        sent_text = ""
        sent_id = None
        if mentions:
            m0 = mentions[0]
            trigger = m0.get("trigger_word") or ""
            sent_id = m0.get("sent_id")
            if isinstance(sent_id, int):
                sentences = doc.get("sentences") or []
                if 0 <= sent_id < len(sentences):
                    sent_text = sentences[sent_id]

        parts = []
        if trigger:
            parts.append(f"event trigger: {trigger}")
        if etype:
            parts.append(f"event type: {etype}")
        if sent_text:
            parts.append(f"context: {sent_text}")

        return " ; ".join(parts)
