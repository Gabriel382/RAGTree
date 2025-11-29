# ragtree/ontologies/mapping.py
from __future__ import annotations

from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import math

from ragtree.ontologies.loader import OntologyIndex, OntologyConcept
from ragtree.services.llm.llm import embed_text  # <--- use service


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


class OntologyEntityLinker:
    def __init__(self, ontology_index: OntologyIndex, backend: str = "ollama"):
        self.ontology_index = ontology_index
        self.backend = backend
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
        top_k: int = 3,
        min_score: float = 0.3,
    ) -> Dict[str, List[Dict[str, Any]]]:
        entities = doc.get("entities", {}) or {}
        links: Dict[str, List[Dict[str, Any]]] = {}

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

            links[ent_id] = [
                {
                    "concept_uri": c.uri,
                    "label": c.label,
                    "score": float(score),
                }
                for score, c in top
            ]

        return links

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
