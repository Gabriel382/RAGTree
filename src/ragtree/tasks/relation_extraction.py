# ragtree/tasks/relation_extraction.py
"""Relation extraction as a first-class RAG task.

Two usage modes:

1. Generic pipeline mode: ``RelationExtractionTask`` builds a prompt from a
   document + entity ids + relation schema and parses strict-JSON output —
   the same output contract the benchmark strategies use.
2. Strategy wrapping: ``results_from_strategy`` runs any existing
   ``BaseRelationStrategy`` (baseline, ICL, CoT, KG-RAG, OG-RAG, ...)
   unchanged and lifts its predictions into core ``RAGResult`` objects.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from ragtree.core.schemas import Document, RAGResult, RelationPrediction
from ragtree.generation.json_utils import extract_first_json, normalize_relations

from .base import BaseTask

__all__ = ["RelationExtractionTask", "results_from_strategy", "format_entities"]


def format_entities(entities: Any) -> str:
    """Render the historical entity structures as prompt lines.

    Supports the preprocessed-dataset format ``{entity_id: {type, mentions}}``
    and a simple list of dicts with an ``id`` field.
    """
    lines: list[str] = []
    if isinstance(entities, dict):
        for ent_id, ent in entities.items():
            ent = ent if isinstance(ent, dict) else {}
            ent_type = ent.get("type", "")
            mentions = ent.get("mentions", [])
            if not isinstance(mentions, list):
                mentions = [mentions]
            triggers = []
            for mention in mentions:
                if isinstance(mention, dict):
                    trigger = mention.get("trigger_word") or mention.get("text") or ""
                    if trigger:
                        triggers.append(str(trigger))
            label = ", ".join(dict.fromkeys(triggers)) or ent_id
            suffix = f" [{ent_type}]" if ent_type else ""
            lines.append(f'- {ent_id}: "{label}"{suffix}')
    elif isinstance(entities, list):
        for ent in entities:
            if isinstance(ent, dict) and "id" in ent:
                label = ent.get("text") or ent.get("label") or ent["id"]
                lines.append(f'- {ent["id"]}: "{label}"')
    return "\n".join(lines) or "(no entities provided)"


class RelationExtractionTask(BaseTask):
    task_type = "relation_extraction"
    default_system_prompt = (
        "You are an expert at document-level relation extraction.\n"
        "You MUST output ONLY valid JSON (no markdown, no explanations).\n"
        "You MUST use ONLY the PROVIDED entity IDs in output pairs.\n"
        "Output keys MUST match the allowed relation types exactly.\n"
        "Values MUST be lists of [HEAD_ID, TAIL_ID] pairs.\n"
        "If unsure, output empty lists for those relation types."
    )

    def __init__(
        self,
        relation_types: Sequence[str],
        document: Document | dict[str, Any] | None = None,
        entities: Any = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.relation_types = list(relation_types)
        self.document = document
        self.entities = entities
        self.output_schema = {
            rtype: "list of [head_id, tail_id] pairs" for rtype in self.relation_types
        }

    # ------------------------------------------------------------------
    def _document_fields(self) -> tuple[str, str, str, Any]:
        doc = self.document
        if isinstance(doc, Document):
            return doc.id, doc.title or "", doc.text, self.entities
        if isinstance(doc, dict):
            doc_id = str(doc.get("document_id") or doc.get("doc_id") or doc.get("id") or "")
            title = str(doc.get("title") or "")
            text = str(doc.get("text") or doc.get("sentence") or "")
            entities = self.entities if self.entities is not None else doc.get("entities")
            return doc_id, title, text, entities
        return "", "", "", self.entities

    @property
    def document_id(self) -> str:
        return self._document_fields()[0]

    def instructions(self) -> str:
        doc_id, title, text, entities = self._document_fields()
        schema_lines = "\n".join(f"- {rtype}" for rtype in self.relation_types)
        return (
            f"Document id: {doc_id}\n"
            f"Title: {title}\n"
            f"Text:\n{text}\n\n"
            f"Entities:\n{format_entities(entities)}\n\n"
            f"Allowed relation types:\n{schema_lines}\n\n"
            "Extract the relations now. Output JSON only."
        )

    def parse_output(self, text: str) -> dict[str, list[list[str]]]:
        return normalize_relations(extract_first_json(text), self.relation_types)

    def make_prediction(
        self,
        relations: dict[str, list[list[str]]],
        method: str | None = None,
        model: str | None = None,
    ) -> RelationPrediction:
        return RelationPrediction(
            document_id=self.document_id,
            relations=relations,
            method=method,
            model=model,
        )


def results_from_strategy(
    strategy: Any,
    documents: Iterable[dict[str, Any]],
    relation_types: Sequence[str] | None = None,
) -> list[RAGResult]:
    """Run an existing ``BaseRelationStrategy`` unchanged over documents.

    ``strategy`` must expose ``predict_relations(doc, relation_types=...)``
    (every benchmark strategy does). Predictions are lifted into
    ``RAGResult`` objects with a ``RelationPrediction`` artifact, keeping the
    historical ``pred_relations`` format intact.
    """
    results: list[RAGResult] = []
    for doc in documents:
        relations = strategy.predict_relations(doc, relation_types=relation_types)
        doc_id = str(doc.get("document_id") or doc.get("doc_id") or doc.get("id") or "")
        prediction = RelationPrediction(
            document_id=doc_id,
            relations=relations,
            method=type(strategy).__name__,
            model=getattr(getattr(strategy, "llm_config", None), "model", None),
        )
        results.append(
            RAGResult(
                task_type="relation_extraction",
                output=relations,
                metadata={"document_id": doc_id},
                artifacts={"prediction": prediction.model_dump()},
            )
        )
    return results
