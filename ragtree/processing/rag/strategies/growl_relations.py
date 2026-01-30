# ragtree/processing/rag/strategies/growl_relations.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ragtree.processing.rag.strategies.baseline_relations import (
    BaselineRelationStrategy,
    LLMBackendConfig,
)


class GrowlRelationStrategy(BaselineRelationStrategy):
    """
    GrOWL / Ontology-guided RAG relation extractor.

    This strategy behaves like the BaselineRelationStrategy but
    enhances the user prompt with ontology-based context derived
    from `doc["ontology_links"]`, which is expected to be produced
    beforehand by `scripts/run_ontology_linking.py`.

    The overall flow remains:
      - build messages (system + user)
      - call LLM
      - parse JSON output into the standard relation schema.

    Only `_build_messages()` is customized to inject an
    "Ontology context" section into the first user message.
    """

    def __init__(
        self,
        llm_config: LLMBackendConfig,
        *,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        super().__init__(llm_config, max_tokens=max_tokens, system_prompt=system_prompt)

    # ------------------------------------------------------------------
    # Public API (same as baseline)
    # ------------------------------------------------------------------
    def predict_relations(
        self,
        doc: Dict[str, Any],
        relation_types: Optional[Sequence[str]] = None,
    ) -> Dict[str, List[List[str]]]:
        """
        Same interface as BaselineRelationStrategy.

        We just rely on our overridden `_build_messages()` to
        embed ontology context into the prompt.
        """
        # This will call our custom _build_messages(), then
        # _call_llm() and _parse_llm_output() from the parent.
        return super().predict_relations(doc, relation_types)

    # ------------------------------------------------------------------
    # Message construction (ontology-augmented)
    # ------------------------------------------------------------------
    def _build_messages(
        self,
        doc: Dict[str, Any],
        relation_types: Sequence[str],
    ) -> List[Dict[str, str]]:
        """
        Build chat messages for the LLM, like the baseline, but with
        an additional "Ontology context" segment based on
        `doc["ontology_links"]` if available.
        """
        # First get the baseline messages: typically [system, user]
        base_messages = super()._build_messages(doc, relation_types)
        if not isinstance(base_messages, list) or not base_messages:
            return base_messages

        # Inject ontology info into the first user message we find
        messages: List[Dict[str, str]] = []
        ontology_context = self._build_ontology_context(doc)
        injected = False

        for msg in base_messages:
            if (not injected) and msg.get("role") == "user" and ontology_context:
                new_msg = dict(msg)  # shallow copy
                content = new_msg.get("content", "")
                content = f"{content}\n\nOntology context:\n{ontology_context}"
                new_msg["content"] = content
                messages.append(new_msg)
                injected = True
            else:
                messages.append(msg)

        return messages

    # ------------------------------------------------------------------
    # Helper: build textual ontology context
    # ------------------------------------------------------------------
    def _build_ontology_context(self, doc: Dict[str, Any]) -> str:
        """
        Turn doc['ontology_links'] into a compact textual context.

        Expected structure (produced by OntologyEntityLinker):

            doc["ontology_links"] = {
                "<ent_id>": [
                    {"concept_uri": "...", "label": "...", "score": 0.87},
                    ...
                ],
                ...
            }

        We also use doc["entities"] to extract a short local description
        (first mention trigger + sentence) when possible.
        """
        links = doc.get("ontology_links") or {}
        if not links:
            return ""

        entities = doc.get("entities") or {}
        lines: List[str] = []

        for ent_id, candidates in links.items():
            if not candidates:
                continue

            ent = entities.get(ent_id, {})
            ent_desc = self._describe_entity(doc, ent_id, ent)

            if ent_desc:
                lines.append(f"- Entity {ent_id}: {ent_desc}")
            else:
                lines.append(f"- Entity {ent_id}:")

            for cand in candidates:
                label = cand.get("label") or cand.get("concept_uri") or "UNKNOWN_CONCEPT"
                uri = cand.get("concept_uri") or ""
                score = cand.get("score", None)

                if score is not None:
                    try:
                        score_str = f"{float(score):.3f}"
                    except (TypeError, ValueError):
                        score_str = str(score)
                    lines.append(f"    * {label} ({uri}), score={score_str}")
                else:
                    lines.append(f"    * {label} ({uri})")

        return "\n".join(lines)

    def _describe_entity(
        self,
        doc: Dict[str, Any],
        ent_id: str,
        ent: Dict[str, Any],
    ) -> str:
        """
        Build a short human-readable description for an entity,
        reusing the same ideas as OntologyEntityLinker._build_entity_text_representation.
        """
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

        parts: List[str] = []
        if trigger:
            parts.append(f"trigger: {trigger}")
        if etype:
            parts.append(f"type: {etype}")
        if sent_text:
            parts.append(f"sent: {sent_text}")

        return " ; ".join(parts)


__all__ = [
    "GrowlRelationStrategy",
    "LLMBackendConfig",
]
