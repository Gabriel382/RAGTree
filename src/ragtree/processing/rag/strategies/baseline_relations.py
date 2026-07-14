# ragtree/processing/rag/baseline_relations.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from ragtree.processing.rag.base_strategy import BaseRelationStrategy, LLMBackendConfig


DEFAULT_FALLBACK_RELATION_TYPE = "causal_relation"


class BaselineRelationStrategy(BaseRelationStrategy):
    """
    Single-LLM baseline relation extractor.

    - No RAG
    - No ontology
    - Only uses the document text (title/sentence) and entities.
    - Outputs a `pred_relations`-style dict:
        { "TYPE": [["EVENT_x", "EVENT_y"], ...], ... }
    """

    # ------------------------------------------------------------------
    # Relation type schema inference
    # ------------------------------------------------------------------
    def _infer_relation_types_from_doc(self, doc: Dict[str, Any]) -> List[str]:
        """
        Try to infer candidate relation types from doc["relations"].
        If unavailable or empty, fall back to [DEFAULT_FALLBACK_RELATION_TYPE].
        """
        rels = doc.get("relations")
        if isinstance(rels, dict):
            keys = list(rels.keys())
            if keys:
                # Even if the lists are empty, the keys define the schema.
                return keys

        return [DEFAULT_FALLBACK_RELATION_TYPE]

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _build_messages(
        self,
        doc: Dict[str, Any],
        relation_types: Sequence[str],
    ) -> List[Dict[str, str]]:
        """
        Build the chat messages (system + user) for the LLM.

        We expect the LLM to return ONLY valid JSON with the structure:

            {
              "RELATION_TYPE_1": [["EVENT_a", "EVENT_b"], ...],
              "RELATION_TYPE_2": [],
              ...
            }
        """
        system_msg = {
            "role": "system",
            "content": self.llm_config.system_prompt,
        }

        title = doc.get("title", "")
        sentence = doc.get("sentence", doc.get("text", ""))  # fallback to 'text' if needed
        entities = doc.get("entities", {})

        # Represent entities in a structured textual way
        entity_lines: List[str] = []

        # Case 1: entities is a dict like { "EVENT_...": { "type": ..., "mentions": [...] }, ... }
        if isinstance(entities, dict):
            for ent_id, ent in entities.items():
                ent_type = ent.get("type", "")
                mentions = ent.get("mentions", [])  # may be list or single item

                # Normalize mentions to a list
                if not isinstance(mentions, list):
                    mentions = [mentions]

                if mentions:
                    for m in mentions:
                        trigger = (
                            m.get("trigger_word")
                            or m.get("text")
                            or ""
                        )
                        sent_id = m.get("sent_id")
                        offset = m.get("offset") or m.get("span")

                        span_str = ""
                        if isinstance(offset, (list, tuple)) and len(offset) == 2:
                            start, end = offset
                            if isinstance(start, int) and isinstance(end, int):
                                span_str = f" (span: {start}-{end})"

                        mention_id = m.get("id", "")

                        meta_bits = []
                        if mention_id:
                            meta_bits.append(f"mention_id={mention_id}")
                        if sent_id is not None:
                            meta_bits.append(f"sent_id={sent_id}")
                        meta_str = f" ({', '.join(meta_bits)})" if meta_bits else ""

                        # One line per mention, keeping the same entity id
                        entity_lines.append(
                            f'- {ent_id}: "{trigger}" [{ent_type}]{span_str}{meta_str}'
                        )
                else:
                    # Entity without mentions (rare but possible)
                    entity_lines.append(
                        f'- {ent_id}: [type={ent_type}] (no mentions)'
                    )

        # Case 2: fallback to original behavior if entities is a list
        elif isinstance(entities, list):
            for ent in entities:
                ent_id = ent.get("id") or ent.get("event_id") or ""
                ent_text = ent.get("text", "")
                ent_type = ent.get("type", "")
                start = ent.get("start")
                end = ent.get("end")

                span_str = ""
                if isinstance(start, int) and isinstance(end, int):
                    span_str = f" (span: {start}-{end})"

                entity_lines.append(
                    f'- {ent_id}: "{ent_text}" [{ent_type}]{span_str}'
                )

        entities_block = "\n".join(entity_lines) if entity_lines else "(no entities provided)"

        rel_types_str = ", ".join(relation_types) if relation_types else "(none)"

        user_instructions = f"""
You are given a sentence (and optionally a title) plus a list of entities with IDs.
Your **only** task is to extract binary relations between entities, using only the following relation types:

  {rel_types_str}

You must output a JSON object where:
  - each key is one of the relation types above, and
  - each value is a list of pairs [head_entity_id, tail_entity_id].

If there are no relations of a given type, use an empty list [].

Do not invent new relation type names. Use only the relation types listed.

Title:
{title}

Sentence:
{sentence}

Entities:
{entities_block}

Now output ONLY the JSON object. Do not include any explanation or text outside the JSON.
        """.strip()

        user_msg = {"role": "user", "content": user_instructions}
        # Uncomment for debugging:
        # print(user_msg)
        return [system_msg, user_msg]

    # ------------------------------------------------------------------
    # Output parsing (with alias mapping)
    # ------------------------------------------------------------------
    def _parse_llm_output(
        self,
        raw_text: str,
        relation_types: Sequence[str],
    ) -> Dict[str, List[List[str]]]:
        """
        Parse the JSON returned by the LLM into the normalized relation dict.

        We also support alias keys from the LLM, e.g.:
          - "P17"          -> "P17 : country"
          - "country"      -> "P17 : country"
          - "P17 : country" -> "P17 : country"

        All aliases are merged into the canonical keys from `relation_types`.
        """
        try:
            data = json.loads(raw_text)
            if not isinstance(data, dict):
                raise ValueError("LLM output is not a JSON object")
        except Exception:
            # In case of any error, fall back to empty lists for all types
            return {rtype: [] for rtype in relation_types}

        # ------------------------------------------------------------------
        # Build an alias map:
        #   - canonical_key (e.g. "P17 : country") -> itself
        #   - short code (e.g. "P17")             -> canonical_key
        #   - label only (e.g. "country"/"Country")-> canonical_key
        # ------------------------------------------------------------------
        alias_map: Dict[str, str] = {}

        for r in relation_types:
            canonical = r
            alias_map[canonical] = canonical  # full form

            # If format looks like "P17 : country"
            if ":" in r:
                code_part, label_part = r.split(":", 1)
                code = code_part.strip()
                label = label_part.strip()

                if code:
                    alias_map[code] = canonical
                if label:
                    alias_map[label] = canonical
                    alias_map[label.lower()] = canonical

        # ------------------------------------------------------------------
        # Remap LLM keys into canonical keys using alias_map
        # ------------------------------------------------------------------
        remapped: Dict[str, Any] = {}

        for key, value in data.items():
            if not isinstance(key, str):
                continue

            k_stripped = key.strip()
            # Try exact, then lower-case
            canonical = alias_map.get(k_stripped) or alias_map.get(k_stripped.lower())
            if canonical is None:
                # Unknown relation type from the model -> ignore
                continue

            # Only accept list-like values (list of pairs)
            if not isinstance(value, list):
                continue

            # Merge values if multiple aliases map to same canonical key
            existing = remapped.setdefault(canonical, [])
            existing.extend(value)

        # Finally, normalize structure and filter/complete with empty lists
        return self._normalize_relation_dict(remapped, relation_types)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict_relations(
        self,
        doc: Dict[str, Any],
        relation_types: Optional[Sequence[str]] = None,
    ) -> Dict[str, List[List[str]]]:
        """
        Main entrypoint.

        - If relation_types is provided, use it as the schema.
        - Otherwise, infer from doc["relations"] or fall back to a single
          'causal_relation' label.
        """
        if relation_types is None or len(relation_types) == 0:
            rel_types = self._infer_relation_types_from_doc(doc)
        else:
            rel_types = list(relation_types)

        messages = self._build_messages(doc, rel_types)
        raw = self._call_llm(messages)
        
        return self._parse_llm_output(raw, rel_types)


__all__ = [
    "LLMBackendConfig",
    "BaselineRelationStrategy",
    "DEFAULT_FALLBACK_RELATION_TYPE",
]
