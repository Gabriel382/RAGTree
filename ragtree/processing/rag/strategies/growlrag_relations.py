# ragtree/processing/rag/strategies/growlrag_relations.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from ragtree.processing.rag.base_strategy import BaseRelationStrategy
from ragtree.ontologies.retrieval.subontology import SubOntologyRetriever, SubOntologyFragment


DEFAULT_FALLBACK_RELATION_TYPE = "causal_relation"


class GrowlRagRelationStrategy(BaseRelationStrategy):
    """
    GrOWL-RAG (ontology-guided RAG) relation extraction strategy.

    Inputs (per document):
      - doc["title"], doc["sentences"] or doc["text"]
      - doc["entities"] (IDs + mentions)
      - doc["ontology_links"] (schema v1 recommended; legacy supported by retriever)

    External inputs (strategy-level):
      - SubOntologyRetriever (ontology TTL loaded/cached once)

    Output:
      - dict { relation_type: [[head_id, tail_id], ...], ... }
      - runner will write this to doc["pred_relations"]

    Notes:
      - This strategy is *read-only* on doc; runner writes outputs.
      - Reuses BaseRelationStrategy._normalize_relation_dict for schema compliance.
    """

    def __init__(
        self,
        llm_config,
        *,
        retriever: SubOntologyRetriever,
        ontology_key: str,
        linking_method: str,
        include_ttl: bool = True,
        include_structured_fragment: bool = True,
        max_sentences_in_prompt: Optional[int] = None,
    ) -> None:
        super().__init__(llm_config)
        self.retriever = retriever
        self.ontology_key = ontology_key
        self.linking_method = linking_method
        self.include_ttl = include_ttl
        self.include_structured_fragment = include_structured_fragment
        self.max_sentences_in_prompt = max_sentences_in_prompt

    # ------------------------------------------------------------------
    # Relation type schema inference
    # ------------------------------------------------------------------
    def _infer_relation_types_from_doc(self, doc: Dict[str, Any]) -> List[str]:
        rels = doc.get("relations")
        if isinstance(rels, dict):
            keys = list(rels.keys())
            if keys:
                return keys
        return [DEFAULT_FALLBACK_RELATION_TYPE]

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------
    def _build_messages(
        self,
        doc: Dict[str, Any],
        relation_types: Sequence[str],
        fragment: SubOntologyFragment,
    ) -> List[Dict[str, str]]:
        system_msg = {"role": "system", "content": self.llm_config.system_prompt}

        title = doc.get("title", "")

        # Prefer sentences list if present; fallback to text.
        sentences = doc.get("sentences")
        if isinstance(sentences, list) and all(isinstance(s, str) for s in sentences):
            sents = sentences
            if self.max_sentences_in_prompt is not None:
                sents = sents[: self.max_sentences_in_prompt]
            doc_text = "\n".join(f"- {s}" for s in sents)
        else:
            doc_text = str(doc.get("text", ""))

        entities = doc.get("entities", {})
        entity_lines: List[str] = []
        if isinstance(entities, dict):
            for ent_id, ent in entities.items():
                ent_type = ent.get("type", "")
                mentions = ent.get("mentions", [])
                if not isinstance(mentions, list):
                    mentions = [mentions]

                # Try to show 12 mentions max for prompt budget
                shown = 0
                for m in mentions:
                    if not isinstance(m, dict):
                        continue
                    trig = m.get("trigger_word") or m.get("text") or ""
                    sent_id = m.get("sent_id")
                    offset = m.get("offset") or m.get("span")
                    entity_lines.append(
                        f"{ent_id}\tTYPE={ent_type}\tTRIGGER={trig}\tSENT_ID={sent_id}\tOFFSET={offset}"
                    )
                    shown += 1
                    if shown >= 2:
                        break
        if not entity_lines:
            entity_lines = ["(no entities found)"]

        # Relation schema list
        rel_schema = "\n".join(f"- {r}" for r in relation_types)

        # Subontology fragment
        frag_structured = json.dumps(fragment.to_dict(), ensure_ascii=False, indent=2)

        ttl_block = ""
        if self.include_ttl:
            ttl_block = fragment.to_ttl()

        user_parts = [
            "You will extract document-level relations between the PROVIDED entity IDs only.",
            "",
            "## Document",
            f"Title: {title}",
            "Text:",
            doc_text,
            "",
            "## Entities (IDs are canonical  use them in output)",
            "\n".join(entity_lines),
            "",
            "## Allowed relation types (output keys must match these exactly)",
            rel_schema,
            "",
            "## Ontology-guided retrieval context",
        ]

        if self.include_structured_fragment:
            user_parts.append("### Sub-ontology fragment (structured JSON)")
            user_parts.append(frag_structured)

        if self.include_ttl:
            user_parts.append("### Sub-ontology fragment (TTL)")
            user_parts.append(ttl_block)

        user_parts += [
            "",
            "## Output format (JSON only, no extra text)",
            "Return a JSON object whose keys are the allowed relation types, and values are lists of [HEAD_ID, TAIL_ID] pairs.",
            "Example:",
            '{"REL_TYPE": [["E1","E2"]], "OTHER": []}',
        ]

        user_msg = {"role": "user", "content": "\n".join(user_parts)}
        return [system_msg, user_msg]

    # ------------------------------------------------------------------
    # Robust JSON parsing (handles code fences)
    # ------------------------------------------------------------------
    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Try hard to recover a JSON object from the LLM output.
        Supports:
          - raw JSON
          - ```json ... ```
          - extra leading/trailing whitespace
        """
        if not text or not isinstance(text, str):
            return None

        s = text.strip()

        # Strip ```json fences if present
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)

        # First attempt: direct parse
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        # Second attempt: find the first {...} block (best-effort)
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            return None

        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def predict_relations(
        self,
        doc: Dict[str, Any],
        relation_types: Optional[Sequence[str]] = None,
    ) -> Dict[str, List[List[str]]]:
        # Determine schema
        rel_types = list(relation_types) if relation_types else self._infer_relation_types_from_doc(doc)

        # Need ontology_links to retrieve a fragment
        ontology_links = doc.get("ontology_links")
        if ontology_links is None:
            # No linking artifact available ? return empty predictions under schema
            return {r: [] for r in rel_types}

        # Retrieve a compact subontology fragment
        fragment = self.retriever.retrieve(
            ontology_links=ontology_links,
            method=self.linking_method,
            params={},  # optionally pass linker params if you have them in doc["_meta"]
        )

        # Build prompt and call LLM
        messages = self._build_messages(doc, rel_types, fragment)
        raw = self._call_llm(messages)

        parsed = self._extract_json_object(raw)
        if not isinstance(parsed, dict):
            return {r: [] for r in rel_types}

        # Step 6: make evaluator-compatible (entity IDs only)
        parsed_entity_only = self._normalize_pred_endpoints_to_entity_ids(
            doc,
            parsed,
            rel_types,
            keep_debug=True,
        )

        # Normalize output to canonical schema
        normalized = self._normalize_relation_dict(parsed_entity_only, rel_types)
        return normalized

    
    def _build_literal_to_entity_index(self, doc: Dict[str, Any]) -> Dict[str, str]:
        """
        Build a deterministic lookup from normalized mention strings -> entity_id.
        Only keeps keys that map to exactly one entity (unambiguous).
        """
        entities = doc.get("entities") or {}
        hits: Dict[str, Set[str]] = {}

        def norm(s: str) -> str:
            return re.sub(r"\s+", " ", s.strip().lower())

        for ent_id, ent in (entities.items() if isinstance(entities, dict) else []):
            mentions = ent.get("mentions") or []
            if not isinstance(mentions, list):
                continue

            for m in mentions:
                if not isinstance(m, dict):
                    continue
                # Primary: trigger_word (your schema)
                texts = []
                if m.get("trigger_word"):
                    texts.append(str(m["trigger_word"]))
                # Optional fallbacks (some datasets use 'text' or other fields)
                if m.get("text"):
                    texts.append(str(m["text"]))

                for t in texts:
                    k = norm(t)
                    if not k:
                        continue
                    hits.setdefault(k, set()).add(ent_id)

        # Keep only unambiguous mappings
        out: Dict[str, str] = {}
        for k, ids in hits.items():
            if len(ids) == 1:
                out[k] = next(iter(ids))
        return out

    def _maybe_map_literal_to_entity(self, doc: Dict[str, Any], lit: str, idx: Dict[str, str]) -> Optional[str]:
        """
        Map a literal string to an entity_id if unambiguous.
        """
        if not isinstance(lit, str):
            return None
        key = re.sub(r"\s+", " ", lit.strip().lower())
        if not key:
            return None
        return idx.get(key)

    def _normalize_pred_endpoints_to_entity_ids(
        self,
        doc: Dict[str, Any],
        pred: Dict[str, Any],
        allowed_relation_types: Sequence[str],
        *,
        keep_debug: bool = True,
    ) -> Dict[str, Any]:
        """
        Enforce evaluator compatibility:
          - Keep only pairs where both endpoints are entity IDs present in doc['entities']
          - If object endpoint is a literal, try to map it to a unique entity via mention strings.
          - If mapping fails, drop that pair (and optionally log it into debug fields).

        Returns a dict suitable for _normalize_relation_dict(...).
        """
        entities = doc.get("entities") or {}
        entity_ids = set(entities.keys()) if isinstance(entities, dict) else set()

        idx = self._build_literal_to_entity_index(doc)

        debug_dropped: Dict[str, List[List[Any]]] = {r: [] for r in allowed_relation_types}
        debug_mapped: Dict[str, List[List[Any]]] = {r: [] for r in allowed_relation_types}

        out: Dict[str, List[List[str]]] = {r: [] for r in allowed_relation_types}

        for r in allowed_relation_types:
            pairs = pred.get(r, [])
            if not isinstance(pairs, list):
                continue

            for pair in pairs:
                if (
                    not isinstance(pair, list)
                    or len(pair) != 2
                ):
                    continue

                h, t = pair[0], pair[1]

                # Head must be an entity id
                if not isinstance(h, str) or h not in entity_ids:
                    if keep_debug:
                        debug_dropped[r].append([h, t])
                    continue

                # Tail: if entity id, keep; else try map literal -> entity id
                if isinstance(t, str) and t in entity_ids:
                    out[r].append([h, t])
                    continue

                mapped = self._maybe_map_literal_to_entity(doc, str(t), idx)
                if mapped and mapped in entity_ids:
                    out[r].append([h, mapped])
                    if keep_debug:
                        debug_mapped[r].append([h, t, mapped])
                else:
                    if keep_debug:
                        debug_dropped[r].append([h, t])

        # Attach debug info (does not affect evaluator)
        if keep_debug:
            doc.setdefault("_debug", {})
            doc["_debug"]["growlrag_normalization"] = {
                "mapped_pairs": debug_mapped,
                "dropped_pairs": debug_dropped,
            }

        return out

