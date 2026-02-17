# ragtree/processing/rag/strategies/ograg_relations.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ragtree.processing.rag.base_strategy import BaseRelationStrategy


DEFAULT_FALLBACK_RELATION_TYPE = "causal_relation"


@dataclass
class OGRagParams:
    ontology_key: str
    ontology_ttl_path: Path
    linking_method: str = "llm_embedding"
    top_k_entities: int = 6
    max_ontology_lines: int = 200
    include_labels: bool = True
    include_comments: bool = True
    include_types: bool = True


class _TTLStringIndex:
    """
    Very lightweight TTL index.
    Goal: extract *some* useful lines mentioning linked URIs without heavy reasoning.

    - Reads TTL as text
    - Builds a mapping from "token-ish" URI fragments -> list of lines
    - At query time: match linked URI string and return nearby/useful lines.

    This keeps dependencies minimal and works with ANY ontology TTL.
    """

    def __init__(self, ttl_path: Path) -> None:
        self.ttl_path = ttl_path
        self._lines: List[str] = []
        self._uri_to_hits: Dict[str, List[int]] = {}

        text = ttl_path.read_text(encoding="utf-8", errors="ignore")
        self._lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]

        # naive URI detector: anything between <...>
        for i, ln in enumerate(self._lines):
            for m in re.finditer(r"<([^>]+)>", ln):
                uri = m.group(1)
                self._uri_to_hits.setdefault(uri, []).append(i)

    def retrieve_for_uris(self, uris: Sequence[str], *, max_lines: int) -> List[str]:
        picked: List[str] = []
        seen_idx = set()

        # gather hit indices + some context window
        candidate_idx: List[int] = []
        for uri in uris:
            for idx in self._uri_to_hits.get(uri, []):
                # include a small context window
                for j in range(max(0, idx - 2), min(len(self._lines), idx + 3)):
                    candidate_idx.append(j)

        # keep order, de-dup
        for idx in candidate_idx:
            if idx in seen_idx:
                continue
            seen_idx.add(idx)
            picked.append(self._lines[idx])
            if len(picked) >= max_lines:
                break

        return picked


class OGRagRelationStrategy(BaseRelationStrategy):
    """
    OG-RAG-style: use ontology grounding as *retrieval context* for relation extraction.

    Inputs (per doc):
      - doc['text' or 'sentences']
      - doc['entities']
      - doc['ontology_links'] (produced by scripts/run_ontology_linking.py)

    Output:
      - { relation_type: [[head_id, tail_id], ...], ... } (entity IDs only)
    """

    def __init__(self, llm_config, *, params: OGRagParams) -> None:
        super().__init__(llm_config)
        self.params = params
        self._ttl_index: Optional[_TTLStringIndex] = None

    # -------------------------
    # Relation schema inference
    # -------------------------
    def _infer_relation_types_from_doc(self, doc: Dict[str, Any]) -> List[str]:
        rels = doc.get("relations")
        if isinstance(rels, dict):
            keys = list(rels.keys())
            if keys:
                return keys
        return [DEFAULT_FALLBACK_RELATION_TYPE]

    # -------------------------
    # Ontology context builder
    # -------------------------
    def _ensure_index(self) -> None:
        if self._ttl_index is None:
            self._ttl_index = _TTLStringIndex(self.params.ontology_ttl_path)

    def _collect_linked_uris(self, doc: Dict[str, Any]) -> List[str]:
        """
        From schema v1:
          doc['ontology_links']['by_entity'][EID]['candidates'] -> concept_uri
        """
        links = doc.get("ontology_links")
        if not isinstance(links, dict):
            return []

        by_ent = links.get("by_entity")
        if not isinstance(by_ent, dict):
            return []

        uris: List[str] = []
        # deterministic iteration
        for ent_id in sorted(by_ent.keys()):
            ent_links = by_ent.get(ent_id) or {}
            cands = ent_links.get("candidates") or []
            if not isinstance(cands, list):
                continue
            for c in cands[: self.params.top_k_entities]:
                if not isinstance(c, dict):
                    continue
                uri = c.get("concept_uri")
                if isinstance(uri, str) and uri:
                    uris.append(uri)

        # de-dup while preserving order
        seen = set()
        out = []
        for u in uris:
            if u in seen:
                continue
            seen.add(u)
            out.append(u)
        return out

    def _build_ontology_context(self, doc: Dict[str, Any]) -> str:
        self._ensure_index()
        assert self._ttl_index is not None

        uris = self._collect_linked_uris(doc)
        if not uris:
            return "(no ontology links available for this document)"

        lines = self._ttl_index.retrieve_for_uris(uris, max_lines=self.params.max_ontology_lines)
        if not lines:
            return "(ontology links found, but no TTL lines matched the linked URIs)"

        # Optionally: filter to readable lines (labels/comments/types)
        if self.params.include_labels or self.params.include_comments or self.params.include_types:
            keep: List[str] = []
            for ln in lines:
                low = ln.lower()
                ok = False
                if self.params.include_labels and ("label" in low or "prefLabel".lower() in low):
                    ok = True
                if self.params.include_comments and ("comment" in low or "definition" in low):
                    ok = True
                if self.params.include_types and (" rdf:type " in low or " a " in low):
                    ok = True
                if ok:
                    keep.append(ln)
            if keep:
                lines = keep[: self.params.max_ontology_lines]

        return "\n".join(lines[: self.params.max_ontology_lines])

    # -------------------------
    # Prompt assembly
    # -------------------------
    def _build_messages(
        self,
        doc: Dict[str, Any],
        relation_types: Sequence[str],
        *,
        ontology_ctx: str,
        few_shots: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, str]]:
        system_msg = {"role": "system", "content": self.llm_config.system_prompt}

        user_parts: List[str] = [
            "You will extract document-level relations between the PROVIDED entity IDs only.",
            "You MUST output ONLY valid JSON (no markdown, no explanations).",
            "You MUST use ONLY the PROVIDED entity IDs in output pairs (no literals).",
            "",
        ]

        if few_shots:
            user_parts.append(self._few_shot_block(few_shots))

        user_parts += [
            "## Document",
            self._doc_text_block(doc),
            "",
            "## Entities (IDs are canonical  use them in output)",
            self._entities_block(doc),
            "",
            "## Allowed relation types (output keys must match these exactly)",
            self._relation_schema_block(relation_types),
            "",
            "## Tool: Ontology grounding context",
            ontology_ctx,
            "",
            "## Output format (JSON only, no extra text)",
            "Return a JSON object whose keys are the allowed relation types, and values are lists of [HEAD_ID, TAIL_ID] pairs.",
            "Example:",
            '{"REL_TYPE": [["E1","E2"]], "OTHER": []}',
        ]

        return [system_msg, {"role": "user", "content": "\n".join(user_parts)}]

    # -------------------------
    # Robust JSON parse
    # -------------------------
    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        if not text or not isinstance(text, str):
            return None

        s = text.strip()
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)

        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            return None

        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    # -------------------------
    # Main API
    # -------------------------
    def predict_relations(
        self,
        doc: Dict[str, Any],
        relation_types: Optional[Sequence[str]] = None,
        *,
        few_shots: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, List[List[str]]]:
        rel_types = list(relation_types) if relation_types else self._infer_relation_types_from_doc(doc)

        ontology_ctx = self._build_ontology_context(doc)

        messages = self._build_messages(doc, rel_types, ontology_ctx=ontology_ctx, few_shots=few_shots)
        raw = self._call_llm(messages)

        parsed = self._extract_json_object(raw)
        if not isinstance(parsed, dict):
            return {r: [] for r in rel_types}

        # enforce schema + entity-id endpoints only (BaseRelationStrategy helper)
        parsed_entity_only = self._normalize_pred_endpoints_to_entity_ids(
            doc, parsed, rel_types, keep_debug=True
        )
        return self._normalize_relation_dict(parsed_entity_only, rel_types)
