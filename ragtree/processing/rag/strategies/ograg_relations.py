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
    

    def _doc_text_block(self, doc: Dict[str, Any]) -> str:
        """Return a readable document text block for prompts."""
        title = doc.get("title") or ""
        # Common normalized form: list[str]
        sentences = doc.get("sentences")
        if isinstance(sentences, list) and all(isinstance(s, str) for s in sentences):
            body = "\n".join(f"- {s}" for s in sentences)
            return f"Title: {title}\nText:\n{body}".strip()

        # DocRED raw form sometimes: sents = list[list[str]]
        sents = doc.get("sents")
        if isinstance(sents, list) and all(isinstance(s, list) for s in sents):
            lines = []
            for s in sents:
                lines.append("- " + " ".join(str(w) for w in s))
            body = "\n".join(lines)
            return f"Title: {title}\nText:\n{body}".strip()

        # Fallback
        text = doc.get("text")
        if not isinstance(text, str):
            text = ""
        return f"Title: {title}\nText:\n{text}".strip()


    def _entities_block(self, doc: Dict[str, Any], *, max_mentions_per_entity: int = 2) -> str:
        """Return a readable entities block (entity_id + type + a few mentions)."""
        entities = doc.get("entities") or {}
        lines: List[str] = []

        if isinstance(entities, dict):
            for ent_id, ent in entities.items():
                if not isinstance(ent_id, str) or not isinstance(ent, dict):
                    continue
                ent_type = ent.get("type", "")
                mentions = ent.get("mentions", [])
                if not isinstance(mentions, list):
                    mentions = [mentions]

                shown = 0
                if mentions:
                    for m in mentions:
                        if not isinstance(m, dict):
                            continue
                        trig = m.get("trigger_word") or m.get("text") or m.get("name") or ""
                        sent_id = m.get("sent_id") or m.get("sentence_id")
                        offset = m.get("offset") or m.get("span")
                        lines.append(
                            f"{ent_id}\tTYPE={ent_type}\tTRIGGER={trig}\tSENT_ID={sent_id}\tOFFSET={offset}"
                        )
                        shown += 1
                        if shown >= max_mentions_per_entity:
                            break
                else:
                    # still list entity if no mentions
                    lines.append(f"{ent_id}\tTYPE={ent_type}")

        return "\n".join(lines) if lines else "(no entities found)"


    def _relation_schema_block(self, relation_types: Sequence[str]) -> str:
        """Return allowed relation types as a bullet list."""
        if not relation_types:
            return "(no relation types provided)"
        return "\n".join(f"- {r}" for r in relation_types)


    def _few_shot_block(self, few_shots: List[Dict[str, Any]]) -> str:
        """Optional few-shot demonstrations block (kept simple + robust)."""
        if not few_shots:
            return ""
        parts: List[str] = ["## Few-shot demonstrations (gold)"]
        for i, ex in enumerate(few_shots, start=1):
            rels = ex.get("relations") if isinstance(ex.get("relations"), dict) else {}
            parts += [
                f"### Demo {i}",
                "#### Document",
                self._doc_text_block(ex),
                "",
                "#### Entities",
                self._entities_block(ex),
                "",
                "#### Gold relations (JSON)",
                json.dumps(rels, ensure_ascii=False),
                "",
            ]
        return "\n".join(parts)

    def _build_entity_surface_map(self, doc: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Returns:
        - ent_id_to_best_surface: entity_id -> best surface form (trigger/text)
        - surface_to_ent_id: normalized surface -> entity_id
        """
        ent_id_to_best: Dict[str, str] = {}
        surface_to_id: Dict[str, str] = {}

        entities = doc.get("entities") or {}
        if not isinstance(entities, dict):
            return ent_id_to_best, surface_to_id

        def norm(s: str) -> str:
            return re.sub(r"\s+", " ", (s or "").strip().lower())

        for ent_id, ent in entities.items():
            if not isinstance(ent_id, str) or not isinstance(ent, dict):
                continue
            mentions = ent.get("mentions", [])
            if not isinstance(mentions, list):
                mentions = [mentions]

            # pick first good surface
            best = ""
            for m in mentions:
                if not isinstance(m, dict):
                    continue
                surf = m.get("trigger_word") or m.get("text") or m.get("name") or ""
                surf = str(surf).strip()
                if surf:
                    best = surf
                    break

            if not best:
                # fallback: sometimes entity has direct label/name
                best = str(ent.get("text") or ent.get("name") or ent.get("label") or "").strip()

            if best:
                ent_id_to_best[ent_id] = best
                surface_to_id.setdefault(norm(best), ent_id)

        return ent_id_to_best, surface_to_id


    def _match_to_entity_id(self, x: Any, surface_to_id: Dict[str, str]) -> Optional[str]:
        """
        Map a predicted endpoint (entity id or surface string) to a known entity id.
        """
        if x is None:
            return None
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return None
            # direct ID match
            if s in surface_to_id.values():
                return s
            # normalized surface match
            key = re.sub(r"\s+", " ", s.lower())
            if key in surface_to_id:
                return surface_to_id[key]
        return None


    def _normalize_pred_endpoints_to_entity_ids(
        self,
        doc: Dict[str, Any],
        pred: Dict[str, Any],
        relation_types: Sequence[str],
        *,
        keep_debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Ensures:
        - keys are restricted to relation_types
        - each value is list of [HEAD_ID, TAIL_ID] where both are valid entity IDs in doc
        """
        _, surface_to_id = self._build_entity_surface_map(doc)

        out: Dict[str, Any] = {}
        debug: Dict[str, Any] = {"dropped": []} if keep_debug else {}

        for r in relation_types:
            pairs = pred.get(r, [])
            if not isinstance(pairs, list):
                pairs = []
            norm_pairs: List[List[str]] = []

            for item in pairs:
                # allow ["E1","E2"] or {"head":..,"tail":..}
                h_raw = t_raw = None
                if isinstance(item, list) and len(item) >= 2:
                    h_raw, t_raw = item[0], item[1]
                elif isinstance(item, dict):
                    h_raw = item.get("head") or item.get("h")
                    t_raw = item.get("tail") or item.get("t")
                else:
                    if keep_debug:
                        debug["dropped"].append({"rel": r, "reason": "bad_pair_format", "pair": item})
                    continue

                h = self._match_to_entity_id(h_raw, surface_to_id)
                t = self._match_to_entity_id(t_raw, surface_to_id)
                if h is None or t is None:
                    if keep_debug:
                        debug["dropped"].append({"rel": r, "reason": "unmatched_endpoint", "pair": [h_raw, t_raw]})
                    continue
                if h == t:
                    # usually skip self-relations unless your datasets allow them
                    if keep_debug:
                        debug["dropped"].append({"rel": r, "reason": "self_relation", "pair": [h, t]})
                    continue

                norm_pairs.append([h, t])

            out[r] = norm_pairs

        if keep_debug:
            out["_debug_normalize"] = debug
        return out