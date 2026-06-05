# ragtree/processing/rag/strategies/chunk_orag_relations.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

from ragtree.processing.rag.base_strategy import BaseRelationStrategy
from ragtree.ontologies.retrieval.chunk_orag_retriever import ChunkORAGRetriever, ChunkORAGChunk


DEFAULT_FALLBACK_RELATION_TYPE = "causal_relation"


@dataclass
class ChunkORAGParams:
    top_k: int = 8
    max_ctx_chars: int = 8000

    # Few-shot formatting controls (budget)
    max_fewshot_sentences: int = 3
    max_fewshot_entities: int = 12
    max_fewshot_pairs_per_rel: int = 6

    # Prompt budget controls
    max_sentences_in_prompt: Optional[int] = None


class ChunkORAGRelationStrategy(BaseRelationStrategy):
    """
    Chunk-O-RAG relation extraction strategy.

    - Retrieves ontology chunks using ChunkORAGRetriever.
    - Supports few-shot examples (compact, like GrowlRAG).
    - Normalizes predicted endpoints to entity IDs (Step 6).
    """

    def __init__(
        self,
        llm_config,
        *,
        retriever: ChunkORAGRetriever,
        params: Optional[ChunkORAGParams] = None,
    ) -> None:
        super().__init__(llm_config)
        self.retriever = retriever
        self.params = params or ChunkORAGParams()

    # -------------------------
    # Schema inference
    # -------------------------

    def _infer_relation_types_from_doc(self, doc: Dict[str, Any]) -> List[str]:
        rels = doc.get("relations")
        if isinstance(rels, dict):
            keys = list(rels.keys())
            if keys:
                return keys
        return [DEFAULT_FALLBACK_RELATION_TYPE]

    # -------------------------
    # Blocks used in prompt
    # -------------------------

    def _doc_text_block(self, doc: Dict[str, Any]) -> str:
        """Prefer sentences list; fallback to doc['text']."""
        title = str(doc.get("title", "") or "")
        sentences = doc.get("sentences")
        if isinstance(sentences, list) and all(isinstance(s, str) for s in sentences):
            sents = sentences
            if self.params.max_sentences_in_prompt is not None:
                sents = sents[: self.params.max_sentences_in_prompt]
            body = "\n".join(f"- {s}" for s in sents)
        else:
            body = str(doc.get("text", "") or "")
        if title.strip():
            return f"Title: {title}\nText:\n{body}"
        return f"Text:\n{body}"

    def _entities_block(self, doc: Dict[str, Any]) -> str:
        """Compact entities block (IDs + 12 mentions)."""
        entities = doc.get("entities", {})
        if not isinstance(entities, dict) or not entities:
            return "(no entities found)"

        lines: List[str] = []
        for ent_id, ent in entities.items():
            if not isinstance(ent, dict):
                continue
            ent_type = ent.get("type", "")
            mentions = ent.get("mentions", [])
            if not isinstance(mentions, list):
                mentions = [mentions]

            shown = 0
            for m in mentions:
                if not isinstance(m, dict):
                    continue
                trig = m.get("trigger_word") or m.get("text") or ""
                sent_id = m.get("sent_id")
                offset = m.get("offset") or m.get("span")
                lines.append(f"{ent_id}\tTYPE={ent_type}\tTRIGGER={trig}\tSENT_ID={sent_id}\tOFFSET={offset}")
                shown += 1
                if shown >= 2:
                    break
        return "\n".join(lines) if lines else "(no entities found)"

    def _relation_schema_block(self, relation_types: Sequence[str]) -> str:
        """Allowed relation types list."""
        return "\n".join(f"- {r}" for r in relation_types)

    # -------------------------
    # Few-shot formatting (GrowlRAG style)
    # -------------------------

    def _format_entities_block(self, entities: Dict[str, Any], *, max_entities: int) -> str:
        lines: List[str] = []
        if not isinstance(entities, dict):
            return "(no entities found)"
        for ent_id, ent in entities.items():
            if not isinstance(ent, dict):
                continue
            ent_type = ent.get("type", "")
            mentions = ent.get("mentions", [])
            if not isinstance(mentions, list):
                mentions = [mentions]
            for m in mentions[:1]:
                if not isinstance(m, dict):
                    continue
                trig = m.get("trigger_word") or m.get("text") or ""
                sent_id = m.get("sent_id")
                offset = m.get("offset") or m.get("span")
                lines.append(f"{ent_id}\tTYPE={ent_type}\tTRIGGER={trig}\tSENT_ID={sent_id}\tOFFSET={offset}")
            if len(lines) >= max_entities:
                break
        return "\n".join(lines) if lines else "(no entities found)"

    def _format_gold_relations_block(
        self,
        relations: Dict[str, Any],
        allowed_relation_types: Sequence[str],
        *,
        max_pairs_per_rel: int,
    ) -> str:
        out: Dict[str, List[List[str]]] = {r: [] for r in allowed_relation_types}
        if not isinstance(relations, dict):
            return json.dumps(out, ensure_ascii=False)
        for r in allowed_relation_types:
            pairs = relations.get(r, [])
            if not isinstance(pairs, list):
                continue
            kept: List[List[str]] = []
            for pair in pairs:
                if isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, str) for x in pair):
                    kept.append([pair[0], pair[1]])
                if len(kept) >= max_pairs_per_rel:
                    break
            out[r] = kept
        return json.dumps(out, ensure_ascii=False)

    def _format_few_shots_block(
        self,
        few_shots: Sequence[Dict[str, Any]],
        allowed_relation_types: Sequence[str],
    ) -> str:
        blocks: List[str] = []
        idx = 1
        for ex in few_shots:
            if not isinstance(ex, dict):
                continue
            rels = ex.get("relations")
            if not isinstance(rels, dict) or not rels:
                continue

            title = ex.get("title", "")
            sentences = ex.get("sentences")
            if isinstance(sentences, list) and all(isinstance(s, str) for s in sentences):
                sents = sentences[: self.params.max_fewshot_sentences]
                ex_text = "\n".join(f"- {s}" for s in sents)
            else:
                ex_text = str(ex.get("text", ""))[:1000]

            entities = ex.get("entities", {})
            entities_block = self._format_entities_block(entities, max_entities=self.params.max_fewshot_entities)
            rels_block = self._format_gold_relations_block(
                rels,
                allowed_relation_types,
                max_pairs_per_rel=self.params.max_fewshot_pairs_per_rel,
            )

            blocks.append(
                "\n".join(
                    [
                        f"### Example {idx}",
                        f"Title: {title}",
                        "Text:",
                        ex_text,
                        "",
                        "Entities (use these IDs):",
                        entities_block,
                        "",
                        "Gold output JSON:",
                        rels_block,
                    ]
                )
            )
            idx += 1
        return "\n\n".join(blocks).strip()

    # -------------------------
    # Retrieval formatting
    # -------------------------

    def _format_chunks(self, chunks: Sequence[Any]) -> str:
        """
        Robust formatter:
        - accepts ChunkORAGChunk
        - accepts dict
        - accepts str (fallback)
        """
        lines: List[str] = []
        for c in chunks:
            if isinstance(c, ChunkORAGChunk):
                prefix = f"[{c.subject_label} | id={c.chunk_id} | score={c.score:.4f}]"
                txt = (c.text or "").strip().replace("\n", " ")
            elif isinstance(c, dict):
                prefix = f"[{c.get('subject_label','')} | id={c.get('chunk_id','')} | score={c.get('score','')}]"
                txt = str(c.get("text", "")).strip().replace("\n", " ")
            else:
                prefix = "[chunk]"
                txt = str(c).strip().replace("\n", " ")
            lines.append(f"- {prefix} {txt}")

        ctx = "\n".join(lines)
        if len(ctx) > self.params.max_ctx_chars:
            ctx = ctx[: self.params.max_ctx_chars] + "\n... (truncated)"
        return ctx if ctx else "(no ontology chunks retrieved)"

    # -------------------------
    # JSON parsing
    # -------------------------

    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        if not isinstance(text, str) or not text.strip():
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
    # Step 6 endpoint normalization (GrowlRAG style)
    # -------------------------

    def _build_literal_to_entity_index(self, doc: Dict[str, Any]) -> Dict[str, str]:
        entities = doc.get("entities") or {}
        hits: Dict[str, Set[str]] = {}

        def norm(s: str) -> str:
            return re.sub(r"\s+", " ", s.strip().lower())

        for ent_id, ent in (entities.items() if isinstance(entities, dict) else []):
            if not isinstance(ent, dict):
                continue
            mentions = ent.get("mentions") or []
            if not isinstance(mentions, list):
                continue
            for m in mentions:
                if not isinstance(m, dict):
                    continue
                texts: List[str] = []
                if m.get("trigger_word"):
                    texts.append(str(m["trigger_word"]))
                if m.get("text"):
                    texts.append(str(m["text"]))
                for t in texts:
                    k = norm(t)
                    if not k:
                        continue
                    hits.setdefault(k, set()).add(ent_id)

        out: Dict[str, str] = {}
        for k, ids in hits.items():
            if len(ids) == 1:
                out[k] = next(iter(ids))
        return out

    def _maybe_map_literal_to_entity(self, lit: str, idx: Dict[str, str]) -> Optional[str]:
        key = re.sub(r"\s+", " ", str(lit).strip().lower())
        if not key:
            return None
        return idx.get(key)

    def _normalize_pred_endpoints_to_entity_ids(
        self,
        doc: Dict[str, Any],
        raw_pred: Dict[str, Any],
        relation_types: Sequence[str],
        *,
        keep_debug: bool = False,
    ) -> Dict[str, List[List[str]]]:
        """
        Normalize relation endpoints to ids.

        Important behavior:
        -------------------
        1. If an endpoint already looks like an ID (e.g. 'Event_xxx', 'Entity_xxx'),
        it is kept as-is.
        2. If not, we optionally try to resolve it from document aliases/names.
        3. If we still cannot resolve it, we keep the original string instead of
        dropping the pair.

        This avoids losing valid pairs when the LLM already outputs canonical ids.
        """
        raw_by_canonical = self._build_canonical_relation_map(raw_pred)

        def _norm_text(x: Any) -> str:
            if not isinstance(x, str):
                return ""
            return " ".join(x.strip().split())

        def _looks_like_id(x: Any) -> bool:
            """
            Heuristic: preserve endpoints that already look like internal ids.
            Examples:
                Event_b102c3c61e75
                Entity_123
                ENT_45
            """
            if not isinstance(x, str):
                return False
            x = x.strip()
            if not x:
                return False

            prefixes = (
                "Event_",
                "Entity_",
                "ENT_",
                "EV_",
                "E_",
            )
            return x.startswith(prefixes)

        def _add_alias(alias_map: Dict[str, str], alias: Any, entity_id: str) -> None:
            alias_norm = _norm_text(alias)
            if alias_norm and alias_norm not in alias_map:
                alias_map[alias_norm] = entity_id

        # Optional alias map from document, used only as fallback
        alias_to_id: Dict[str, str] = {}
        alias_to_id_lower: Dict[str, str] = {}

        entities = doc.get("entities", [])
        if isinstance(entities, list):
            for ent in entities:
                if not isinstance(ent, dict):
                    continue

                entity_id = ent.get("id")
                if not isinstance(entity_id, str) or not entity_id.strip():
                    continue

                entity_id = entity_id.strip()

                for key in ("name", "text", "mention", "label", "title"):
                    if key in ent:
                        _add_alias(alias_to_id, ent.get(key), entity_id)

                aliases = ent.get("aliases", [])
                if isinstance(aliases, list):
                    for alias in aliases:
                        _add_alias(alias_to_id, alias, entity_id)

        for alias, entity_id in alias_to_id.items():
            alias_to_id_lower[alias.lower()] = entity_id

        def _resolve_endpoint(pred_value: Any) -> Optional[str]:
            """
            Resolution policy:
            - if already an id -> keep it
            - else try alias lookup
            - else keep original string if it is non-empty
            """
            if not isinstance(pred_value, str):
                return None

            candidate = pred_value.strip()
            if not candidate:
                return None

            # Most important fix: preserve already-canonical ids
            if _looks_like_id(candidate):
                return candidate

            # Fallback alias matching
            candidate_norm = _norm_text(candidate)
            if candidate_norm in alias_to_id:
                return alias_to_id[candidate_norm]

            candidate_lower = candidate_norm.lower()
            if candidate_lower in alias_to_id_lower:
                return alias_to_id_lower[candidate_lower]

            # Final fallback: keep original string instead of dropping the pair
            return candidate

        normalized: Dict[str, List[List[str]]] = {}

        for rtype in relation_types:
            canonical_rtype = self._canonical_relation_type(rtype)
            raw_pairs = raw_by_canonical.get(canonical_rtype, [])
            clean_pairs: List[List[str]] = []

            if not isinstance(raw_pairs, list):
                normalized[rtype] = []
                continue

            for item in raw_pairs:
                if not (isinstance(item, list) and len(item) == 2):
                    continue

                head_raw, tail_raw = item[0], item[1]
                head_id = _resolve_endpoint(head_raw)
                tail_id = _resolve_endpoint(tail_raw)

                if head_id is not None and tail_id is not None:
                    clean_pairs.append([head_id, tail_id])
                elif keep_debug:
                    print(
                        f"[DEBUG] Dropped pair for relation {rtype!r}: "
                        f"head={head_raw!r}->{head_id!r}, "
                        f"tail={tail_raw!r}->{tail_id!r}"
                    )

            normalized[rtype] = clean_pairs

        return normalized

    # -------------------------
    # Prompt assembly
    # -------------------------

    def _build_extraction_messages(
        self,
        doc: Dict[str, Any],
        relation_types: Sequence[str],
        *,
        ontology_chunk_ctx: str,
        few_shots: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[Dict[str, str]]:
        system_msg = {"role": "system", "content": self.llm_config.system_prompt}

        fewshot_block = ""
        if few_shots:
            fewshot_block = self._format_few_shots_block(few_shots, relation_types)

        user_parts: List[str] = [
            "You will extract document-level relations between the PROVIDED entity IDs only.",
            "You MUST output ONLY valid JSON (no markdown, no explanations).",
            "You MUST use ONLY the PROVIDED entity IDs in output pairs (no literals).",
            "",
        ]

        if fewshot_block:
            user_parts += [
                "## Few-shot examples (optional guidance)",
                "Follow the pattern exactly: output JSON with allowed relation types and ID pairs.",
                fewshot_block,
                "",
            ]

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
            "## Tool: Chunk-O-RAG ontology context (retrieved chunks)",
            ontology_chunk_ctx,
            "",
            "## Output format (JSON only, no extra text)",
            "Return a JSON object whose keys are the allowed relation types, and values are lists of [HEAD_ID, TAIL_ID] pairs.",
            "Example:",
            '{"REL_TYPE": [["E1","E2"]], "OTHER": []}',
        ]

        return [system_msg, {"role": "user", "content": "\n".join(user_parts)}]

    # -------------------------
    # Main API
    # -------------------------

    def _build_query(self, doc: Dict[str, Any]) -> str:
        """Short query for retrieval."""
        parts: List[str] = []
        title = doc.get("title")
        if isinstance(title, str) and title.strip():
            parts.append(title.strip())

        sents = doc.get("sentences")
        if isinstance(sents, list) and sents and all(isinstance(x, str) for x in sents):
            parts.extend(sents[:3])
        else:
            txt = doc.get("text")
            if isinstance(txt, str) and txt.strip():
                parts.append(txt[:800])

        # Add a few entity triggers (helps retrieval)
        ents = doc.get("entities") or {}
        if isinstance(ents, dict):
            count = 0
            for _, ent in ents.items():
                if not isinstance(ent, dict):
                    continue
                mentions = ent.get("mentions") or []
                if not isinstance(mentions, list):
                    continue
                if mentions and isinstance(mentions[0], dict):
                    trig = mentions[0].get("trigger_word") or mentions[0].get("text")
                    if trig:
                        parts.append(str(trig))
                        count += 1
                        if count >= 6:
                            break

        return "\n".join([p for p in parts if p]).strip()

    def _canonical_relation_type(self, rtype: str) -> str:
        """
        Normalize a relation type label to its canonical property id.

        Examples
        --------
        "P571" -> "P571"
        "P571 : inception" -> "P571"
        """
        if not isinstance(rtype, str):
            return ""
        return rtype.split(":", 1)[0].strip()


    def _build_canonical_relation_map(
        self,
        raw: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build a canonical-key view of a raw relation dict.

        Supports either:
            {"P571": ...}
        or:
            {"P571 : inception": ...}
        """
        canonical: Dict[str, Any] = {}

        if not isinstance(raw, dict):
            return canonical

        for key, value in raw.items():
            if not isinstance(key, str):
                continue
            canonical_key = self._canonical_relation_type(key)
            if canonical_key:
                canonical[canonical_key] = value

        return canonical

    def predict_relations(
        self,
        doc: Dict[str, Any],
        relation_types: Optional[Sequence[str]] = None,
        *,
        few_shots: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, List[List[str]]]:
        rel_types = list(relation_types) if relation_types else self._infer_relation_types_from_doc(doc)

        query = self._build_query(doc)
        chunks = self.retriever.retrieve(query, top_k=self.params.top_k)
        ctx = self._format_chunks(chunks)

        msgs = self._build_extraction_messages(
            doc,
            rel_types,
            ontology_chunk_ctx=ctx,
            few_shots=few_shots,
        )
        raw = self._call_llm(msgs)

        parsed = self._extract_json_object(raw)

        if not isinstance(parsed, dict):
            return {r: [] for r in rel_types}

        parsed_entity_only = self._normalize_pred_endpoints_to_entity_ids(
            doc,
            parsed,
            rel_types,
            keep_debug=True,
        )

        final_output = self._normalize_relation_dict(parsed_entity_only, rel_types)

        return final_output