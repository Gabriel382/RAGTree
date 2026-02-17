# ragtree/processing/rag/strategies/chunk_orag_relations.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ragtree.ontologies.retrieval.chunk_orag_retriever import ChunkORAGRetriever
from ragtree.processing.rag.base_strategy import BaseRelationStrategy


DEFAULT_FALLBACK_RELATION_TYPE = "causal_relation"


@dataclass
class ChunkORAGParams:
    # retrieval
    top_k: int = 12
    rerank_top_n: int = 6
    auto_merge: bool = True
    max_ctx_chars: int = 8000

    # query builder
    max_sentences_in_query: int = 3
    max_entity_mentions_in_query: int = 6


class ChunkORAGRelationStrategy(BaseRelationStrategy):
    """
    Chunk-O-RAG for DocRE-style relation extraction:
      - Build a query from the doc (title + first N sents + entity triggers)
      - Retrieve top ontology chunks from ChunkORAGRetriever
      - Inject into LLM extraction prompt
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

    # ---------- schema inference ----------
    def _infer_relation_types_from_doc(self, doc: Dict[str, Any]) -> List[str]:
        rels = doc.get("relations")
        if isinstance(rels, dict):
            keys = list(rels.keys())
            if keys:
                return keys
        return [DEFAULT_FALLBACK_RELATION_TYPE]

    # ---------- robust json ----------
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

    # ---------- query builder ----------
    def _build_query(self, doc: Dict[str, Any]) -> str:
        parts: List[str] = []

        title = doc.get("title")
        if isinstance(title, str) and title.strip():
            parts.append(title.strip())

        sents = doc.get("sentences")
        if isinstance(sents, list) and sents and all(isinstance(x, str) for x in sents):
            parts.extend(sents[: self.params.max_sentences_in_query])
        else:
            txt = doc.get("text")
            if isinstance(txt, str) and txt.strip():
                parts.append(txt[:800])

        # entity trigger words (if available)
        ents = doc.get("entities") or {}
        if isinstance(ents, dict):
            count = 0
            for _, ent in ents.items():
                if not isinstance(ent, dict):
                    continue
                mentions = ent.get("mentions") or []
                if not isinstance(mentions, list):
                    mentions = [mentions]
                for m in mentions[:1]:
                    if isinstance(m, dict):
                        trig = m.get("trigger_word") or m.get("text")
                        if trig:
                            parts.append(str(trig))
                            count += 1
                            if count >= self.params.max_entity_mentions_in_query:
                                break
                if count >= self.params.max_entity_mentions_in_query:
                    break

        return "\n".join([p for p in parts if p]).strip()

    # ---------- prompt ----------
    def _format_chunks(self, chunks: List[Any]) -> str:
        lines: List[str] = []
        for c in chunks:
            # include subject label for readability
            prefix = f"[{c.subject_label}]"
            t = c.text.strip().replace("\n", " ")
            lines.append(f"- {prefix} {t}")

        ctx = "\n".join(lines)
        if len(ctx) > self.params.max_ctx_chars:
            ctx = ctx[: self.params.max_ctx_chars] + "\n... (truncated)"
        return ctx if ctx else "(no chunks retrieved)"

    def _build_extraction_messages(
        self,
        doc: Dict[str, Any],
        relation_types: Sequence[str],
        *,
        ontology_chunk_ctx: str,
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
            "## Tool: Chunk-O-RAG ontology context (retrieved chunks)",
            ontology_chunk_ctx,
            "",
            "## Output format (JSON only, no extra text)",
            "Return a JSON object whose keys are the allowed relation types, and values are lists of [HEAD_ID, TAIL_ID] pairs.",
            "Example:",
            '{"REL_TYPE": [["E1","E2"]], "OTHER": []}',
        ]

        return [system_msg, {"role": "user", "content": "\n".join(user_parts)}]

    # ---------- main ----------
    def predict_relations(
        self,
        doc: Dict[str, Any],
        relation_types: Optional[Sequence[str]] = None,
        *,
        few_shots: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, List[List[str]]]:
        rel_types = list(relation_types) if relation_types else self._infer_relation_types_from_doc(doc)

        query = self._build_query(doc)
        chunks = self.retriever.retrieve(
            query,
            top_k=self.params.top_k,
            rerank_top_n=self.params.rerank_top_n,
            auto_merge=self.params.auto_merge,
        )
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

        # enforce entity-id-only endpoints if BaseRelationStrategy provides it
        try:
            parsed = self._normalize_pred_endpoints_to_entity_ids(doc, parsed, rel_types, keep_debug=True)
        except Exception:
            # fallback: just normalize keys/values format
            pass

        return self._normalize_relation_dict(parsed, rel_types)
