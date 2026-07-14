# ragtree/processing/rag/strategies/kg_rag_relations.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from ragtree.processing.rag.base_strategy import BaseRelationStrategy
from ragtree.processing.kg_rag.kg_retriever import KGRetriever, KGFragment


DEFAULT_FALLBACK_RELATION_TYPE = "causal_relation"


class KGRagRelationStrategy(BaseRelationStrategy):
    """
    KG-RAG DocRE strategy.

    Inputs:
      - doc["sentences"] or doc["text"]
      - doc["entities"] (IDs + mentions)
      - (optional) doc["ontology_links"] if you want hybrid OG-RAG + KG-RAG later

    External:
      - KGRetriever (built on a local KG)

    Output:
      - dict { rel_type: [[head_id, tail_id], ...], ... }
    """

    def __init__(
        self,
        llm_config,
        *,
        retriever: KGRetriever,
        max_sentences_in_prompt: Optional[int] = None,
    ) -> None:
        super().__init__(llm_config)
        self.retriever = retriever
        self.max_sentences_in_prompt = max_sentences_in_prompt

    def _infer_relation_types_from_doc(self, doc: Dict[str, Any]) -> List[str]:
        rels = doc.get("relations")
        if isinstance(rels, dict):
            keys = list(rels.keys())
            if keys:
                return keys
        return [DEFAULT_FALLBACK_RELATION_TYPE]

    def _build_messages(
        self,
        doc: Dict[str, Any],
        relation_types: Sequence[str],
        fragment: KGFragment,
    ) -> List[Dict[str, str]]:
        system_msg = {"role": "system", "content": self.llm_config.system_prompt}

        title = doc.get("title", "")

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

        rel_schema = "\n".join(f"- {r}" for r in relation_types)

        kg_json = json.dumps(fragment.to_dict(), ensure_ascii=False, indent=2)
        kg_text = fragment.to_text(max_triples=200)

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
            "## Knowledge Graph retrieval context",
            "### KG fragment (structured JSON)",
            kg_json,
            "",
            "### KG fragment (triples)",
            kg_text,
            "",
            "## Output format (JSON only, no extra text)",
            "Return a JSON object whose keys are the allowed relation types, and values are lists of [HEAD_ID, TAIL_ID] pairs.",
            "Example:",
            '{"REL_TYPE": [["E1","E2"]], "OTHER": []}',
        ]

        user_msg = {"role": "user", "content": "\n".join(user_parts)}
        return [system_msg, user_msg]

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

    def predict_relations(
        self,
        doc: Dict[str, Any],
        relation_types: Optional[Sequence[str]] = None,
    ) -> Dict[str, List[List[str]]]:
        rel_types = list(relation_types) if relation_types else self._infer_relation_types_from_doc(doc)

        fragment = self.retriever.retrieve(doc, rel_types=rel_types)
        messages = self._build_messages(doc, rel_types, fragment)
        raw = self._call_llm(messages)

        parsed = self._extract_json_object(raw)
        if not isinstance(parsed, dict):
            return {r: [] for r in rel_types}

        normalized = self._normalize_relation_dict(parsed, rel_types)
        return normalized
