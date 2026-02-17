# ragtree/processing/rag/strategies/triple_kg_rag_relations.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from ragtree.processing.rag.base_strategy import BaseRelationStrategy
from ragtree.processing.kg_rag.triple_kg_retriever import TripleKGRetriever, SimpleKGFragment


DEFAULT_FALLBACK_RELATION_TYPE = "causal_relation"


class TripleKGRagRelationStrategy(BaseRelationStrategy):
    """
    Simple triple-based KG-RAG DocRE strategy (no KG-Linker, no multi-retriever toolkit).
    Uses:
      - doc entity IDs as KG seed nodes
      - BFS up to N hops
      - token-overlap scoring to keep top-K triples
    """

    def __init__(
        self,
        llm_config,
        *,
        retriever: TripleKGRetriever,
        max_sentences_in_prompt: Optional[int] = None,
        max_triples_in_text: Optional[int] = 80,
    ) -> None:
        super().__init__(llm_config)
        self.retriever = retriever
        self.max_sentences_in_prompt = max_sentences_in_prompt
        self.max_triples_in_text = max_triples_in_text

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
        fragment: SimpleKGFragment,
    ) -> List[Dict[str, str]]:
        system_msg = {"role": "system", "content": self.llm_config.system_prompt}

        title = doc.get("title", "")

        # doc text
        sentences = doc.get("sentences")
        if isinstance(sentences, list) and all(isinstance(s, str) for s in sentences):
            sents = sentences
            if self.max_sentences_in_prompt is not None:
                sents = sents[: self.max_sentences_in_prompt]
            doc_text = "\n".join(f"- {s}" for s in sents)
        else:
            doc_text = str(doc.get("text", ""))

        # entities
        entities = doc.get("entities", {})
        entity_lines: List[str] = []
        if isinstance(entities, dict):
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
        kg_text = fragment.to_text(max_lines=self.max_triples_in_text)

        user_parts: List[str] = [
            "You will extract document-level relations between the PROVIDED entity IDs only.",
            "You MUST output ONLY valid JSON (no markdown, no explanations).",
            "You MUST use ONLY the PROVIDED entity IDs in output pairs (no literals).",
            "",
            "## Document Title",
            str(title),
            "",
            "## Document",
            doc_text,
            "",
            "## Entities (IDs are canonical  use them in output)",
            "\n".join(entity_lines),
            "",
            "## Allowed relation types (output keys must match these exactly)",
            rel_schema,
            "",
            "## KG Context (top retrieved triples)",
            kg_text,
            "",
            "## KG Context (structured JSON)",
            kg_json,
            "",
            "## Output format (JSON only, no extra text)",
            "Return a JSON object whose keys are the allowed relation types, and values are lists of [HEAD_ID, TAIL_ID] pairs.",
            'Example: {"REL_TYPE": [["E1","E2"]], "OTHER": []}',
        ]

        return [system_msg, {"role": "user", "content": "\n".join(user_parts)}]

    def predict_relations(
        self,
        doc: Dict[str, Any],
        relation_types: Optional[Sequence[str]] = None,
        *,
        few_shots: Optional[List[Dict[str, Any]]] = None,  # kept for interface consistency
    ) -> Dict[str, List[List[str]]]:
        rel_types = list(relation_types) if relation_types else self._infer_relation_types_from_doc(doc)

        fragment = self.retriever.retrieve(doc)
        msgs = self._build_messages(doc, rel_types, fragment)

        raw = self._call_llm(msgs)
        pred = self._parse_relations_json(raw, rel_types)  # provided by BaseRelationStrategy
        return pred
