# ragtree/processing/rag/strategies/triple_kg_rag_relations.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from ragtree.processing.rag.base_strategy import BaseRelationStrategy
from ragtree.processing.kg_rag.triple_kg_retriever import TripleKGRetriever, SimpleKGFragment

DEFAULT_FALLBACK_RELATION_TYPE = "causal_relation"


class TripleKGRagRelationStrategy(BaseRelationStrategy):
    def __init__(
        self,
        llm_config,
        *,
        retriever: TripleKGRetriever,
        max_sentences_in_prompt: Optional[int] = None,
        max_triples_in_text: int = 80,
    ) -> None:
        super().__init__(llm_config)
        self.retriever = retriever
        self.max_sentences_in_prompt = max_sentences_in_prompt
        self.max_triples_in_text = max_triples_in_text

    def _infer_relation_types_from_doc(self, doc: Dict[str, Any]) -> List[str]:
        rels = doc.get("relations")
        if isinstance(rels, dict) and rels:
            return list(rels.keys())
        return [DEFAULT_FALLBACK_RELATION_TYPE]

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

    def _build_messages(
        self,
        doc: Dict[str, Any],
        relation_types: Sequence[str],
        fragment: SimpleKGFragment,
        *,
        few_shots: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, str]]:
        system_msg = {"role": "system", "content": self.llm_config.system_prompt}

        title = str(doc.get("title", ""))

        sentences = doc.get("sentences")
        if isinstance(sentences, list) and all(isinstance(s, str) for s in sentences):
            sents = sentences[: self.max_sentences_in_prompt] if self.max_sentences_in_prompt else sentences
            doc_text = "\n".join(f"- {s}" for s in sents)
        else:
            doc_text = str(doc.get("text", ""))

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

        kg_text = fragment.to_text(max_lines=self.max_triples_in_text)

        user_parts: List[str] = [
            "You will extract document-level relations between the PROVIDED entity IDs only.",
            "You MUST output ONLY valid JSON (no markdown, no explanations).",
            "You MUST use ONLY the PROVIDED entity IDs in output pairs (no literals).",
            "",
        ]

        # Few-shot block (same helper you already use elsewhere)
        if few_shots:
            user_parts.append(self._few_shot_block(few_shots))

        user_parts += [
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
            "## Tool: triple_kg_rag KG evidence (retrieved triples)",
            kg_text,
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
        few_shots: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, List[List[str]]]:
        rel_types = list(relation_types) if relation_types else self._infer_relation_types_from_doc(doc)

        fragment = self.retriever.retrieve(doc)
        msgs = self._build_messages(doc, rel_types, fragment, few_shots=few_shots)
        raw = self._call_llm(msgs)

        parsed = self._extract_json_object(raw)
        if not isinstance(parsed, dict):
            return {r: [] for r in rel_types}

        try:
            parsed = self._normalize_pred_endpoints_to_entity_ids(doc, parsed, rel_types, keep_debug=True)
        except Exception:
            pass

        return self._normalize_relation_dict(parsed, rel_types)
    
    # --- ADD THIS METHOD INSIDE class TripleKGRagRelationStrategy(BaseRelationStrategy) ---

    def _few_shot_block(self, few_shots: List[Dict[str, Any]], *, max_shots: int = 3) -> str:
        """
        Local few-shot formatter (because BaseRelationStrategy in your repo does not provide _few_shot_block).
        Includes (doc + entities + gold relations) for each example.
        """
        blocks: List[str] = []
        for i, ex in enumerate(few_shots[:max_shots], start=1):
            # text
            title = str(ex.get("title", ""))
            sents = ex.get("sentences")
            if isinstance(sents, list) and all(isinstance(s, str) for s in sents):
                text = "\n".join(f"- {s}" for s in sents[:15])  # cap a bit
            else:
                text = str(ex.get("text", ""))[:1500]

            # entities
            ents = ex.get("entities", {})
            ent_lines: List[str] = []
            if isinstance(ents, dict):
                for ent_id, ent in ents.items():
                    if not isinstance(ent, dict):
                        continue
                    et = ent.get("type", "")
                    mentions = ent.get("mentions", [])
                    if not isinstance(mentions, list):
                        mentions = [mentions]
                    trig = ""
                    for m in mentions[:1]:
                        if isinstance(m, dict):
                            trig = m.get("trigger_word") or m.get("text") or ""
                    ent_lines.append(f"{ent_id}\tTYPE={et}\tTRIGGER={trig}")
            if not ent_lines:
                ent_lines = ["(no entities)"]

            # gold relations
            rels = ex.get("relations", {})
            if not isinstance(rels, dict):
                rels = {}
            rels_json = json.dumps(rels, ensure_ascii=False)

            blocks.append(
                "\n".join(
                    [
                        f"### Few-shot example {i}",
                        f"Title: {title}",
                        "Text:",
                        text,
                        "Entities:",
                        "\n".join(ent_lines),
                        "Gold relations (JSON):",
                        rels_json,
                    ]
                )
            )

        return "\n\n".join(blocks) + "\n"