# ragtree/processing/rag/strategies/agentic_hybrid_relations.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ragtree.processing.rag.base_strategy import BaseRelationStrategy
from ragtree.ontologies.retrieval.subontology import SubOntologyRetriever, SubOntologyFragment


DEFAULT_FALLBACK_RELATION_TYPE = "causal_relation"


@dataclass
class AgenticHybridParams:
    include_ontology_structured: bool = True
    include_ontology_ttl: bool = False
    kg_max_triples: int = 40
    max_sentences_in_prompt: Optional[int] = None
    max_llm_calls: int = 1  # keep it 1 by default (you requested low LLM usage)


class AgenticHybridRelationStrategy(BaseRelationStrategy):
    """
    Single-agent "hybrid" RAG for DocRE:
      - Tool A: ontology fragment (GrOWL-style) via SubOntologyRetriever
      - Tool B: KG triples context (BYOKG-style) from prebuilt KG artifacts
      - One-shot prompting by default (1 LLM call per doc)
      - Optional few-shot demonstrations (passed via predict_kwargs)

    The runner is expected to inject:
      - doc["ontology_links"] (from ontology-linking artifact) OR strategy returns empty
      - doc["_kg_context"]["triples"] (from KG artifact) OR empty list
    """

    def __init__(
        self,
        llm_config,
        *,
        retriever: SubOntologyRetriever,
        ontology_key: str,
        linking_method: str,
        params: Optional[AgenticHybridParams] = None,
    ) -> None:
        super().__init__(llm_config)
        self.retriever = retriever
        self.ontology_key = ontology_key
        self.linking_method = linking_method
        self.params = params or AgenticHybridParams()

    # ----------------------------
    # Relation types inference
    # ----------------------------
    def _infer_relation_types_from_doc(self, doc: Dict[str, Any]) -> List[str]:
        rels = doc.get("relations")
        if isinstance(rels, dict):
            keys = list(rels.keys())
            if keys:
                return keys
        return [DEFAULT_FALLBACK_RELATION_TYPE]

    # ----------------------------
    # Robust JSON parsing
    # ----------------------------
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

    # ----------------------------
    # Prompt assembly
    # ----------------------------
    def _doc_text_block(self, doc: Dict[str, Any]) -> str:
        title = doc.get("title", "")
        sentences = doc.get("sentences")
        if isinstance(sentences, list) and all(isinstance(s, str) for s in sentences):
            sents = sentences
            if self.params.max_sentences_in_prompt is not None:
                sents = sents[: self.params.max_sentences_in_prompt]
            text = "\n".join(f"- {s}" for s in sents)
        else:
            text = str(doc.get("text", ""))
        return f"Title: {title}\nText:\n{text}"

    def _entities_block(self, doc: Dict[str, Any]) -> str:
        entities = doc.get("entities", {})
        lines: List[str] = []
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
                    lines.append(
                        f"{ent_id}\tTYPE={ent_type}\tTRIGGER={trig}\tSENT_ID={sent_id}\tOFFSET={offset}"
                    )
                    shown += 1
                    if shown >= 2:
                        break
        if not lines:
            lines = ["(no entities found)"]
        return "\n".join(lines)

    def _relation_schema_block(self, relation_types: Sequence[str]) -> str:
        return "\n".join(f"- {r}" for r in relation_types)

    def _ontology_context_block(self, doc: Dict[str, Any]) -> str:
        ontology_links = doc.get("ontology_links")
        if ontology_links is None:
            return "(no ontology_links found for this document)"

        fragment: SubOntologyFragment = self.retriever.retrieve(
            ontology_links=ontology_links,
            method=self.linking_method,
            params={},
        )

        parts: List[str] = []
        if self.params.include_ontology_structured:
            parts.append("### Sub-ontology fragment (structured JSON)")
            parts.append(json.dumps(fragment.to_dict(), ensure_ascii=False, indent=2))

        if self.params.include_ontology_ttl:
            parts.append("### Sub-ontology fragment (TTL)")
            parts.append(fragment.to_ttl())

        return "\n".join(parts) if parts else "(ontology fragment disabled by params)"

    def _kg_context_block(self, doc: Dict[str, Any], relation_types: Sequence[str]) -> str:
        kg_ctx = doc.get("_kg_context", {})
        triples = []
        if isinstance(kg_ctx, dict):
            triples = kg_ctx.get("triples", []) or []

        # keep it compact
        kept: List[str] = []
        n = 0
        for t in triples:
            if n >= self.params.kg_max_triples:
                break
            if isinstance(t, dict):
                h = t.get("h") or t.get("head")
                r = t.get("r") or t.get("rel") or t.get("relation")
                tail = t.get("t") or t.get("tail")
                ev = t.get("evidence")
                kept.append(f"({h}) -[{r}]-> ({tail})" + (f" | ev={ev}" if ev else ""))
                n += 1
            elif isinstance(t, (list, tuple)) and len(t) >= 3:
                kept.append(f"({t[0]}) -[{t[1]}]-> ({t[2]})")
                n += 1

        if not kept:
            return "(no KG triples available for this document)"

        rel_hint = ", ".join(list(relation_types)[:20])
        return "\n".join(
            [
                "### Retrieved KG triples (local KG artifact)",
                f"(Relation schema hint: {rel_hint})",
                *kept,
            ]
        )

    def _few_shot_block(self, few_shots: List[Dict[str, Any]]) -> str:
        """
        Few-shot demos are full docs with gold relations.
        We render them as compact input->output examples to condition the model.
        """
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

    def _build_messages(
        self,
        doc: Dict[str, Any],
        relation_types: Sequence[str],
        *,
        few_shots: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, str]]:
        system_msg = {"role": "system", "content": self.llm_config.system_prompt}

        user_parts: List[str] = [
            "You will extract document-level relations between the PROVIDED entity IDs only.",
            "You must output ONLY valid JSON (no extra text).",
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
            "## Tool: Ontology context (GrOWL-style)",
            self._ontology_context_block(doc),
            "",
            "## Tool: KG context (BYOKG-style, local triples)",
            self._kg_context_block(doc, relation_types),
            "",
            "## Output format (JSON only, no extra text)",
            "Return a JSON object whose keys are the allowed relation types, and values are lists of [HEAD_ID, TAIL_ID] pairs.",
            "Example:",
            '{"REL_TYPE": [["E1","E2"]], "OTHER": []}',
        ]

        return [system_msg, {"role": "user", "content": "\n".join(user_parts)}]

    # ----------------------------
    # Main API
    # ----------------------------
    def predict_relations(
        self,
        doc: Dict[str, Any],
        relation_types: Optional[Sequence[str]] = None,
        *,
        few_shots: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, List[List[str]]]:
        rel_types = list(relation_types) if relation_types else self._infer_relation_types_from_doc(doc)

        # If no ontology_links and no KG context, still run (model can output empty)
        messages = self._build_messages(doc, rel_types, few_shots=few_shots)

        raw = self._call_llm(messages)
        parsed = self._extract_json_object(raw)
        if not isinstance(parsed, dict):
            return {r: [] for r in rel_types}

        # Normalize output schema (entity IDs only, keys match schema)
        normalized = self._normalize_relation_dict(parsed, rel_types)
        return normalized
