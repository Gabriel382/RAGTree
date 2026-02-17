# ragtree/processing/rag/strategies/marag_relations.py
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict

from ragtree.processing.rag.base_strategy import BaseRelationStrategy
from ragtree.ontologies.retrieval.subontology import SubOntologyRetriever, SubOntologyFragment

try:
    # LangGraph
    from langgraph.graph import StateGraph, END
except Exception as e:  # pragma: no cover
    StateGraph = None  # type: ignore
    END = None  # type: ignore


DEFAULT_FALLBACK_RELATION_TYPE = "causal_relation"


@dataclass
class MARAGParams:
    # Core budget
    max_llm_calls: int = 1

    # Whether to run an LLM planning step (costs 1 call)
    enable_planner: bool = False

    # Tools toggles (planner may also decide)
    enable_ontology: bool = True
    enable_kg: bool = True
    enable_web: bool = False
    enable_wikidata: bool = False

    # Retrieval limits
    kg_max_triples: int = 40
    max_sentences_in_prompt: Optional[int] = None

    # Ontology fragment rendering
    include_ontology_structured: bool = True
    include_ontology_ttl: bool = False

    # Web / Wikidata limits
    web_max_snippets: int = 3
    wikidata_max_entities: int = 2

    # Debug
    keep_debug: bool = True
    verbose: bool = False


class MARAGState(TypedDict, total=False):
    doc: Dict[str, Any]
    relation_types: List[str]
    few_shots: List[Dict[str, Any]]

    # Tool contexts
    ontology_ctx: str
    kg_ctx: str
    web_ctx: str
    wikidata_ctx: str

    # Planner
    plan: Dict[str, Any]

    # LLM usage
    llm_calls_used: int

    # Predictions
    pred_raw: Dict[str, Any]
    pred_norm: Dict[str, List[List[str]]]


class MARagRelationStrategy(BaseRelationStrategy):
    """
    MA-RAG (Multi-Agent RAG) implemented with LangGraph.

    Agents (nodes):
      - Planner (optional, LLM): decides which tools to activate
      - OntologyAgent (no LLM): uses SubOntologyRetriever from ontology_links
      - KGAgent (no LLM): formats KG triples context
      - WikidataAgent (no LLM, optional): lightweight Wikidata lookup (HTTP)
      - WebAgent (no LLM, optional): lightweight web snippets (HTTP; stub by default)
      - Extractor (LLM): final DocRE extraction with all gathered context

    Notes:
      - Default is FAST: max_llm_calls=1, enable_planner=False, web/wikidata off.
      - Few-shot is optional: if shot num is 0 => no examples in prompt.
      - Output is evaluator-compatible:
          {rel_type: [[HEAD_ID, TAIL_ID], ...], ...}
        endpoints MUST be entity IDs.
    """

    def __init__(
        self,
        llm_config,
        *,
        params: Optional[MARAGParams] = None,
        subontology_retriever: Optional[SubOntologyRetriever] = None,
        linking_method: str = "llm_embedding",
        ontology_links_by_docid: Optional[Dict[str, Any]] = None,
        kg_triples_by_docid: Optional[Dict[str, List[List[str]]]] = None,
    ) -> None:
        super().__init__(llm_config)

        if StateGraph is None:
            raise ImportError(
                "LangGraph is not installed/available. "
                "Install it with: pip install langgraph"
            )

        self.params = params or MARAGParams()
        self.subontology_retriever = subontology_retriever
        self.linking_method = linking_method

        # Optional external maps (doc_id -> artifact)
        self.ontology_links_by_docid = ontology_links_by_docid or {}
        self.kg_triples_by_docid = kg_triples_by_docid or {}

        self._graph = None
        self._log = logging.getLogger("ragtree.marag")
        if self.params.verbose:
            self._log.setLevel(logging.INFO)

    # ---------------------------
    # Relation type inference
    # ---------------------------
    def _infer_relation_types_from_doc(self, doc: Dict[str, Any]) -> List[str]:
        rels = doc.get("relations")
        if isinstance(rels, dict):
            keys = list(rels.keys())
            if keys:
                return keys
        return [DEFAULT_FALLBACK_RELATION_TYPE]

    # ---------------------------
    # Utilities: doc blocks
    # ---------------------------
    def _doc_text_block(self, doc: Dict[str, Any]) -> str:
        sents = doc.get("sentences")
        if isinstance(sents, list) and all(isinstance(x, str) for x in sents):
            use = sents
            if self.params.max_sentences_in_prompt is not None:
                use = use[: self.params.max_sentences_in_prompt]
            return "\n".join(f"- {s}" for s in use)
        return str(doc.get("text", ""))

    def _entities_block(self, doc: Dict[str, Any]) -> str:
        ents = doc.get("entities") or {}
        if not isinstance(ents, dict) or not ents:
            return "(no entities found)"

        lines: List[str] = []
        for ent_id, ent in ents.items():
            if not isinstance(ent, dict):
                continue
            ent_type = ent.get("type", "")
            mentions = ent.get("mentions") or []
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
        return "\n".join(lines) if lines else "(no entities found)"

    def _relation_schema_block(self, relation_types: Sequence[str]) -> str:
        return "\n".join(f"- {r}" for r in relation_types)

    def _few_shot_block(self, few_shots: List[Dict[str, Any]]) -> str:
        """
        Keep it compact: include doc text + entities + gold relations.
        """
        chunks: List[str] = ["## Few-shot examples"]
        for i, ex in enumerate(few_shots, 1):
            chunks.append(f"\n### Example {i}")
            chunks.append("Document:")
            chunks.append(self._doc_text_block(ex))
            chunks.append("\nEntities:")
            chunks.append(self._entities_block(ex))
            chunks.append("\nGold relations (for learning):")
            chunks.append(json.dumps(ex.get("relations", {}), ensure_ascii=False))
        return "\n".join(chunks)

    # ---------------------------
    # Utilities: strict JSON parse
    # ---------------------------
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

    # ---------------------------
    # Endpoint normalization (IDs only)
    # ---------------------------
    def _build_literal_to_entity_index(self, doc: Dict[str, Any]) -> Dict[str, str]:
        entities = doc.get("entities") or {}
        if not isinstance(entities, dict):
            return {}

        hits: Dict[str, set] = {}

        def norm(x: str) -> str:
            return re.sub(r"\s+", " ", x.strip().lower())

        for ent_id, ent in entities.items():
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

    def _normalize_pred_endpoints_to_entity_ids(
        self,
        doc: Dict[str, Any],
        pred: Dict[str, Any],
        allowed_relation_types: Sequence[str],
    ) -> Dict[str, List[List[str]]]:
        entities = doc.get("entities") or {}
        entity_ids = set(entities.keys()) if isinstance(entities, dict) else set()
        idx = self._build_literal_to_entity_index(doc)

        def norm(x: str) -> str:
            return re.sub(r"\s+", " ", x.strip().lower())

        out: Dict[str, List[List[str]]] = {r: [] for r in allowed_relation_types}

        debug_dropped: Dict[str, List[List[Any]]] = {r: [] for r in allowed_relation_types}
        debug_mapped: Dict[str, List[List[Any]]] = {r: [] for r in allowed_relation_types}

        for r in allowed_relation_types:
            pairs = pred.get(r, [])
            if not isinstance(pairs, list):
                continue
            for pair in pairs:
                if not isinstance(pair, list) or len(pair) != 2:
                    continue
                h, t = pair[0], pair[1]

                if not isinstance(h, str) or h not in entity_ids:
                    if self.params.keep_debug:
                        debug_dropped[r].append([h, t])
                    continue

                if isinstance(t, str) and t in entity_ids:
                    out[r].append([h, t])
                    continue

                # Try map literal -> entity
                mapped = None
                if isinstance(t, str):
                    mapped = idx.get(norm(t))
                else:
                    mapped = idx.get(norm(str(t)))

                if mapped and mapped in entity_ids:
                    out[r].append([h, mapped])
                    if self.params.keep_debug:
                        debug_mapped[r].append([h, t, mapped])
                else:
                    if self.params.keep_debug:
                        debug_dropped[r].append([h, t])

        if self.params.keep_debug:
            doc.setdefault("_debug", {})
            doc["_debug"]["marag_normalization"] = {
                "mapped_pairs": debug_mapped,
                "dropped_pairs": debug_dropped,
            }

        return out

    # ---------------------------
    # Tool contexts: ontology / kg
    # ---------------------------
    def _resolve_doc_ontology_links(self, doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Prefer in-doc artifact
        if isinstance(doc.get("ontology_links"), dict):
            return doc["ontology_links"]
        doc_id = doc.get("document_id") or doc.get("id")
        if doc_id and doc_id in self.ontology_links_by_docid:
            return self.ontology_links_by_docid[doc_id]
        return None

    def _resolve_doc_kg_triples(self, doc: Dict[str, Any]) -> List[List[str]]:
        # Prefer already attached
        kgc = doc.get("_kg_context", {})
        if isinstance(kgc, dict) and isinstance(kgc.get("triples"), list):
            return kgc.get("triples") or []

        doc_id = doc.get("document_id") or doc.get("id")
        if doc_id and doc_id in self.kg_triples_by_docid:
            triples = self.kg_triples_by_docid[doc_id]
            if isinstance(triples, list):
                return triples
        return []

    def _ontology_context(self, doc: Dict[str, Any]) -> str:
        if not self.params.enable_ontology:
            return "(ontology disabled)"
        if self.subontology_retriever is None:
            return "(no ontology retriever configured)"

        links = self._resolve_doc_ontology_links(doc)
        if links is None:
            return "(no ontology_links found)"

        fragment: SubOntologyFragment = self.subontology_retriever.retrieve(
            ontology_links=links,
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

        return "\n".join(parts) if parts else "(empty ontology fragment)"

    def _kg_context(self, doc: Dict[str, Any]) -> str:
        if not self.params.enable_kg:
            return "(kg disabled)"

        triples = self._resolve_doc_kg_triples(doc)
        if not triples:
            return "(no KG triples found for this doc)"

        use = triples[: self.params.kg_max_triples]
        # Expect triples as [head, rel, tail] OR dict-ish; normalize to text lines.
        lines: List[str] = ["### KG triples (top-k)"]
        for t in use:
            if isinstance(t, list) and len(t) == 3:
                lines.append(f"- {t[0]}  {t[1]}  {t[2]}")
            else:
                lines.append(f"- {t}")
        return "\n".join(lines)

    # ---------------------------
    # Optional tools: Wikidata / Web (lightweight, safe defaults)
    # ---------------------------
    def _wikidata_context(self, doc: Dict[str, Any]) -> str:
        if not self.params.enable_wikidata:
            return "(wikidata disabled)"

        # Lightweight label-only context without heavy calls by default.
        # If you want real calls, you can extend this with requests to wbsearchentities.
        ents = doc.get("entities") or {}
        if not isinstance(ents, dict) or not ents:
            return "(no entities for wikidata)"

        # Just list candidate surface forms (so planner/LLM can decide).
        picks: List[str] = []
        for ent_id, ent in ents.items():
            if not isinstance(ent, dict):
                continue
            mentions = ent.get("mentions") or []
            if not isinstance(mentions, list):
                continue
            if not mentions:
                continue
            m0 = mentions[0]
            if isinstance(m0, dict):
                trig = m0.get("trigger_word") or m0.get("text")
                if trig:
                    picks.append(f"{ent_id}: {trig}")
            if len(picks) >= self.params.wikidata_max_entities:
                break

        if not picks:
            return "(no wikidata candidates)"

        return "### Wikidata candidates (surface forms)\n" + "\n".join(f"- {x}" for x in picks)

    def _web_context(self, doc: Dict[str, Any]) -> str:
        if not self.params.enable_web:
            return "(web disabled)"

        # Stub: keep deterministic and cheap (no HTTP here by default).
        # You can later plug your own search provider and return snippets.
        title = str(doc.get("title", "")).strip()
        if not title:
            return "(no title for web query)"
        return f"### Web query suggestion\n- {title}\n\n(web retrieval not configured; provide snippets provider to enable)"

    # ---------------------------
    # Planner + Extraction prompts
    # ---------------------------
    def _build_planner_messages(
        self,
        doc: Dict[str, Any],
        relation_types: Sequence[str],
    ) -> List[Dict[str, str]]:
        system_msg = {"role": "system", "content": self.llm_config.system_prompt}
        user_parts = [
            "You are a planning agent for document-level relation extraction.",
            "Decide which tools to use among: ontology, kg, web, wikidata.",
            "Return ONLY valid JSON.",
            "",
            "## Document",
            self._doc_text_block(doc),
            "",
            "## Entities",
            self._entities_block(doc),
            "",
            "## Allowed relation types",
            self._relation_schema_block(relation_types),
            "",
            "## Output JSON schema",
            '{"use_ontology": true, "use_kg": true, "use_web": false, "use_wikidata": false, "notes": "short"}',
        ]
        return [system_msg, {"role": "user", "content": "\n".join(user_parts)}]

    def _build_extraction_messages(
        self,
        doc: Dict[str, Any],
        relation_types: Sequence[str],
        *,
        ontology_ctx: str,
        kg_ctx: str,
        web_ctx: str,
        wikidata_ctx: str,
        few_shots: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, str]]:
        system_msg = {"role": "system", "content": self.llm_config.system_prompt}

        user_parts: List[str] = [
            "You will extract document-level relations between the PROVIDED entity IDs only.",
            "You MUST output ONLY valid JSON (no markdown, no explanations).",
            "You MUST use ONLY the PROVIDED entity IDs in output pairs (no literals).",
            "Output keys MUST match the allowed relation types exactly.",
            "Values MUST be lists of [HEAD_ID, TAIL_ID] pairs.",
            "",
        ]

        if few_shots:
            user_parts.append(self._few_shot_block(few_shots))
            user_parts.append("")  # spacer

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
            "## Tool: Ontology context",
            ontology_ctx,
            "",
            "## Tool: KG context",
            kg_ctx,
            "",
            "## Tool: Web context (optional)",
            web_ctx,
            "",
            "## Tool: Wikidata context (optional)",
            wikidata_ctx,
            "",
            "## Output format (JSON only, no extra text)",
            'Return a JSON object whose keys are the allowed relation types, and values are lists of [HEAD_ID, TAIL_ID] pairs.',
            'Example: {"REL_TYPE":[["E1","E2"]],"OTHER":[]}',
        ]
        return [system_msg, {"role": "user", "content": "\n".join(user_parts)}]

    # ---------------------------
    # LangGraph build
    # ---------------------------
    def _build_graph(self):
        g = StateGraph(MARAGState)

        def node_plan(state: MARAGState) -> MARAGState:
            used = int(state.get("llm_calls_used", 0))
            if not self.params.enable_planner:
                state["plan"] = {
                    "use_ontology": self.params.enable_ontology,
                    "use_kg": self.params.enable_kg,
                    "use_web": self.params.enable_web,
                    "use_wikidata": self.params.enable_wikidata,
                    "notes": "planner disabled",
                }
                return state

            if used >= self.params.max_llm_calls:
                # no budget, fallback deterministic
                state["plan"] = {
                    "use_ontology": self.params.enable_ontology,
                    "use_kg": self.params.enable_kg,
                    "use_web": False,
                    "use_wikidata": False,
                    "notes": "no llm budget for planner",
                }
                return state

            msgs = self._build_planner_messages(state["doc"], state["relation_types"])
            raw = self._call_llm(msgs)
            state["llm_calls_used"] = used + 1

            obj = self._extract_json_object(raw) or {}
            # sanitize
            plan = {
                "use_ontology": bool(obj.get("use_ontology", self.params.enable_ontology)),
                "use_kg": bool(obj.get("use_kg", self.params.enable_kg)),
                "use_web": bool(obj.get("use_web", False)),
                "use_wikidata": bool(obj.get("use_wikidata", False)),
                "notes": str(obj.get("notes", ""))[:200],
            }
            state["plan"] = plan
            return state

        def node_ontology(state: MARAGState) -> MARAGState:
            plan = state.get("plan") or {}
            if plan.get("use_ontology", False):
                state["ontology_ctx"] = self._ontology_context(state["doc"])
            else:
                state["ontology_ctx"] = "(ontology not used)"
            return state

        def node_kg(state: MARAGState) -> MARAGState:
            plan = state.get("plan") or {}
            if plan.get("use_kg", False):
                state["kg_ctx"] = self._kg_context(state["doc"])
            else:
                state["kg_ctx"] = "(kg not used)"
            return state

        def node_web(state: MARAGState) -> MARAGState:
            plan = state.get("plan") or {}
            if plan.get("use_web", False):
                state["web_ctx"] = self._web_context(state["doc"])
            else:
                state["web_ctx"] = "(web not used)"
            return state

        def node_wikidata(state: MARAGState) -> MARAGState:
            plan = state.get("plan") or {}
            if plan.get("use_wikidata", False):
                state["wikidata_ctx"] = self._wikidata_context(state["doc"])
            else:
                state["wikidata_ctx"] = "(wikidata not used)"
            return state

        def node_extract(state: MARAGState) -> MARAGState:
            used = int(state.get("llm_calls_used", 0))
            if used >= self.params.max_llm_calls:
                # No budget left: empty output
                rel_types = state["relation_types"]
                state["pred_raw"] = {r: [] for r in rel_types}
                state["pred_norm"] = {r: [] for r in rel_types}
                return state

            doc = state["doc"]
            rel_types = state["relation_types"]
            few_shots = state.get("few_shots") or []

            msgs = self._build_extraction_messages(
                doc,
                rel_types,
                ontology_ctx=state.get("ontology_ctx", ""),
                kg_ctx=state.get("kg_ctx", ""),
                web_ctx=state.get("web_ctx", ""),
                wikidata_ctx=state.get("wikidata_ctx", ""),
                few_shots=few_shots,
            )
            raw = self._call_llm(msgs)
            state["llm_calls_used"] = used + 1

            parsed = self._extract_json_object(raw)
            if not isinstance(parsed, dict):
                parsed = {r: [] for r in rel_types}

            state["pred_raw"] = parsed

            # Ensure evaluator compatibility
            pred_ids_only = self._normalize_pred_endpoints_to_entity_ids(doc, parsed, rel_types)
            normalized = self._normalize_relation_dict(pred_ids_only, rel_types)

            state["pred_norm"] = normalized

            if self.params.keep_debug:
                doc.setdefault("_debug", {})
                doc["_debug"]["marag"] = {
                    "plan": state.get("plan"),
                    "llm_calls_used": state.get("llm_calls_used"),
                    "ontology_used": state.get("plan", {}).get("use_ontology"),
                    "kg_used": state.get("plan", {}).get("use_kg"),
                    "web_used": state.get("plan", {}).get("use_web"),
                    "wikidata_used": state.get("plan", {}).get("use_wikidata"),
                }

            return state

        g.add_node("plan", node_plan)
        g.add_node("onto", node_ontology)
        g.add_node("kg", node_kg)
        g.add_node("web", node_web)
        g.add_node("wikidata", node_wikidata)
        g.add_node("extract", node_extract)

        # Linear pipeline (cheap + deterministic). Planner decides whether contexts are used.
        g.set_entry_point("plan")
        g.add_edge("plan", "onto")
        g.add_edge("onto", "kg")
        g.add_edge("kg", "web")
        g.add_edge("web", "wikidata")
        g.add_edge("wikidata", "extract")
        g.add_edge("extract", END)

        return g.compile()

    # ---------------------------
    # Main API
    # ---------------------------
    def predict_relations(
        self,
        doc: Dict[str, Any],
        relation_types: Optional[Sequence[str]] = None,
        *,
        few_shots: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, List[List[str]]]:
        rel_types = list(relation_types) if relation_types else self._infer_relation_types_from_doc(doc)

        if self._graph is None:
            self._graph = self._build_graph()

        state: MARAGState = {
            "doc": doc,
            "relation_types": rel_types,
            "few_shots": few_shots or [],
            "llm_calls_used": 0,
        }

        out: MARAGState = self._graph.invoke(state)
        pred = out.get("pred_norm")
        if isinstance(pred, dict):
            return pred
        return {r: [] for r in rel_types}
