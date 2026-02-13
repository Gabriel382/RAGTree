# ragtree/processing/rag/strategies/langgraph_agentic_hybrid_relations.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ragtree.processing.rag.base_strategy import BaseRelationStrategy
from ragtree.ontologies.retrieval.subontology import SubOntologyRetriever, SubOntologyFragment

DEFAULT_FALLBACK_RELATION_TYPE = "causal_relation"


@dataclass
class LangGraphAgenticHybridParams:
    # Context toggles
    include_ontology_structured: bool = True
    include_ontology_ttl: bool = False
    kg_max_triples: int = 40
    max_sentences_in_prompt: Optional[int] = None

    # Planner knobs (fast by default)
    planner_mode: str = "rule"  # "rule" (default) or "llm"
    planner_depth: int = 1      # how many plan iterations (1 minimal)
    planner_verbosity: str = "minimal"  # "minimal"|"normal"|"verbose"

    # Online tools (OFF by default)
    enable_web: bool = False
    enable_wikidata: bool = False

    # LLM budget
    max_llm_calls: int = 1  # keep minimal; if planner_mode="llm" you likely want >=2

    # Safety / constraints
    web_timeout_sec: float = 3.0
    web_max_chars: int = 1500  # hard cap on internet text injected into prompt


class LangGraphAgenticHybridRelationStrategy(BaseRelationStrategy):
    """
    LangGraph-based single-agent hybrid RAG for DocRE:

      Tools:
        - Ontology tool (GrOWL-style) via SubOntologyRetriever + doc["ontology_links"]
        - KG tool (BYOKG-style) via doc["_kg_context"]["triples"]
        - Optional Web tool (Wikipedia REST summary)  [disabled by default]
        - Optional Wikidata tool (wbsearchentities + small summary) [disabled by default]

      Planner:
        - Always pulls ontology + KG.
        - Decides whether to run Web/Wikidata based on heuristics (rule planner),
          or optional LLM planner (costs calls).

      Output:
        dict { relation_type: [[head_id, tail_id], ...], ... }

    Runner responsibilities:
      - inject doc["ontology_links"] (from your ontology-link artifact)
      - inject doc["_kg_context"]["triples"] (from your KG artifact per doc)
    """

    def __init__(
        self,
        llm_config,
        *,
        retriever: SubOntologyRetriever,
        ontology_key: str,
        linking_method: str,
        params: Optional[LangGraphAgenticHybridParams] = None,
    ) -> None:
        super().__init__(llm_config)
        self.retriever = retriever
        self.ontology_key = ontology_key
        self.linking_method = linking_method
        self.params = params or LangGraphAgenticHybridParams()

        # Lazy-built graph (built on first call)
        self._graph = None

        # Simple in-memory caches (per strategy instance)
        self._cache_wikipedia: Dict[str, str] = {}
        self._cache_wikidata: Dict[str, str] = {}

    # ----------------------------
    # Relation type inference
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
    # Text blocks
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

    def _few_shot_block(self, few_shots: List[Dict[str, Any]]) -> str:
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

    # ----------------------------
    # Tool: Ontology
    # ----------------------------
    def _tool_ontology(self, doc: Dict[str, Any]) -> str:
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

    # ----------------------------
    # Tool: KG triples
    # ----------------------------
    def _tool_kg(self, doc: Dict[str, Any], relation_types: Sequence[str]) -> str:
        kg_ctx = doc.get("_kg_context", {})
        triples = []
        if isinstance(kg_ctx, dict):
            triples = kg_ctx.get("triples", []) or []

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

    # ----------------------------
    # Tool: Internet (Wikipedia summary)
    # ----------------------------
    def _extract_entity_queries(self, doc: Dict[str, Any], max_q: int = 5) -> List[str]:
        """
        Build small search queries from entity mention triggers.
        """
        entities = doc.get("entities") or {}
        qs: List[str] = []
        if not isinstance(entities, dict):
            return qs

        for _, ent in entities.items():
            mentions = ent.get("mentions") or []
            if not isinstance(mentions, list):
                continue
            for m in mentions[:1]:
                if not isinstance(m, dict):
                    continue
                trig = m.get("trigger_word") or m.get("text")
                if isinstance(trig, str):
                    s = trig.strip()
                    if s and s.lower() not in {q.lower() for q in qs}:
                        qs.append(s)
                        if len(qs) >= max_q:
                            return qs
        return qs

    def _tool_wikipedia(self, query: str) -> str:
        """
        Wikipedia REST summary. Optional + cached.
        """
        if query in self._cache_wikipedia:
            return self._cache_wikipedia[query]

        import requests  # local runtime dependency

        # Use the REST summary endpoint; it expects a title-ish string.
        # We keep it best-effort (this is optional tool).
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(query)}"
        try:
            r = requests.get(url, timeout=self.params.web_timeout_sec, headers={"accept": "application/json"})
            if r.status_code != 200:
                self._cache_wikipedia[query] = ""
                return ""
            data = r.json()
            extract = data.get("extract")
            if not isinstance(extract, str):
                self._cache_wikipedia[query] = ""
                return ""
            out = extract.strip()
            if len(out) > self.params.web_max_chars:
                out = out[: self.params.web_max_chars] + "..."
            self._cache_wikipedia[query] = out
            return out
        except Exception:
            self._cache_wikipedia[query] = ""
            return ""

    def _tool_internet_block(self, doc: Dict[str, Any]) -> str:
        if not self.params.enable_web:
            return "(internet tool disabled)"

        qs = self._extract_entity_queries(doc, max_q=4)
        if not qs:
            return "(no internet queries inferred)"

        parts: List[str] = ["### Internet lookups (Wikipedia summaries)"]
        for q in qs:
            txt = self._tool_wikipedia(q)
            if txt:
                parts.append(f"- Query: {q}\n  Summary: {txt}")
        return "\n".join(parts) if len(parts) > 1 else "(no internet info retrieved)"

    # ----------------------------
    # Tool: Wikidata (search entity)
    # ----------------------------
    def _tool_wikidata_search(self, query: str) -> str:
        if query in self._cache_wikidata:
            return self._cache_wikidata[query]

        import requests  # local runtime dependency

        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "search": query,
            "language": "en",
            "format": "json",
            "limit": 1,
        }
        try:
            r = requests.get(url, params=params, timeout=self.params.web_timeout_sec, headers={"accept": "application/json"})
            if r.status_code != 200:
                self._cache_wikidata[query] = ""
                return ""
            data = r.json()
            hits = data.get("search")
            if not isinstance(hits, list) or not hits:
                self._cache_wikidata[query] = ""
                return ""
            top = hits[0]
            label = top.get("label")
            desc = top.get("description")
            qid = top.get("id")
            if not isinstance(label, str) or not isinstance(qid, str):
                self._cache_wikidata[query] = ""
                return ""
            out = f"{label} ({qid})" + (f": {desc}" if isinstance(desc, str) else "")
            if len(out) > self.params.web_max_chars:
                out = out[: self.params.web_max_chars] + "..."
            self._cache_wikidata[query] = out
            return out
        except Exception:
            self._cache_wikidata[query] = ""
            return ""

    def _tool_wikidata_block(self, doc: Dict[str, Any]) -> str:
        if not self.params.enable_wikidata:
            return "(wikidata tool disabled)"

        qs = self._extract_entity_queries(doc, max_q=4)
        if not qs:
            return "(no wikidata queries inferred)"

        parts: List[str] = ["### Wikidata lookups (top entity match)"]
        for q in qs:
            txt = self._tool_wikidata_search(q)
            if txt:
                parts.append(f"- Query: {q}\n  Hit: {txt}")
        return "\n".join(parts) if len(parts) > 1 else "(no wikidata info retrieved)"

    # ----------------------------
    # Normalization: enforce entity IDs
    # ----------------------------
    def _build_literal_to_entity_index(self, doc: Dict[str, Any]) -> Dict[str, str]:
        entities = doc.get("entities") or {}
        hits: Dict[str, Set[str]] = {}

        def norm(s: str) -> str:
            return re.sub(r"\s+", " ", s.strip().lower())

        if not isinstance(entities, dict):
            return {}

        for ent_id, ent in entities.items():
            mentions = ent.get("mentions") or []
            if not isinstance(mentions, list):
                continue
            for m in mentions:
                if not isinstance(m, dict):
                    continue
                texts = []
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
    ) -> Dict[str, Any]:
        entities = doc.get("entities") or {}
        entity_ids = set(entities.keys()) if isinstance(entities, dict) else set()
        idx = self._build_literal_to_entity_index(doc)

        def norm(s: str) -> str:
            return re.sub(r"\s+", " ", s.strip().lower())

        out: Dict[str, List[List[str]]] = {r: [] for r in allowed_relation_types}

        for r in allowed_relation_types:
            pairs = pred.get(r, [])
            if not isinstance(pairs, list):
                continue
            for pair in pairs:
                if not isinstance(pair, list) or len(pair) != 2:
                    continue
                h, t = pair[0], pair[1]
                if not isinstance(h, str) or h not in entity_ids:
                    continue
                # Tail is ID
                if isinstance(t, str) and t in entity_ids:
                    out[r].append([h, t])
                    continue
                # Tail is literal -> try map
                if isinstance(t, str):
                    mapped = idx.get(norm(t))
                    if mapped and mapped in entity_ids:
                        out[r].append([h, mapped])
        return out

    # ----------------------------
    # Prompt assembly for extractor
    # ----------------------------
    def _build_extraction_messages(
        self,
        doc: Dict[str, Any],
        relation_types: Sequence[str],
        *,
        ontology_ctx: str,
        kg_ctx: str,
        internet_ctx: str,
        wikidata_ctx: str,
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
            "## Tool: Ontology context",
            ontology_ctx,
            "",
            "## Tool: KG context",
            kg_ctx,
            "",
            "## Tool: Internet context (optional)",
            internet_ctx,
            "",
            "## Tool: Wikidata context (optional)",
            wikidata_ctx,
            "",
            "## Output format (JSON only, no extra text)",
            "Return a JSON object whose keys are the allowed relation types, and values are lists of [HEAD_ID, TAIL_ID] pairs.",
            "Example:",
            '{"REL_TYPE": [["E1","E2"]], "OTHER": []}',
        ]

        return [system_msg, {"role": "user", "content": "\n".join(user_parts)}]

    # ----------------------------
    # Planner (rule or LLM)
    # ----------------------------
    def _rule_plan(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Default fast planner: always run ontology+kg; run web/wikidata only if enabled AND
        doc looks sparse/ambiguous (heuristics).
        """
        entities = doc.get("entities") or {}
        n_ent = len(entities) if isinstance(entities, dict) else 0

        # Heuristic: if many entities and text is long, internet might help, but keep off by default anyway.
        # We only "activate" if user enabled enable_web/enable_wikidata AND planner thinks it's useful.
        do_web = False
        do_wikidata = False

        if self.params.enable_web or self.params.enable_wikidata:
            text = str(doc.get("text") or "")
            sentences = doc.get("sentences")
            n_sent = len(sentences) if isinstance(sentences, list) else 0
            long_doc = len(text) > 1200 or n_sent > 15

            # If doc is long and has many entities, extra grounding may help.
            if n_ent >= 8 and long_doc:
                do_web = bool(self.params.enable_web)
                do_wikidata = bool(self.params.enable_wikidata)

        return {"do_web": do_web, "do_wikidata": do_wikidata}

    def _llm_plan(self, doc: Dict[str, Any], relation_types: Sequence[str], llm_calls_used: int) -> Tuple[Dict[str, Any], int]:
        """
        Optional LLM-based planner. Costs 1 LLM call per planning step.
        If max_llm_calls budget doesn't allow, falls back to rule plan.

        Returns: (plan_dict, new_llm_calls_used)
        """
        if llm_calls_used >= self.params.max_llm_calls:
            return self._rule_plan(doc), llm_calls_used

        verbosity = self.params.planner_verbosity
        detail_line = {
            "minimal": "Keep it extremely short.",
            "normal": "Be concise.",
            "verbose": "Be explicit and detailed.",
        }.get(verbosity, "Keep it extremely short.")

        sys = {"role": "system", "content": "You are a planning assistant for tool usage."}
        user = {
            "role": "user",
            "content": "\n".join(
                [
                    "Decide whether running Internet and/or Wikidata lookup will help relation extraction.",
                    "Ontology and KG tools will ALWAYS run.",
                    f"Internet enabled: {self.params.enable_web}",
                    f"Wikidata enabled: {self.params.enable_wikidata}",
                    "",
                    "Return ONLY valid JSON with keys: do_web (bool), do_wikidata (bool).",
                    detail_line,
                    "",
                    "Document:",
                    self._doc_text_block(doc),
                    "",
                    "Entities:",
                    self._entities_block(doc),
                    "",
                    "Allowed relation types:",
                    self._relation_schema_block(relation_types),
                ]
            ),
        }

        raw = self._call_llm([sys, user])
        llm_calls_used += 1
        obj = self._extract_json_object(raw) or {}
        do_web = bool(obj.get("do_web")) if self.params.enable_web else False
        do_wikidata = bool(obj.get("do_wikidata")) if self.params.enable_wikidata else False
        return {"do_web": do_web, "do_wikidata": do_wikidata}, llm_calls_used

    # ----------------------------
    # LangGraph construction
    # ----------------------------
    def _build_graph(self):
        try:
            from langgraph.graph import StateGraph, END
        except Exception as e:
            raise ImportError(
                "LangGraph is not installed. Install with: pip install langgraph"
            ) from e

        # State is a plain dict for minimal friction.
        g = StateGraph(dict)

        def node_prepare(state: Dict[str, Any]) -> Dict[str, Any]:
            # Ensure keys exist
            state.setdefault("ontology_ctx", "")
            state.setdefault("kg_ctx", "")
            state.setdefault("internet_ctx", "")
            state.setdefault("wikidata_ctx", "")
            state.setdefault("plan", {})
            state.setdefault("llm_calls_used", 0)
            state.setdefault("pred_raw", None)
            state.setdefault("pred_norm", None)
            return state

        def node_ontology(state: Dict[str, Any]) -> Dict[str, Any]:
            doc = state["doc"]
            state["ontology_ctx"] = self._tool_ontology(doc)
            return state

        def node_kg(state: Dict[str, Any]) -> Dict[str, Any]:
            doc = state["doc"]
            rel_types = state["relation_types"]
            state["kg_ctx"] = self._tool_kg(doc, rel_types)
            return state

        def node_plan(state: Dict[str, Any]) -> Dict[str, Any]:
            doc = state["doc"]
            rel_types = state["relation_types"]
            used = int(state.get("llm_calls_used", 0))

            if self.params.planner_mode == "llm":
                plan, used2 = self._llm_plan(doc, rel_types, used)
                state["plan"] = plan
                state["llm_calls_used"] = used2
            else:
                state["plan"] = self._rule_plan(doc)
            return state

        def node_internet(state: Dict[str, Any]) -> Dict[str, Any]:
            doc = state["doc"]
            plan = state.get("plan") or {}
            if plan.get("do_web"):
                state["internet_ctx"] = self._tool_internet_block(doc)
            else:
                state["internet_ctx"] = "(internet not used)"
            return state

        def node_wikidata(state: Dict[str, Any]) -> Dict[str, Any]:
            doc = state["doc"]
            plan = state.get("plan") or {}
            if plan.get("do_wikidata"):
                state["wikidata_ctx"] = self._tool_wikidata_block(doc)
            else:
                state["wikidata_ctx"] = "(wikidata not used)"
            return state

        def node_extract(state: Dict[str, Any]) -> Dict[str, Any]:
            doc = state["doc"]
            rel_types = state["relation_types"]
            few_shots = state.get("few_shots")

            used = int(state.get("llm_calls_used", 0))
            if used >= self.params.max_llm_calls:
                # No budget left: output empty under schema
                state["pred_raw"] = {r: [] for r in rel_types}
                return state

            msgs = self._build_extraction_messages(
                doc,
                rel_types,
                ontology_ctx=state["ontology_ctx"],
                kg_ctx=state["kg_ctx"],
                internet_ctx=state["internet_ctx"],
                wikidata_ctx=state["wikidata_ctx"],
                few_shots=few_shots,
            )
            raw = self._call_llm(msgs)
            used += 1
            state["llm_calls_used"] = used

            parsed = self._extract_json_object(raw)
            state["pred_raw"] = parsed if isinstance(parsed, dict) else {r: [] for r in rel_types}
            return state

        def node_normalize(state: Dict[str, Any]) -> Dict[str, Any]:
            doc = state["doc"]
            rel_types = state["relation_types"]
            pred_raw = state.get("pred_raw") or {}

            # 1) enforce entity IDs
            pred_entity_only = self._normalize_pred_endpoints_to_entity_ids(doc, pred_raw, rel_types)
            # 2) enforce schema keys + list-of-pairs
            pred_norm = self._normalize_relation_dict(pred_entity_only, rel_types)

            state["pred_norm"] = pred_norm
            return state

        # Wiring
        g.add_node("prepare", node_prepare)
        g.add_node("ontology", node_ontology)
        g.add_node("kg", node_kg)
        g.add_node("plan", node_plan)
        g.add_node("internet", node_internet)
        g.add_node("wikidata", node_wikidata)
        g.add_node("extract", node_extract)
        g.add_node("normalize", node_normalize)

        g.set_entry_point("prepare")
        g.add_edge("prepare", "ontology")
        g.add_edge("ontology", "kg")
        g.add_edge("kg", "plan")

        # always pass through these nodes (they are cheap; they no-op if not used)
        g.add_edge("plan", "internet")
        g.add_edge("internet", "wikidata")
        g.add_edge("wikidata", "extract")
        g.add_edge("extract", "normalize")
        g.add_edge("normalize", END)

        return g.compile()

    # ----------------------------
    # Public API
    # ----------------------------
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

        state = {
            "doc": doc,
            "relation_types": rel_types,
            "few_shots": few_shots or [],
            "llm_calls_used": 0,
        }

        out = self._graph.invoke(state)
        pred = out.get("pred_norm")
        if isinstance(pred, dict):
            return pred
        return {r: [] for r in rel_types}
