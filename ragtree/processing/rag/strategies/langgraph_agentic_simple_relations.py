# ragtree/processing/rag/strategies/langgraph_agentic_simple_relations.py
from __future__ import annotations

import json
import re
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import requests

from ragtree.processing.rag.base_strategy import BaseRelationStrategy

DEFAULT_FALLBACK_RELATION_TYPE = "causal_relation"


@dataclass
class LangGraphAgenticSimpleParams:
    # Prompt sizing
    max_sentences_in_prompt: Optional[int] = None

    # Planner knobs (fast by default)
    planner_mode: str = "rule"          # "rule" (default) or "llm"
    planner_depth: int = 1              # iterations (1 minimal)
    planner_verbosity: str = "minimal"  # "minimal"|"normal"|"verbose"

    # Online tools (ON by default for this method)
    enable_web: bool = True
    enable_wikidata: bool = True

    # Tool constraints
    web_timeout_sec: float = 3.0
    web_max_chars: int = 1500
    wikidata_timeout_sec: float = 3.0
    wikidata_max_chars: int = 1500

    # Retrieval sizes
    web_top_k: int = 2
    wikidata_top_k: int = 2

    # LLM budget
    max_llm_calls: int = 1  # keep minimal; if planner_mode="llm" you likely want >=2


class LangGraphAgenticSimpleRelationStrategy(BaseRelationStrategy):
    """
    LangGraph-based single-agent "simple agentic RAG" for DocRE:

      Tools:
        - Web tool (Wikipedia REST summary)           [ON by default]
        - Wikidata tool (wbsearchentities + summary) [ON by default]

      No ontology / no KG.
      Planner (rule by default) decides whether to run Web/Wikidata.
      Extraction is one-shot by default (1 LLM call per doc).

    Output:
      dict { relation_type: [[head_id, tail_id], ...], ... }
    """

    def __init__(self, llm_config, *, params: Optional[LangGraphAgenticSimpleParams] = None) -> None:
        super().__init__(llm_config)
        self.params = params or LangGraphAgenticSimpleParams()
        self._graph = None

        # Small per-instance caches
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
    # Text blocks (prompt)
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
                    mentions = []
                m0 = mentions[0] if mentions else {}
                trig = m0.get("trigger_word") or m0.get("text") or ""
                lines.append(f"- {ent_id} | {ent_type} | {trig}")
        return "\n".join(lines) if lines else "- (none)"

    def _relation_schema_block(self, relation_types: Sequence[str]) -> str:
        return "\n".join(f"- {r}" for r in relation_types)

    def _few_shot_block(self, few_shots: List[Dict[str, Any]]) -> str:
        chunks: List[str] = []
        for i, shot in enumerate(few_shots, start=1):
            chunks.append(f"### Example {i}")
            chunks.append("Document:")
            chunks.append(self._doc_text_block(shot))
            chunks.append("Entities:")
            chunks.append(self._entities_block(shot))
            chunks.append("Gold relations (JSON):")
            rels = shot.get("relations") or {}
            chunks.append(json.dumps(rels, ensure_ascii=False))
            chunks.append("")
        return "\n".join(chunks).strip()

    def _build_extraction_messages(
        self,
        doc: Dict[str, Any],
        relation_types: Sequence[str],
        *,
        web_ctx: str,
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
            "## Tool: Web context",
            web_ctx or "(empty)",
            "",
            "## Tool: Wikidata context",
            wikidata_ctx or "(empty)",
            "",
            "## Output format (JSON only, no extra text)",
            "Return a JSON object whose keys are the allowed relation types, and values are lists of [HEAD_ID, TAIL_ID] pairs.",
            "Example:",
            '{"REL_TYPE": [["E1","E2"]], "OTHER": []}',
        ]

        return [system_msg, {"role": "user", "content": "\n".join(user_parts)}]

    # ----------------------------
    # Utilities
    # ----------------------------
    def _all_entity_ids(self, doc: Dict[str, Any]) -> Set[str]:
        ents = doc.get("entities", {})
        return set(ents.keys()) if isinstance(ents, dict) else set()

    def _pick_query_terms(self, doc: Dict[str, Any], max_terms: int = 3) -> List[str]:
        """
        Build short query terms from entity trigger words + title.
        """
        terms: List[str] = []
        title = str(doc.get("title", "")).strip()
        if title:
            terms.append(title)

        ents = doc.get("entities", {})
        if isinstance(ents, dict):
            for _, ent in ents.items():
                mentions = ent.get("mentions", [])
                if not isinstance(mentions, list) or not mentions:
                    continue
                m0 = mentions[0] if isinstance(mentions[0], dict) else {}
                trig = (m0.get("trigger_word") or m0.get("text") or "").strip()
                if trig and trig not in terms:
                    terms.append(trig)
                if len(terms) >= max_terms:
                    break

        # clean trivial terms
        out = []
        for t in terms:
            t2 = re.sub(r"\s+", " ", t).strip()
            if len(t2) >= 2:
                out.append(t2)
        return out[:max_terms]

    def _truncate(self, s: str, max_chars: int) -> str:
        if not s:
            return ""
        s = s.strip()
        return s if len(s) <= max_chars else (s[: max_chars].rstrip() + "")

    # ----------------------------
    # Tool: Web (Wikipedia summaries)
    # ----------------------------
    def _tool_web_context(self, doc: Dict[str, Any]) -> str:
        if not self.params.enable_web:
            return ""

        terms = self._pick_query_terms(doc, max_terms=3)
        if not terms:
            return ""

        key = "||".join(terms)
        if key in self._cache_wikipedia:
            return self._cache_wikipedia[key]

        # Use Wikipedia search API, then REST summary for top pages
        sess = requests.Session()
        snippets: List[str] = []
        for q in terms[: self.params.web_top_k]:
            try:
                # Search
                params = {
                    "action": "query",
                    "list": "search",
                    "srsearch": q,
                    "format": "json",
                }
                r = sess.get("https://en.wikipedia.org/w/api.php", params=params, timeout=self.params.web_timeout_sec)
                data = r.json()
                hits = (data.get("query", {}).get("search", []) or [])[:1]
                if not hits:
                    continue
                title = hits[0].get("title")
                if not title:
                    continue

                # Summary
                t_enc = urllib.parse.quote(str(title).replace(" ", "_"))
                rs = sess.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{t_enc}",
                    timeout=self.params.web_timeout_sec,
                    headers={"accept": "application/json"},
                )
                js = rs.json()
                extract = js.get("extract") or ""
                if extract:
                    snippets.append(f"[Wikipedia] {title}: {extract}")
            except Exception:
                continue

        out = self._truncate("\n".join(snippets).strip(), self.params.web_max_chars)
        self._cache_wikipedia[key] = out
        return out

    # ----------------------------
    # Tool: Wikidata (search + short entity descriptions)
    # ----------------------------
    def _tool_wikidata_context(self, doc: Dict[str, Any]) -> str:
        if not self.params.enable_wikidata:
            return ""

        terms = self._pick_query_terms(doc, max_terms=3)
        if not terms:
            return ""

        key = "||".join(terms)
        if key in self._cache_wikidata:
            return self._cache_wikidata[key]

        sess = requests.Session()
        chunks: List[str] = []

        for q in terms[: self.params.wikidata_top_k]:
            try:
                params = {
                    "action": "wbsearchentities",
                    "search": q,
                    "language": "en",
                    "format": "json",
                    "limit": 2,
                }
                r = sess.get("https://www.wikidata.org/w/api.php", params=params, timeout=self.params.wikidata_timeout_sec)
                data = r.json()
                results = data.get("search", []) or []
                for item in results[:2]:
                    qid = item.get("id")
                    label = item.get("label") or ""
                    desc = item.get("description") or ""
                    if qid:
                        chunks.append(f"[Wikidata] {qid} | {label}  {desc}")
            except Exception:
                continue

        out = self._truncate("\n".join(chunks).strip(), self.params.wikidata_max_chars)
        self._cache_wikidata[key] = out
        return out

    # ----------------------------
    # Post-filter / normalize output to valid entity ids
    # ----------------------------
    def _normalize_pred_endpoints_to_entity_ids(
        self,
        pred: Dict[str, Any],
        *,
        entity_ids: Set[str],
        relation_types: Sequence[str],
    ) -> Dict[str, List[List[str]]]:
        out: Dict[str, List[List[str]]] = {r: [] for r in relation_types}

        if not isinstance(pred, dict):
            return out

        for r in relation_types:
            pairs = pred.get(r, [])
            if not isinstance(pairs, list):
                continue
            kept: List[List[str]] = []
            for p in pairs:
                if not (isinstance(p, list) or isinstance(p, tuple)) or len(p) != 2:
                    continue
                h, t = p[0], p[1]
                if isinstance(h, str) and isinstance(t, str) and h in entity_ids and t in entity_ids:
                    kept.append([h, t])
            out[r] = kept

        return out

    # ----------------------------
    # LangGraph
    # ----------------------------
    def _build_graph(self):
        from langgraph.graph import StateGraph, END

        def node_prepare(state: Dict[str, Any]) -> Dict[str, Any]:
            state["web_ctx"] = ""
            state["wikidata_ctx"] = ""
            state["plan"] = {"do_web": False, "do_wikidata": False}
            return state

        def node_plan(state: Dict[str, Any]) -> Dict[str, Any]:
            # Fast rule-based planner by default (0 LLM cost)
            doc = state["doc"]

            ents = doc.get("entities", {})
            n_ents = len(ents) if isinstance(ents, dict) else 0

            do_web = bool(self.params.enable_web)
            do_wd = bool(self.params.enable_wikidata)

            # tiny heuristic for speed: if no entities, tools won't help much
            if n_ents == 0:
                do_web = False
                do_wd = False

            state["plan"] = {"do_web": do_web, "do_wikidata": do_wd}
            return state

        def node_web(state: Dict[str, Any]) -> Dict[str, Any]:
            if not state.get("plan", {}).get("do_web", False):
                return state
            try:
                state["web_ctx"] = self._tool_web_context(state["doc"])
            except Exception:
                state["web_ctx"] = ""
            return state

        def node_wikidata(state: Dict[str, Any]) -> Dict[str, Any]:
            if not state.get("plan", {}).get("do_wikidata", False):
                return state
            try:
                state["wikidata_ctx"] = self._tool_wikidata_context(state["doc"])
            except Exception:
                state["wikidata_ctx"] = ""
            return state

        def node_extract(state: Dict[str, Any]) -> Dict[str, Any]:
            rel_types = state["relation_types"]
            used = int(state.get("llm_calls_used", 0))
            if used >= int(self.params.max_llm_calls):
                state["pred_raw"] = {r: [] for r in rel_types}
                return state

            msgs = self._build_extraction_messages(
                state["doc"],
                rel_types,
                web_ctx=state.get("web_ctx", ""),
                wikidata_ctx=state.get("wikidata_ctx", ""),
                few_shots=state.get("few_shots") or [],
            )
            raw = self._call_llm(msgs)
            state["llm_calls_used"] = used + 1
            state["pred_text"] = raw

            obj = self._extract_json_object(raw)
            state["pred_raw"] = obj if isinstance(obj, dict) else {r: [] for r in rel_types}
            return state

        def node_normalize(state: Dict[str, Any]) -> Dict[str, Any]:
            doc = state["doc"]
            rel_types = state["relation_types"]
            entity_ids = self._all_entity_ids(doc)

            pred_raw = state.get("pred_raw") or {}
            pred_norm = self._normalize_pred_endpoints_to_entity_ids(
                pred_raw,
                entity_ids=entity_ids,
                relation_types=rel_types,
            )
            state["pred_norm"] = pred_norm
            return state

        g = StateGraph(dict)
        g.add_node("prepare", node_prepare)
        g.add_node("plan", node_plan)
        g.add_node("web", node_web)
        g.add_node("wikidata", node_wikidata)
        g.add_node("extract", node_extract)
        g.add_node("normalize", node_normalize)

        g.set_entry_point("prepare")
        g.add_edge("prepare", "plan")

        # sequential tool calls (cheap); plan decides to skip
        g.add_edge("plan", "web")
        g.add_edge("web", "wikidata")
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
