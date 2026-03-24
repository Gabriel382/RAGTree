# ragtree/processing/rag/strategies/marag_relations.py
from __future__ import annotations

import json
import re
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, TypedDict, Tuple

import requests

from ragtree.processing.rag.base_strategy import BaseRelationStrategy
from ragtree.ontologies.retrieval.subontology import SubOntologyRetriever, SubOntologyFragment

try:
    from langgraph.graph import StateGraph, END
except Exception:
    StateGraph = None
    END = None

DEFAULT_FALLBACK_RELATION_TYPE = "causal_relation"


# ----------------------------
# Params
# ----------------------------
@dataclass
class MARAGParams:
    # LLM budget
    max_llm_calls: int = 3

    # High-level toggles
    enable_planner: bool = True
    enable_relation_type_selector: bool = True
    enable_multi_proposers: bool = True
    enable_verifier: bool = True

    # Tool toggles
    enable_ontology: bool = True
    enable_kg: bool = True
    enable_web: bool = True          # Wikipedia summary
    enable_wikidata: bool = True     # wbsearchentities + short desc

    # Prompt budget controls
    max_sentences_in_prompt: Optional[int] = None
    max_relation_types_in_prompt: int = 18   # IMPORTANT: keep small
    proposer_max_pairs_per_rel: int = 12     # cap verbosity per proposer
    kg_max_triples: int = 40
    web_max_chars: int = 1400
    wikidata_max_chars: int = 1400

    # Multi proposer
    num_proposers: int = 3           # parameter you wanted
    proposer_entity_overlap: int = 2 # slight overlap to reduce misses

    # HTTP safety
    http_timeout_sec: float = 4.0
    http_user_agent: str = "ragtree-marag/0.1"

    # Debug
    keep_debug: bool = True
    verbose: bool = False


class MARAGState(TypedDict, total=False):
    doc: Dict[str, Any]
    relation_types_all: List[str]        # full schema for output
    relation_types_focus: List[str]      # shortlisted for LLM prompts
    few_shots: List[Dict[str, Any]]

    # tool contexts
    ontology_ctx: str
    kg_ctx: str
    web_ctx: str
    wikidata_ctx: str

    # plan
    plan: Dict[str, Any]

    # llm usage
    llm_calls_used: int

    # proposer outputs
    proposer_preds: List[Dict[str, Any]]
    pred_merged_raw: Dict[str, Any]
    pred_verified_raw: Dict[str, Any]
    pred_norm: Dict[str, List[List[str]]]


class MARagRelationStrategy(BaseRelationStrategy):
    """
    Strong MA-RAG (LangGraph) for DocRE:

    Fan-out retrieval:
      - Ontology fragment from ontology_links (SubOntologyRetriever)
      - KG triples from doc["_kg_context"]["triples"] or external map
      - Wikipedia REST summary (title-based)
      - Wikidata entity search (surface forms from entity mentions)

    Multi-agent reasoning:
      - Optional planner (LLM) decides which tools to trust/use
      - Relation-type selector (LLM) reduces 90-rel schema to top-N
      - N proposer agents (LLM) each handles a slice of entity IDs
      - Verifier agent (LLM) prunes unsupported pairs using evidence
      - Final normalization enforces ID-only + full schema keys

    This is intentionally slower but should beat single-agent hybrid on quality.
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
            raise ImportError("LangGraph not available. Install with: pip install langgraph")

        self.params = params or MARAGParams()
        self.subontology_retriever = subontology_retriever
        self.linking_method = linking_method
        self.ontology_links_by_docid = ontology_links_by_docid or {}
        self.kg_triples_by_docid = kg_triples_by_docid or {}

        self._graph = None
        self._log = logging.getLogger("ragtree.marag")
        if self.params.verbose:
            self._log.setLevel(logging.INFO)

        # caches to avoid repeated HTTP calls
        self._cache_wikipedia: Dict[str, str] = {}
        self._cache_wikidata: Dict[str, str] = {}

    # ----------------------------
    # Basic blocks
    # ----------------------------
    def _infer_relation_types_from_doc(self, doc: Dict[str, Any]) -> List[str]:
        rels = doc.get("relations")
        if isinstance(rels, dict):
            keys = list(rels.keys())
            if keys:
                return keys
        return [DEFAULT_FALLBACK_RELATION_TYPE]

    def _doc_text_block(self, doc: Dict[str, Any]) -> str:
        title = str(doc.get("title", "") or "").strip()
        sents = doc.get("sentences")
        if isinstance(sents, list) and all(isinstance(x, str) for x in sents):
            use = sents
            if self.params.max_sentences_in_prompt is not None:
                use = use[: self.params.max_sentences_in_prompt]
            txt = "\n".join(f"- {s}" for s in use)
        else:
            txt = str(doc.get("text", "") or "")
        return f"Title: {title}\nText:\n{txt}"

    def _entities_block(self, doc: Dict[str, Any], *, limit_entities: Optional[Set[str]] = None) -> str:
        ents = doc.get("entities") or {}
        if not isinstance(ents, dict) or not ents:
            return "(no entities found)"
        lines: List[str] = []
        for ent_id, ent in ents.items():
            if limit_entities is not None and ent_id not in limit_entities:
                continue
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
                lines.append(f"{ent_id}\tTYPE={ent_type}\tTRIGGER={trig}\tSENT_ID={sent_id}\tOFFSET={offset}")
                shown += 1
                if shown >= 2:
                    break
        return "\n".join(lines) if lines else "(no entities found)"

    def _relation_schema_block(self, relation_types: Sequence[str]) -> str:
        return "\n".join(f"- {r}" for r in relation_types)

    # ----------------------------
    # Few-shot: compact + safe
    # ----------------------------
    def _few_shot_block(self, few_shots: List[Dict[str, Any]], allowed_relation_types: Sequence[str]) -> str:
        blocks: List[str] = []
        idx = 1
        for ex in few_shots:
            rels = ex.get("relations")
            if not isinstance(rels, dict) or not rels:
                continue
            sents = ex.get("sentences")
            if isinstance(sents, list) and all(isinstance(s, str) for s in sents):
                ex_text = "\n".join(f"- {s}" for s in sents[:3])
            else:
                ex_text = str(ex.get("text", ""))[:900]

            # entities (subset)
            ents = ex.get("entities") or {}
            ent_lines: List[str] = []
            if isinstance(ents, dict):
                for ent_id, ent in list(ents.items())[:12]:
                    if not isinstance(ent, dict):
                        continue
                    m0 = (ent.get("mentions") or [{}])[0]
                    if isinstance(m0, dict):
                        trig = m0.get("trigger_word") or m0.get("text") or ""
                        ent_lines.append(f"{ent_id}\tTYPE={ent.get('type','')}\tTRIGGER={trig}")
            if not ent_lines:
                ent_lines = ["(no entities)"]

            # gold json with allowed keys only
            out = {r: [] for r in allowed_relation_types}
            for r in allowed_relation_types:
                pairs = rels.get(r, [])
                if isinstance(pairs, list):
                    kept: List[List[str]] = []
                    for p in pairs:
                        if isinstance(p, list) and len(p) == 2 and all(isinstance(x, str) for x in p):
                            kept.append([p[0], p[1]])
                        if len(kept) >= 6:
                            break
                    out[r] = kept

            blocks.append(
                "\n".join(
                    [
                        f"### Example {idx}",
                        "Text:",
                        ex_text,
                        "",
                        "Entities:",
                        "\n".join(ent_lines),
                        "",
                        "Gold output JSON:",
                        json.dumps(out, ensure_ascii=False),
                    ]
                )
            )
            idx += 1

        return "\n\n".join(blocks).strip() if blocks else ""

    # ----------------------------
    # JSON extraction
    # ----------------------------
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

    # ----------------------------
    # Endpoint normalization (ID-only)
    # ----------------------------
    def _build_literal_to_entity_index(self, doc: Dict[str, Any]) -> Dict[str, str]:
        ents = doc.get("entities") or {}
        if not isinstance(ents, dict):
            return {}
        hits: Dict[str, Set[str]] = {}

        def norm(x: str) -> str:
            return re.sub(r"\s+", " ", x.strip().lower())

        for ent_id, ent in ents.items():
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
                    if k:
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
        ents = doc.get("entities") or {}
        entity_ids = set(ents.keys()) if isinstance(ents, dict) else set()
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

    # ----------------------------
    # Tool context resolvers
    # ----------------------------
    def _resolve_doc_ontology_links(self, doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if isinstance(doc.get("ontology_links"), dict):
            return doc["ontology_links"]
        doc_id = doc.get("document_id") or doc.get("id")
        if doc_id and doc_id in self.ontology_links_by_docid:
            return self.ontology_links_by_docid[doc_id]
        return None

    def _resolve_doc_kg_triples(self, doc: Dict[str, Any]) -> List[List[str]]:
        kgc = doc.get("_kg_context", {})
        if isinstance(kgc, dict) and isinstance(kgc.get("triples"), list):
            return kgc.get("triples") or []
        doc_id = doc.get("document_id") or doc.get("id")
        if doc_id and doc_id in self.kg_triples_by_docid:
            triples = self.kg_triples_by_docid[doc_id]
            return triples if isinstance(triples, list) else []
        return []

    def _ontology_context(self, doc: Dict[str, Any]) -> str:
        if not self.params.enable_ontology:
            return "(ontology disabled)"
        if self.subontology_retriever is None:
            return "(no ontology retriever configured)"
        links = self._resolve_doc_ontology_links(doc)
        if links is None:
            return "(no ontology_links found)"
        frag: SubOntologyFragment = self.subontology_retriever.retrieve(
            ontology_links=links,
            method=self.linking_method,
            params={},
        )
        parts: List[str] = []
        parts.append("### Sub-ontology fragment (structured JSON)")
        parts.append(json.dumps(frag.to_dict(), ensure_ascii=False, indent=2))
        return "\n".join(parts)

    def _kg_context(self, doc: Dict[str, Any]) -> str:
        if not self.params.enable_kg:
            return "(kg disabled)"
        triples = self._resolve_doc_kg_triples(doc)
        if not triples:
            return "(no kg triples found)"
        triples = [t for t in triples if isinstance(t, list) and len(t) == 3][: self.params.kg_max_triples]
        lines = [f"- {h} | {r} | {t}" for h, r, t in triples]
        return "### KG triples\n" + "\n".join(lines)

    # ----------------------------
    # Web (Wikipedia) + Wikidata
    # ----------------------------
    def _http_get_json(self, url: str, *, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        try:
            headers = {"User-Agent": self.params.http_user_agent}
            resp = requests.get(url, params=params, headers=headers, timeout=self.params.http_timeout_sec)
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception:
            return None

    def _wikipedia_summary(self, title: str) -> str:
        key = title.strip().lower()
        if not key:
            return "(no title)"
        if key in self._cache_wikipedia:
            return self._cache_wikipedia[key]
        # Wikipedia REST summary endpoint
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + requests.utils.quote(title)
        obj = self._http_get_json(url)
        if not isinstance(obj, dict):
            out = "(wikipedia: no summary)"
        else:
            extract = str(obj.get("extract") or "").strip()
            if not extract:
                out = "(wikipedia: empty summary)"
            else:
                out = "### Wikipedia summary\n" + extract[: self.params.web_max_chars]
        self._cache_wikipedia[key] = out
        return out

    def _wikidata_snippets_from_entities(self, doc: Dict[str, Any]) -> str:
        # Use a few surface forms from entity mentions
        ents = doc.get("entities") or {}
        if not isinstance(ents, dict) or not ents:
            return "(no entities)"
        picks: List[str] = []
        for ent_id, ent in ents.items():
            if not isinstance(ent, dict):
                continue
            m0 = (ent.get("mentions") or [{}])[0]
            if isinstance(m0, dict):
                trig = m0.get("trigger_word") or m0.get("text")
                if trig:
                    picks.append(str(trig))
            if len(picks) >= 4:
                break

        if not picks:
            return "(no wikidata candidates)"

        key = "||".join(picks).lower()
        if key in self._cache_wikidata:
            return self._cache_wikidata[key]

        snippets: List[str] = []
        for q in picks[:3]:
            obj = self._http_get_json(
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbsearchentities",
                    "search": q,
                    "language": "en",
                    "format": "json",
                    "limit": 3,
                },
            )
            if not isinstance(obj, dict) or not isinstance(obj.get("search"), list):
                continue
            for item in obj["search"][:2]:
                if not isinstance(item, dict):
                    continue
                label = item.get("label")
                desc = item.get("description")
                wid = item.get("id")
                if label and wid:
                    line = f"- {label} ({wid})" + (f": {desc}" if desc else "")
                    snippets.append(line)

        out = "### Wikidata candidates\n" + ("\n".join(snippets) if snippets else "(wikidata: no results)")
        out = out[: self.params.wikidata_max_chars]
        self._cache_wikidata[key] = out
        return out

    def _web_context(self, doc: Dict[str, Any]) -> str:
        if not self.params.enable_web:
            return "(web disabled)"
        title = str(doc.get("title", "") or "").strip()
        if not title:
            return "(no title for wikipedia)"
        return self._wikipedia_summary(title)

    def _wikidata_context(self, doc: Dict[str, Any]) -> str:
        if not self.params.enable_wikidata:
            return "(wikidata disabled)"
        return self._wikidata_snippets_from_entities(doc)

    # ----------------------------
    # LLM message builders
    # ----------------------------
    def _build_planner_messages(self, doc: Dict[str, Any]) -> List[Dict[str, str]]:
        system_msg = {"role": "system", "content": self.llm_config.system_prompt}
        user = "\n".join(
            [
                "You are a planning agent for DocRE.",
                "Decide which contexts to use: ontology, kg, web, wikidata.",
                "Return ONLY valid JSON.",
                "",
                "## Document",
                self._doc_text_block(doc),
                "",
                "## Output JSON schema",
                '{"use_ontology": true, "use_kg": true, "use_web": true, "use_wikidata": true, "notes": "short"}',
            ]
        )
        return [system_msg, {"role": "user", "content": user}]

    def _build_relation_type_selector_messages(
        self,
        doc: Dict[str, Any],
        relation_types_all: Sequence[str],
        *,
        ontology_ctx: str,
        kg_ctx: str,
    ) -> List[Dict[str, str]]:
        system_msg = {"role": "system", "content": self.llm_config.system_prompt}
        # Give the model the full list, but demand TOP-N only.
        user = "\n".join(
            [
                "You are a schema selection agent for document-level relation extraction.",
                f"Select the most likely relation types for this document, up to {self.params.max_relation_types_in_prompt}.",
                "Return ONLY valid JSON.",
                "",
                "## Document",
                self._doc_text_block(doc),
                "",
                "## Entities",
                self._entities_block(doc),
                "",
                "## Tool: Ontology context",
                ontology_ctx,
                "",
                "## Tool: KG context",
                kg_ctx,
                "",
                "## Candidate relation types (choose a subset)",
                "\n".join(f"- {r}" for r in relation_types_all),
                "",
                "## Output JSON schema",
                '{"selected": ["REL1", "REL2"], "notes": "short"}',
            ]
        )
        return [system_msg, {"role": "user", "content": user}]

    def _build_proposer_messages(
        self,
        doc: Dict[str, Any],
        relation_types_focus: Sequence[str],
        *,
        ontology_ctx: str,
        kg_ctx: str,
        web_ctx: str,
        wikidata_ctx: str,
        few_shots: Optional[List[Dict[str, Any]]],
        entity_subset: Optional[Set[str]] = None,
        proposer_id: int = 0,
    ) -> List[Dict[str, str]]:
        system_msg = {"role": "system", "content": self.llm_config.system_prompt}

        fewshot_block = self._few_shot_block(few_shots or [], relation_types_focus)

        header = [
            "You are a DocRE proposer agent.",
            "Extract relations between the PROVIDED entity IDs only.",
            "Output ONLY JSON (no markdown, no explanations).",
            "Endpoints MUST be entity IDs.",
            "Keys MUST match allowed relation types exactly.",
            f"Keep it concise: <= {self.params.proposer_max_pairs_per_rel} pairs per relation type.",
            "",
            f"(Proposer #{proposer_id}) If an Entities Subset is provided, focus on relations where HEAD is in that subset.",
            "",
        ]

        user_parts: List[str] = []
        user_parts += header

        if fewshot_block:
            user_parts += [
                "## Few-shot examples",
                fewshot_block,
                "",
            ]

        user_parts += [
            "## Document",
            self._doc_text_block(doc),
            "",
            "## Entities (IDs are canonical)",
            self._entities_block(doc, limit_entities=None),
            "",
        ]

        if entity_subset:
            user_parts += [
                "## Entities Subset (HEAD candidates)",
                "\n".join(f"- {e}" for e in sorted(entity_subset)),
                "",
            ]

        user_parts += [
            "## Allowed relation types",
            self._relation_schema_block(relation_types_focus),
            "",
            "## Tool: Ontology context",
            ontology_ctx,
            "",
            "## Tool: KG context",
            kg_ctx,
            "",
            "## Tool: Web context",
            web_ctx,
            "",
            "## Tool: Wikidata context",
            wikidata_ctx,
            "",
            "## Output format",
            'Return JSON: { "REL": [["E1","E2"], ...], ... }',
        ]

        return [system_msg, {"role": "user", "content": "\n".join(user_parts)}]

    def _build_verifier_messages(
        self,
        doc: Dict[str, Any],
        relation_types_focus: Sequence[str],
        *,
        merged_pred: Dict[str, Any],
        ontology_ctx: str,
        kg_ctx: str,
        web_ctx: str,
        wikidata_ctx: str,
    ) -> List[Dict[str, str]]:
        system_msg = {"role": "system", "content": self.llm_config.system_prompt}
        user = "\n".join(
            [
                "You are a verification agent for DocRE.",
                "Given the document + evidence contexts, REMOVE unsupported relation pairs.",
                "Do NOT add new pairs unless absolutely necessary; focus on pruning.",
                "Return ONLY valid JSON.",
                "",
                "## Document",
                self._doc_text_block(doc),
                "",
                "## Entities",
                self._entities_block(doc),
                "",
                "## Allowed relation types",
                self._relation_schema_block(relation_types_focus),
                "",
                "## Proposed relations (to verify)",
                json.dumps(merged_pred, ensure_ascii=False),
                "",
                "## Evidence: Ontology",
                ontology_ctx,
                "",
                "## Evidence: KG",
                kg_ctx,
                "",
                "## Evidence: Web",
                web_ctx,
                "",
                "## Evidence: Wikidata",
                wikidata_ctx,
            ]
        )
        return [system_msg, {"role": "user", "content": user}]

    # ----------------------------
    # Multi-proposer splitting
    # ----------------------------
    def _split_entities_for_proposers(self, doc: Dict[str, Any]) -> List[Set[str]]:
        ents = doc.get("entities") or {}
        if not isinstance(ents, dict) or not ents:
            return [set()]
        ids = list(ents.keys())
        n = max(1, int(self.params.num_proposers))
        if n == 1 or len(ids) <= 4:
            return [set(ids)]
        # chunk ids into n groups with slight overlap
        groups: List[Set[str]] = []
        size = max(1, len(ids) // n)
        for i in range(n):
            start = i * size
            end = len(ids) if i == n - 1 else (i + 1) * size
            g = set(ids[start:end])
            # overlap with previous
            if self.params.proposer_entity_overlap > 0 and i > 0:
                g |= set(ids[max(0, start - self.params.proposer_entity_overlap):start])
            groups.append(g)
        return groups

    # ----------------------------
    # Graph
    # ----------------------------
    def _build_graph(self):
        g = StateGraph(MARAGState)

        # --- nodes ---
        def node_plan(state: MARAGState) -> MARAGState:
            if not self.params.enable_planner:
                state["plan"] = {"use_ontology": True, "use_kg": True, "use_web": self.params.enable_web, "use_wikidata": self.params.enable_wikidata}
                return state
            if state.get("llm_calls_used", 0) >= self.params.max_llm_calls:
                state["plan"] = {"use_ontology": True, "use_kg": True, "use_web": False, "use_wikidata": False, "notes": "budget_exhausted"}
                return state
            msgs = self._build_planner_messages(state["doc"])
            raw = self._call_llm(msgs)
            state["llm_calls_used"] = state.get("llm_calls_used", 0) + 1
            parsed = self._extract_json_object(raw) or {}
            # defaults
            plan = {
                "use_ontology": bool(parsed.get("use_ontology", True)),
                "use_kg": bool(parsed.get("use_kg", True)),
                "use_web": bool(parsed.get("use_web", self.params.enable_web)),
                "use_wikidata": bool(parsed.get("use_wikidata", self.params.enable_wikidata)),
                "notes": str(parsed.get("notes", ""))[:200],
            }
            state["plan"] = plan
            return state

        def node_onto(state: MARAGState) -> MARAGState:
            use = state.get("plan", {}).get("use_ontology", True)
            state["ontology_ctx"] = self._ontology_context(state["doc"]) if use else "(ontology skipped)"
            return state

        def node_kg(state: MARAGState) -> MARAGState:
            use = state.get("plan", {}).get("use_kg", True)
            state["kg_ctx"] = self._kg_context(state["doc"]) if use else "(kg skipped)"
            return state

        def node_web(state: MARAGState) -> MARAGState:
            use = state.get("plan", {}).get("use_web", False)
            state["web_ctx"] = self._web_context(state["doc"]) if use else "(web skipped)"
            return state

        def node_wikidata(state: MARAGState) -> MARAGState:
            use = state.get("plan", {}).get("use_wikidata", False)
            state["wikidata_ctx"] = self._wikidata_context(state["doc"]) if use else "(wikidata skipped)"
            return state

        def node_select_reltypes(state: MARAGState) -> MARAGState:
            # If disabled, just take a truncated slice to avoid 90-rel prompt explosions
            all_rel = state["relation_types_all"]
            if not self.params.enable_relation_type_selector:
                state["relation_types_focus"] = list(all_rel)[: self.params.max_relation_types_in_prompt]
                return state

            if state.get("llm_calls_used", 0) >= self.params.max_llm_calls:
                state["relation_types_focus"] = list(all_rel)[: self.params.max_relation_types_in_prompt]
                return state

            msgs = self._build_relation_type_selector_messages(
                state["doc"],
                all_rel,
                ontology_ctx=state.get("ontology_ctx", ""),
                kg_ctx=state.get("kg_ctx", ""),
            )
            raw = self._call_llm(msgs)
            state["llm_calls_used"] = state.get("llm_calls_used", 0) + 1
            parsed = self._extract_json_object(raw) or {}
            sel = parsed.get("selected", [])
            if isinstance(sel, list):
                sel = [x for x in sel if isinstance(x, str) and x in set(all_rel)]
            else:
                sel = []
            if not sel:
                sel = list(all_rel)[: self.params.max_relation_types_in_prompt]
            state["relation_types_focus"] = sel[: self.params.max_relation_types_in_prompt]
            return state

        def node_proposers(state: MARAGState) -> MARAGState:
            # multi-proposer: each proposer gets one LLM call (if budget allows)
            focus = state["relation_types_focus"]
            doc = state["doc"]

            groups = self._split_entities_for_proposers(doc) if self.params.enable_multi_proposers else [None]
            preds: List[Dict[str, Any]] = []

            for i, subset in enumerate(groups):
                if state.get("llm_calls_used", 0) >= self.params.max_llm_calls:
                    break
                msgs = self._build_proposer_messages(
                    doc,
                    focus,
                    ontology_ctx=state.get("ontology_ctx", ""),
                    kg_ctx=state.get("kg_ctx", ""),
                    web_ctx=state.get("web_ctx", ""),
                    wikidata_ctx=state.get("wikidata_ctx", ""),
                    few_shots=state.get("few_shots", []),
                    entity_subset=subset if isinstance(subset, set) else None,
                    proposer_id=i + 1,
                )
                raw = self._call_llm(msgs)
                state["llm_calls_used"] = state.get("llm_calls_used", 0) + 1
                parsed = self._extract_json_object(raw)
                preds.append(parsed if isinstance(parsed, dict) else {r: [] for r in focus})

            state["proposer_preds"] = preds
            return state

        def node_merge(state: MARAGState) -> MARAGState:
            focus = state["relation_types_focus"]
            merged: Dict[str, List[List[str]]] = {r: [] for r in focus}

            # union (deduplicate pairs)
            seen = {r: set() for r in focus}
            for p in state.get("proposer_preds", []):
                if not isinstance(p, dict):
                    continue
                for r in focus:
                    pairs = p.get(r, [])
                    if not isinstance(pairs, list):
                        continue
                    for pair in pairs:
                        if not (isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, str) for x in pair)):
                            continue
                        key = (pair[0], pair[1])
                        if key not in seen[r]:
                            seen[r].add(key)
                            merged[r].append([pair[0], pair[1]])

            state["pred_merged_raw"] = merged
            return state

        def node_verify(state: MARAGState) -> MARAGState:
            if not self.params.enable_verifier:
                state["pred_verified_raw"] = state.get("pred_merged_raw", {})
                return state
            if state.get("llm_calls_used", 0) >= self.params.max_llm_calls:
                state["pred_verified_raw"] = state.get("pred_merged_raw", {})
                return state

            focus = state["relation_types_focus"]
            msgs = self._build_verifier_messages(
                state["doc"],
                focus,
                merged_pred=state.get("pred_merged_raw", {}),
                ontology_ctx=state.get("ontology_ctx", ""),
                kg_ctx=state.get("kg_ctx", ""),
                web_ctx=state.get("web_ctx", ""),
                wikidata_ctx=state.get("wikidata_ctx", ""),
            )
            raw = self._call_llm(msgs)
            state["llm_calls_used"] = state.get("llm_calls_used", 0) + 1
            parsed = self._extract_json_object(raw)
            state["pred_verified_raw"] = parsed if isinstance(parsed, dict) else state.get("pred_merged_raw", {})
            return state

        def node_finalize(state: MARAGState) -> MARAGState:
            doc = state["doc"]
            all_rel = state["relation_types_all"]
            focus = state["relation_types_focus"]
            raw_pred = state.get("pred_verified_raw") or state.get("pred_merged_raw") or {}

            # normalize ID-only on FULL schema
            # first, ensure raw_pred has at least focus keys
            for r in focus:
                raw_pred.setdefault(r, [])

            pred_ids_only = self._normalize_pred_endpoints_to_entity_ids(doc, raw_pred, focus)
            pred_focus_norm = self._normalize_relation_dict(pred_ids_only, focus)

            # expand to full schema keys
            full_out: Dict[str, List[List[str]]] = {r: [] for r in all_rel}
            for r in focus:
                full_out[r] = pred_focus_norm.get(r, [])

            full_out = self._normalize_relation_dict(full_out, all_rel)
            state["pred_norm"] = full_out

            if self.params.keep_debug:
                doc.setdefault("_debug", {})
                doc["_debug"]["marag"] = {
                    "plan": state.get("plan", {}),
                    "llm_calls_used": state.get("llm_calls_used", 0),
                    "relation_types_focus": focus,
                    "num_proposers": self.params.num_proposers,
                }
            return state

        # --- wiring: fan-out retrieval then LLM pipeline ---
        g.add_node("plan", node_plan)
        g.add_node("onto", node_onto)
        g.add_node("kg", node_kg)
        g.add_node("web", node_web)
        g.add_node("wikidata", node_wikidata)
        g.add_node("select_reltypes", node_select_reltypes)
        g.add_node("proposers", node_proposers)
        g.add_node("merge", node_merge)
        g.add_node("verify", node_verify)
        g.add_node("finalize", node_finalize)

        g.set_entry_point("plan")

        # retrieval fan-out (still executed sequentially by default runtime,
        # but logically separated & easy to parallelize later)
        g.add_edge("plan", "onto")
        g.add_edge("onto", "kg")
        g.add_edge("kg", "web")
        g.add_edge("web", "wikidata")

        # reasoning
        g.add_edge("wikidata", "select_reltypes")
        g.add_edge("select_reltypes", "proposers")
        g.add_edge("proposers", "merge")
        g.add_edge("merge", "verify")
        g.add_edge("verify", "finalize")
        g.add_edge("finalize", END)

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
        relation_types_all = list(relation_types) if relation_types else self._infer_relation_types_from_doc(doc)

        if self._graph is None:
            self._graph = self._build_graph()

        state: MARAGState = {
            "doc": doc,
            "relation_types_all": relation_types_all,
            "few_shots": few_shots or [],
            "llm_calls_used": 0,
        }

        out: MARAGState = self._graph.invoke(state)
        pred = out.get("pred_norm")
        if isinstance(pred, dict):
            return pred
        return {r: [] for r in relation_types_all}