from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, TypedDict

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
    """
    Recall-oriented MA-RAG parameters.

    The key idea is:
    - ontology + KG are ALWAYS included
    - one base extractor produces strong initial predictions
    - one gap critic proposes missing relations
    - one optional verifier lightly prunes only clearly unsupported pairs
    """
    # LLM budget
    max_llm_calls: int = 3

    # Tool toggles
    enable_ontology: bool = True
    enable_kg: bool = True
    enable_web: bool = False
    enable_wikidata: bool = False

    # Agent toggles
    enable_gap_critic: bool = True
    enable_verifier: bool = True

    # Prompt budget controls
    max_sentences_in_prompt: Optional[int] = None
    kg_max_triples: int = 40
    web_max_chars: int = 1400
    wikidata_max_chars: int = 1400
    max_pairs_per_relation: int = 20

    # HTTP safety
    http_timeout_sec: float = 4.0
    http_user_agent: str = "ragtree-marag/0.2"

    # Debug
    keep_debug: bool = True
    verbose: bool = False


class MARAGState(TypedDict, total=False):
    """
    LangGraph state for the recall-oriented MA-RAG.
    """
    doc: Dict[str, Any]
    relation_types_all: List[str]
    few_shots: List[Dict[str, Any]]

    # Retrieved contexts
    ontology_ctx: str
    kg_ctx: str
    web_ctx: str
    wikidata_ctx: str

    # LLM usage
    llm_calls_used: int

    # Agent outputs
    pred_base_raw: Dict[str, Any]
    pred_gap_raw: Dict[str, Any]
    pred_merged_raw: Dict[str, Any]
    pred_verified_raw: Dict[str, Any]
    pred_norm: Dict[str, List[List[str]]]


class MARagRelationStrategy(BaseRelationStrategy):
    """
    Recall-oriented MA-RAG.

    Design:
    1) Always retrieve ontology + KG.
    2) Base extractor agent behaves like the strong single-agent hybrid.
    3) Gap critic agent searches specifically for missed relations.
    4) Light verifier agent only removes clearly unsupported pairs.
    5) Final merge + normalization.

    This is intentionally designed to improve recall without destroying precision.
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
        """
        Initialize the MA-RAG strategy.
        """
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

        # Small caches for optional web tools
        self._cache_wikipedia: Dict[str, str] = {}
        self._cache_wikidata: Dict[str, str] = {}

    # ----------------------------
    # LLM caller
    # ----------------------------
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Self-contained LLM caller.

        Supported backends:
          - ollama
          - vllm (OpenAI-compatible /v1/chat/completions)
          - openrouter (OpenAI-compatible /chat/completions with /api/v1 base)
        """
        backend = str(getattr(self.llm_config, "backend", "ollama") or "ollama").lower()
        model = str(getattr(self.llm_config, "model", "") or "").strip()
        temperature = float(getattr(self.llm_config, "temperature", 0.0) or 0.0)
        max_tokens = int(getattr(self.llm_config, "max_tokens", 1024) or 1024)
        base_url = getattr(self.llm_config, "base_url", None)
        api_key = getattr(self.llm_config, "api_key", None)

        if not model:
            raise ValueError("llm_config.model is missing")

        # Ollama
        if backend == "ollama":
            host = str(base_url or "http://localhost:11434").rstrip("/")
            url = f"{host}/api/chat"
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            resp = requests.post(
                url,
                json=payload,
                timeout=max(30.0, self.params.http_timeout_sec * 10),
                headers={"User-Agent": self.params.http_user_agent},
            )
            resp.raise_for_status()
            data = resp.json()

            message = data.get("message", {})
            content = message.get("content")
            if content is not None:
                return str(content)

            raise ValueError(f"No message content returned by backend '{backend}': {data}")

        # OpenAI-compatible APIs
        if backend in {"vllm", "openrouter"}:
            if backend == "openrouter":
                final_base_url = str(base_url or "https://openrouter.ai/api/v1").rstrip("/")
                final_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
                url = f"{final_base_url}/chat/completions"
            else:
                final_base_url = str(base_url or "http://localhost:8000").rstrip("/")
                final_api_key = api_key or "dummy"
                url = f"{final_base_url}/v1/chat/completions"

            headers = {
                "Content-Type": "application/json",
                "User-Agent": self.params.http_user_agent,
            }
            if final_api_key:
                headers["Authorization"] = f"Bearer {final_api_key}"

            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            resp = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=max(30.0, self.params.http_timeout_sec * 10),
            )
            resp.raise_for_status()
            data = resp.json()

            choices = data.get("choices", [])
            if not choices:
                raise ValueError(f"No choices returned by backend '{backend}': {data}")

            choice0 = choices[0]
            message = choice0.get("message", {}) or {}

            # Standard content
            content = message.get("content")
            if content is not None:
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts: List[str] = []
                    for item in content:
                        if isinstance(item, dict):
                            txt = item.get("text")
                            if txt:
                                parts.append(str(txt))
                        elif isinstance(item, str):
                            parts.append(item)
                    if parts:
                        return "\n".join(parts)

            # Some servers expose reasoning separately
            reasoning = message.get("reasoning")
            if reasoning is not None:
                return str(reasoning)

            # Fallbacks
            if "text" in choice0 and choice0["text"] is not None:
                return str(choice0["text"])

            delta = choice0.get("delta")
            if isinstance(delta, dict):
                delta_content = delta.get("content")
                if delta_content is not None:
                    return str(delta_content)

            finish_reason = choice0.get("finish_reason")
            raise ValueError(
                f"No usable text returned by backend '{backend}'. "
                f"finish_reason={finish_reason}, raw_response={data}"
            )

        raise ValueError(f"Unsupported backend '{backend}' for MA-RAG")

    # ----------------------------
    # Basic blocks
    # ----------------------------
    def _infer_relation_types_from_doc(self, doc: Dict[str, Any]) -> List[str]:
        """
        Infer allowed relation types from the gold schema if present.
        """
        rels = doc.get("relations")
        if isinstance(rels, dict):
            keys = list(rels.keys())
            if keys:
                return keys
        return [DEFAULT_FALLBACK_RELATION_TYPE]

    def _doc_text_block(self, doc: Dict[str, Any]) -> str:
        """
        Build the document text block for prompting.
        """
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

    def _entities_block(self, doc: Dict[str, Any]) -> str:
        """
        Build the entities block.
        """
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
        """
        Build the allowed relation schema block.
        """
        return "\n".join(f'- "{r}"' for r in relation_types)

    def _few_shot_block(self, few_shots: List[Dict[str, Any]], allowed_relation_types: Sequence[str]) -> str:
        """
        Build a compact few-shot block.
        """
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

            ents = ex.get("entities") or {}
            ent_lines: List[str] = []
            if isinstance(ents, dict):
                for ent_id, ent in list(ents.items())[:12]:
                    if not isinstance(ent, dict):
                        continue
                    mentions = ent.get("mentions") or [{}]
                    m0 = mentions[0] if isinstance(mentions, list) and mentions else {}
                    if isinstance(m0, dict):
                        trig = m0.get("trigger_word") or m0.get("text") or ""
                        ent_lines.append(f"{ent_id}\tTYPE={ent.get('type', '')}\tTRIGGER={trig}")

            if not ent_lines:
                ent_lines = ["(no entities)"]

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
        """
        Extract a JSON object from LLM output.
        """
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
    # Endpoint normalization
    # ----------------------------
    def _build_literal_to_entity_index(self, doc: Dict[str, Any]) -> Dict[str, str]:
        """
        Map unique normalized mention strings to entity IDs.
        """
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
        """
        Enforce entity-ID-only predictions.
        """
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
    # Context resolvers
    # ----------------------------
    def _resolve_doc_ontology_links(self, doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Resolve ontology links from the document or the external map.
        """
        if isinstance(doc.get("ontology_links"), dict):
            return doc["ontology_links"]

        doc_id = doc.get("document_id") or doc.get("id")
        if doc_id and doc_id in self.ontology_links_by_docid:
            return self.ontology_links_by_docid[doc_id]

        return None

    def _resolve_doc_kg_triples(self, doc: Dict[str, Any]) -> List[List[str]]:
        """
        Resolve KG triples from the document or the external map.
        """
        kgc = doc.get("_kg_context", {})
        if isinstance(kgc, dict) and isinstance(kgc.get("triples"), list):
            return kgc.get("triples") or []

        doc_id = doc.get("document_id") or doc.get("id")
        if doc_id and doc_id in self.kg_triples_by_docid:
            triples = self.kg_triples_by_docid[doc_id]
            return triples if isinstance(triples, list) else []

        return []

    def _ontology_context(self, doc: Dict[str, Any]) -> str:
        """
        Retrieve ontology context.
        """
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
        """
        Retrieve KG context.
        """
        if not self.params.enable_kg:
            return "(kg disabled)"

        triples = self._resolve_doc_kg_triples(doc)
        if not triples:
            return "(no kg triples found)"

        triples = [t for t in triples if isinstance(t, list) and len(t) == 3][: self.params.kg_max_triples]
        lines = [f"- {h} | {r} | {t}" for h, r, t in triples]
        return "### KG triples\n" + "\n".join(lines)

    # ----------------------------
    # Optional web tools
    # ----------------------------
    def _http_get_json(self, url: str, *, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Small helper for GET JSON requests.
        """
        try:
            headers = {"User-Agent": self.params.http_user_agent}
            resp = requests.get(url, params=params, headers=headers, timeout=self.params.http_timeout_sec)
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception:
            return None

    def _wikipedia_summary(self, title: str) -> str:
        """
        Fetch a short Wikipedia summary by title.
        """
        key = title.strip().lower()
        if not key:
            return "(no title)"

        if key in self._cache_wikipedia:
            return self._cache_wikipedia[key]

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
        """
        Fetch small Wikidata snippets using a few entity mentions.
        """
        ents = doc.get("entities") or {}
        if not isinstance(ents, dict) or not ents:
            return "(no entities)"

        picks: List[str] = []
        for _, ent in ents.items():
            if not isinstance(ent, dict):
                continue

            mentions = ent.get("mentions") or [{}]
            m0 = mentions[0] if isinstance(mentions, list) and mentions else {}
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
        """
        Retrieve optional web context.
        """
        if not self.params.enable_web:
            return "(web disabled)"

        title = str(doc.get("title", "") or "").strip()
        if not title:
            return "(no title for wikipedia)"

        return self._wikipedia_summary(title)

    def _wikidata_context(self, doc: Dict[str, Any]) -> str:
        """
        Retrieve optional Wikidata context.
        """
        if not self.params.enable_wikidata:
            return "(wikidata disabled)"
        return self._wikidata_snippets_from_entities(doc)

    # ----------------------------
    # Message builders
    # ----------------------------
    def _build_base_extractor_messages(
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
        """
        Build the base extractor prompt.

        This is intentionally very close in spirit to the strong single-agent hybrid.
        """
        system_msg = {"role": "system", "content": self.llm_config.system_prompt}

        fewshot_block = self._few_shot_block(few_shots or [], relation_types)

        user_parts: List[str] = [
            "You are an expert at document-level relation extraction.",
            "Extract relations between the PROVIDED entity IDs only.",
            "You MUST output ONLY valid JSON (no markdown, no explanations).",
            "You MUST use ONLY the PROVIDED entity IDs in output pairs.",
            "Output keys MUST match the allowed relation types exactly.",
            "Values MUST be lists of [HEAD_ID, TAIL_ID] pairs.",
            "If a relation type has no valid pair, output an empty list for that key.",
            "",
        ]

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
            self._entities_block(doc),
            "",
            "## Allowed relation types",
            self._relation_schema_block(relation_types),
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
            'Return ONE JSON object: { "REL": [["E1","E2"], ...], ... }',
        ]

        return [system_msg, {"role": "user", "content": "\n".join(user_parts)}]

    def _build_gap_critic_messages(
        self,
        doc: Dict[str, Any],
        relation_types: Sequence[str],
        *,
        current_pred: Dict[str, Any],
        ontology_ctx: str,
        kg_ctx: str,
        web_ctx: str,
        wikidata_ctx: str,
    ) -> List[Dict[str, str]]:
        """
        Build the gap critic prompt.

        This agent is recall-oriented:
        it should propose likely missing relations, not re-do everything from scratch.
        """
        system_msg = {"role": "system", "content": self.llm_config.system_prompt}

        user = "\n".join(
            [
                "You are a recall-oriented critic for document-level relation extraction.",
                "Your task is to find PLAUSIBLE MISSING relations that were not captured yet.",
                "Do NOT remove existing predictions.",
                "Only propose additional relation pairs if they are supported by the document and/or the provided contexts.",
                "You MUST output ONLY valid JSON.",
                "Use ONLY the provided entity IDs.",
                "Output keys MUST match the allowed relation types exactly.",
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
                "## Current predictions",
                json.dumps(current_pred, ensure_ascii=False),
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
                "Return ONLY additional candidate pairs in JSON form.",
                'Example: { "REL": [["E1","E2"]], "REL2": [] }',
            ]
        )

        return [system_msg, {"role": "user", "content": user}]

    def _build_verifier_messages(
        self,
        doc: Dict[str, Any],
        relation_types: Sequence[str],
        *,
        merged_pred: Dict[str, Any],
        ontology_ctx: str,
        kg_ctx: str,
        web_ctx: str,
        wikidata_ctx: str,
    ) -> List[Dict[str, str]]:
        """
        Build the verifier prompt.

        This verifier is intentionally light:
        it should only remove pairs that are clearly unsupported or impossible.
        """
        system_msg = {"role": "system", "content": self.llm_config.system_prompt}

        user = "\n".join(
            [
                "You are a light verification agent for document-level relation extraction.",
                "Your task is to REMOVE only clearly unsupported or impossible relation pairs.",
                "Be conservative in pruning: keep plausible pairs.",
                "Do NOT aggressively reduce recall.",
                "Do NOT invent many new pairs.",
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
                "## Candidate predictions to verify",
                json.dumps(merged_pred, ensure_ascii=False),
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
                'Return ONE JSON object: { "REL": [["E1","E2"], ...], ... }',
            ]
        )

        return [system_msg, {"role": "user", "content": user}]

    # ----------------------------
    # Merge helpers
    # ----------------------------
    def _merge_prediction_dicts(
        self,
        relation_types: Sequence[str],
        *preds: Dict[str, Any],
    ) -> Dict[str, List[List[str]]]:
        """
        Merge multiple prediction dicts by union of string pairs.
        """
        merged: Dict[str, List[List[str]]] = {r: [] for r in relation_types}
        seen: Dict[str, Set[tuple[str, str]]] = {r: set() for r in relation_types}

        for pred in preds:
            if not isinstance(pred, dict):
                continue

            for r in relation_types:
                pairs = pred.get(r, [])
                if not isinstance(pairs, list):
                    continue

                for pair in pairs:
                    if not (
                        isinstance(pair, list)
                        and len(pair) == 2
                        and all(isinstance(x, str) for x in pair)
                    ):
                        continue

                    key = (pair[0], pair[1])
                    if key not in seen[r]:
                        seen[r].add(key)
                        merged[r].append([pair[0], pair[1]])

        return merged

    # ----------------------------
    # Graph
    # ----------------------------
    def _build_graph(self):
        """
        Build the LangGraph flow.
        """
        g = StateGraph(MARAGState)

        def node_onto(state: MARAGState) -> MARAGState:
            state["ontology_ctx"] = self._ontology_context(state["doc"])
            return state

        def node_kg(state: MARAGState) -> MARAGState:
            state["kg_ctx"] = self._kg_context(state["doc"])
            return state

        def node_web(state: MARAGState) -> MARAGState:
            state["web_ctx"] = self._web_context(state["doc"])
            return state

        def node_wikidata(state: MARAGState) -> MARAGState:
            state["wikidata_ctx"] = self._wikidata_context(state["doc"])
            return state

        def node_base_extract(state: MARAGState) -> MARAGState:
            rel_types = state["relation_types_all"]

            if state.get("llm_calls_used", 0) >= self.params.max_llm_calls:
                state["pred_base_raw"] = {r: [] for r in rel_types}
                return state

            msgs = self._build_base_extractor_messages(
                state["doc"],
                rel_types,
                ontology_ctx=state.get("ontology_ctx", ""),
                kg_ctx=state.get("kg_ctx", ""),
                web_ctx=state.get("web_ctx", ""),
                wikidata_ctx=state.get("wikidata_ctx", ""),
                few_shots=state.get("few_shots", []),
            )
            raw = self._call_llm(msgs)
            state["llm_calls_used"] = state.get("llm_calls_used", 0) + 1

            parsed = self._extract_json_object(raw)
            state["pred_base_raw"] = parsed if isinstance(parsed, dict) else {r: [] for r in rel_types}
            return state

        def node_gap_critic(state: MARAGState) -> MARAGState:
            rel_types = state["relation_types_all"]

            if not self.params.enable_gap_critic:
                state["pred_gap_raw"] = {r: [] for r in rel_types}
                return state

            if state.get("llm_calls_used", 0) >= self.params.max_llm_calls:
                state["pred_gap_raw"] = {r: [] for r in rel_types}
                return state

            msgs = self._build_gap_critic_messages(
                state["doc"],
                rel_types,
                current_pred=state.get("pred_base_raw", {}),
                ontology_ctx=state.get("ontology_ctx", ""),
                kg_ctx=state.get("kg_ctx", ""),
                web_ctx=state.get("web_ctx", ""),
                wikidata_ctx=state.get("wikidata_ctx", ""),
            )
            raw = self._call_llm(msgs)
            state["llm_calls_used"] = state.get("llm_calls_used", 0) + 1

            parsed = self._extract_json_object(raw)
            state["pred_gap_raw"] = parsed if isinstance(parsed, dict) else {r: [] for r in rel_types}
            return state

        def node_merge(state: MARAGState) -> MARAGState:
            rel_types = state["relation_types_all"]

            merged = self._merge_prediction_dicts(
                rel_types,
                state.get("pred_base_raw", {}),
                state.get("pred_gap_raw", {}),
            )
            state["pred_merged_raw"] = merged
            return state

        def node_verify(state: MARAGState) -> MARAGState:
            rel_types = state["relation_types_all"]

            if not self.params.enable_verifier:
                state["pred_verified_raw"] = state.get("pred_merged_raw", {})
                return state

            if state.get("llm_calls_used", 0) >= self.params.max_llm_calls:
                state["pred_verified_raw"] = state.get("pred_merged_raw", {})
                return state

            msgs = self._build_verifier_messages(
                state["doc"],
                rel_types,
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
            rel_types = state["relation_types_all"]
            raw_pred = state.get("pred_verified_raw") or state.get("pred_merged_raw") or {}

            for r in rel_types:
                raw_pred.setdefault(r, [])

            pred_ids_only = self._normalize_pred_endpoints_to_entity_ids(doc, raw_pred, rel_types)
            pred_norm = self._normalize_relation_dict(pred_ids_only, rel_types)
            state["pred_norm"] = pred_norm

            if self.params.keep_debug:
                doc.setdefault("_debug", {})
                doc["_debug"]["marag"] = {
                    "llm_calls_used": state.get("llm_calls_used", 0),
                    "base_pred_raw": state.get("pred_base_raw", {}),
                    "gap_pred_raw": state.get("pred_gap_raw", {}),
                    "merged_pred_raw": state.get("pred_merged_raw", {}),
                }

            return state

        g.add_node("onto", node_onto)
        g.add_node("kg", node_kg)
        g.add_node("web", node_web)
        g.add_node("wikidata", node_wikidata)
        g.add_node("base_extract", node_base_extract)
        g.add_node("gap_critic", node_gap_critic)
        g.add_node("merge", node_merge)
        g.add_node("verify", node_verify)
        g.add_node("finalize", node_finalize)

        g.set_entry_point("onto")
        g.add_edge("onto", "kg")
        g.add_edge("kg", "web")
        g.add_edge("web", "wikidata")
        g.add_edge("wikidata", "base_extract")
        g.add_edge("base_extract", "gap_critic")
        g.add_edge("gap_critic", "merge")
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
        """
        Predict document-level relations.
        """
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