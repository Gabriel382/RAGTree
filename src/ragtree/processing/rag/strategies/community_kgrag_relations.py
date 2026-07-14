# ragtree/processing/rag/strategies/community_kgrag_relations.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ragtree.kg.community_kgrag.retriever import CommunityKGRetriever
from ragtree.processing.rag.base_strategy import BaseRelationStrategy


# Fallback relation type if doc doesn't provide relation schema
DEFAULT_FALLBACK_RELATION_TYPE = "causal_relation"


@dataclass
class CommunityKGRAGParams:
    """
    Parameters controlling retrieval + prompt context size.
    """
    # Add these fields inside CommunityKGRAGParams (with comments)
    max_relation_types_in_prompt: int = 20   # prune schema shown to LLM
    min_relation_types_in_prompt: int = 6    # never show fewer than this
    # Community stage
    top_communities: int = 50
    delta_percent: Optional[float] = None  # if set, overrides top_communities via percentage
    # Sentence stage
    top_sentences: int = 12
    lambda_percent: Optional[float] = None  # if set, keeps only a percent of ranked candidates
    use_sentence_faiss: bool = True
    # Prompt truncation
    max_ctx_chars: int = 8000

    # Query building (from doc)
    max_sentences_in_query: int = 3
    max_entity_mentions_in_query: int = 6


class CommunityKGRAGRelationStrategy(BaseRelationStrategy):
    """
    CommunityKG-RAG for DocRE:
      - Build a query from (title + first sentences + entity triggers)
      - Retrieve evidence sentences using KG communities
      - Inject evidence into the LLM prompt
      - Parse strict JSON relations output
    """

    def __init__(
        self,
        llm_config,
        *,
        retriever: CommunityKGRetriever,
        params: Optional[CommunityKGRAGParams] = None,
    ) -> None:
        # BaseRelationStrategy handles llm_config + _call_llm + normalization helpers
        super().__init__(llm_config)
        self.retriever = retriever
        self.params = params or CommunityKGRAGParams()

    # ---------------------------------------------------------------------
    # Schema inference
    # ---------------------------------------------------------------------

    def _infer_relation_types_from_doc(self, doc: Dict[str, Any]) -> List[str]:
        """
        Use doc['relations'] keys (gold schema), otherwise fallback.
        """
        rels = doc.get("relations")
        if isinstance(rels, dict) and rels:
            return list(rels.keys())
        return [DEFAULT_FALLBACK_RELATION_TYPE]

    # ---------------------------------------------------------------------
    # Robust JSON extraction
    # ---------------------------------------------------------------------

    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract a JSON object from model output, allowing for accidental markdown fences.
        """
        if not isinstance(text, str) or not text.strip():
            return None

        s = text.strip()
        # Remove code fences if present
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)

        # First try full parse
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        # Fallback: find first {...} block
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    # ---------------------------------------------------------------------
    # Query building for retrieval
    # ---------------------------------------------------------------------

    def _build_query(self, doc: Dict[str, Any]) -> str:
        """
        Build a retrieval query from:
          - title (if any)
          - first N sentences (if doc['sentences'] exists)
          - first mention trigger_word for up to M entities
        """
        parts: List[str] = []

        # Title
        title = doc.get("title")
        if isinstance(title, str) and title.strip():
            parts.append(title.strip())

        # Sentences (preferred)
        sents = doc.get("sentences")
        if isinstance(sents, list) and sents and all(isinstance(x, str) for x in sents):
            parts.extend(sents[: self.params.max_sentences_in_query])
        else:
            # Fallback to doc['text']
            txt = doc.get("text")
            if isinstance(txt, str) and txt.strip():
                parts.append(txt[:800])

        # Entity trigger words
        ents = doc.get("entities") or {}
        if isinstance(ents, dict):
            count = 0
            for _, ent in ents.items():
                if not isinstance(ent, dict):
                    continue
                mentions = ent.get("mentions") or []
                if not isinstance(mentions, list):
                    mentions = [mentions]

                # Use only the first mention text to avoid inflating query
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

    # ---------------------------------------------------------------------
    # Evidence formatting
    # ---------------------------------------------------------------------

    def _format_evidence(self, evidence: List[Any]) -> str:
        """
        Convert retrieved evidence sentences into a prompt-ready context string.
        """
        lines: List[str] = []
        for e in evidence:
            # EvidenceSentence has attributes: text, document_id, sentence_id, community_id, score
            txt = (getattr(e, "text", "") or "").strip().replace("\n", " ")
            did = getattr(e, "document_id", None) or "unknown_doc"
            sid = getattr(e, "sentence_id", None) or "unknown_sid"
            cid = getattr(e, "community_id", None)
            score = getattr(e, "score", None)

            # Keep it short but informative
            if score is not None:
                lines.append(f"- [comm={cid} | score={score:.4f} | doc={did} | sid={sid}] {txt}")
            else:
                lines.append(f"- [comm={cid} | doc={did} | sid={sid}] {txt}")

        ctx = "\n".join(lines)
        if len(ctx) > self.params.max_ctx_chars:
            ctx = ctx[: self.params.max_ctx_chars] + "\n... (truncated)"
        return ctx if ctx else "(no evidence retrieved)"

    # ---------------------------------------------------------------------
    # Local prompt blocks (since BaseRelationStrategy doesn't provide them)
    # ---------------------------------------------------------------------

    def _doc_text_block(self, doc: Dict[str, Any]) -> str:
        """
        Render document text consistently from common schemas.
        """
        title = doc.get("title")
        title_line = f"Title: {title}\n" if isinstance(title, str) and title.strip() else ""

        sents = doc.get("sentences")
        if isinstance(sents, list) and sents and all(isinstance(x, str) for x in sents):
            body = "\n".join(f"- {s}" for s in sents)
            return title_line + body

        txt = doc.get("text")
        if isinstance(txt, str) and txt.strip():
            return title_line + txt

        return title_line + "(no document text found)"

    def _entities_block(self, doc: Dict[str, Any], *, max_mentions_per_entity: int = 2) -> str:
        """
        Render entity IDs with a couple of mentions (DocRE style).
        """
        ents = doc.get("entities") or {}
        if not isinstance(ents, dict) or not ents:
            return "(no entities found)"

        lines: List[str] = []
        for ent_id, ent in ents.items():
            if not isinstance(ent, dict):
                continue

            ent_type = ent.get("type", "")
            mentions = ent.get("mentions", [])
            if not isinstance(mentions, list):
                mentions = [mentions]

            # Show limited mentions for readability
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
                if shown >= max_mentions_per_entity:
                    break

            # If no mention dicts, still list the entity
            if shown == 0:
                lines.append(f"{ent_id}\tTYPE={ent_type}")

        return "\n".join(lines) if lines else "(no entities found)"

    def _relation_schema_block(self, relation_types: Sequence[str]) -> str:
        """
        Render allowed relation types list.
        """
        if not relation_types:
            return "- (none)"
        return "\n".join(f"- {r}" for r in relation_types)

    def _few_shot_block(self, few_shots: List[Dict[str, Any]], *, max_shots: int = 3) -> str:
        """
        Local few-shot formatter (robust: does not depend on BaseRelationStrategy).
        Each example includes:
          - text
          - entities
          - gold relations (as JSON)
        """
        blocks: List[str] = []
        for i, ex in enumerate(few_shots[:max_shots], start=1):
            ex_text = self._doc_text_block(ex)
            ex_ents = self._entities_block(ex)
            rels = ex.get("relations")
            if not isinstance(rels, dict):
                rels = {}
            rels_json = json.dumps(rels, ensure_ascii=False)

            blocks.append(
                "\n".join(
                    [
                        f"### Few-shot example {i}",
                        "## Document",
                        ex_text,
                        "",
                        "## Entities",
                        ex_ents,
                        "",
                        "## Gold relations (JSON)",
                        rels_json,
                    ]
                )
            )

        return "\n\n".join(blocks) + "\n"

    # ---------------------------------------------------------------------
    # Prompt assembly
    # ---------------------------------------------------------------------
    def _build_messages(
        self,
        doc: Dict[str, Any],
        relation_types: Sequence[str],
        *,
        evidence_ctx: str,
        prompt_relation_types: Optional[Sequence[str]] = None,
        few_shots: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, str]]:
        """
        Create system + user messages for the LLM call.
        """
        system_msg = {"role": "system", "content": self.llm_config.system_prompt}

        user_parts: List[str] = [
            "You will extract document-level relations between the PROVIDED entity IDs only.",
            "You MUST output ONLY valid JSON (no markdown, no explanations).",
            "You MUST use ONLY the PROVIDED entity IDs in output pairs (no literals).",
            "",
        ]

        # Optional few-shot block
        if few_shots:
            user_parts.append(self._few_shot_block(few_shots))

        # Main task
        user_parts += [
            "## Document",
            self._doc_text_block(doc),
            "",
            "## Entities (IDs are canonical  use them in output)",
            self._entities_block(doc),
            "",
            "## Allowed relation types (output keys must match these exactly)",
            self._relation_schema_block(prompt_relation_types or relation_types),
            "",
            "## Tool: CommunityKG-RAG evidence sentences (retrieved from corpus via KG communities)",
            evidence_ctx,
            "",
            "## Output format (JSON only, no extra text)",
            "You may output ONLY relation types you are confident about.",
            "Any missing relation types will be treated as empty.",
            "Return a JSON object with keys as relation types and values as lists of [HEAD_ID, TAIL_ID] pairs.",
            "Example:",
            '{"P17 : country": [["E1","E2"]]}',
        ]

        return [system_msg, {"role": "user", "content": "\n".join(user_parts)}]

    # ---------------------------------------------------------------------
    # Public API used by relations_runner
    # ---------------------------------------------------------------------

    def predict_relations(
        self,
        doc: Dict[str, Any],
        relation_types: Optional[Sequence[str]] = None,
        *,
        few_shots: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, List[List[str]]]:
        """
        Predict relations for one document.
        """
        # Inside predict_relations(...)

        rel_types_full = list(relation_types) if relation_types else self._infer_relation_types_from_doc(doc)

        # Retrieval
        query = self._build_query(doc)
        evidence = self.retriever.retrieve(
            query,
            top_communities=self.params.top_communities,
            delta_percent=self.params.delta_percent,
            top_sentences=self.params.top_sentences,
            lambda_percent=self.params.lambda_percent,
            use_sentence_faiss=self.params.use_sentence_faiss,
        )
        ctx = self._format_evidence(evidence)

        # NEW: prune schema shown to the LLM
        rel_types_prompt = self._prune_relation_types_for_prompt(doc, rel_types_full, ctx)

        # Build prompt with pruned schema (but keep full schema for final normalization)
        msgs = self._build_messages(
            doc,
            rel_types_full,
            evidence_ctx=ctx,
            prompt_relation_types=rel_types_prompt,
            few_shots=few_shots,
        )
        raw = self._call_llm(msgs)

        parsed = self._extract_json_object(raw)
        if not isinstance(parsed, dict):
            return {r: [] for r in rel_types_full}

        # Normalize endpoints if you keep that step
        try:
            parsed = self._normalize_pred_endpoints_to_entity_ids(doc, parsed, rel_types_full, keep_debug=True)
        except Exception:
            pass

        # IMPORTANT: normalize to FULL schema, not pruned schema
        return self._normalize_relation_dict(parsed, rel_types_full)
    
    # Add these methods inside CommunityKGRAGRelationStrategy

    def _relation_label_tokens(self, rel: str) -> List[str]:
        """
        Extract lexical tokens from a DocRED relation label like:
        'P17 : country' -> ['country']
        'P6 : head of government' -> ['head', 'government']
        """
        # Keep everything after ':' if present, else use full string
        label = rel.split(":", 1)[-1] if ":" in rel else rel
        label = label.replace("_", " ").strip().lower()
        # Simple tokenization
        toks = re.findall(r"[a-z0-9]+", label)
        # Drop extremely common junk tokens
        stop = {"of", "the", "a", "an", "in", "on", "to", "for", "and", "or"}
        toks = [t for t in toks if t not in stop]
        return toks


    def _text_for_scoring(self, doc: Dict[str, Any], evidence_ctx: str) -> str:
        """
        Build a scoring text used ONLY for pruning relation types.
        """
        parts: List[str] = []
        # doc title + first sentences
        title = doc.get("title")
        if isinstance(title, str):
            parts.append(title)
        sents = doc.get("sentences")
        if isinstance(sents, list) and all(isinstance(s, str) for s in sents):
            parts.append(" ".join(sents[:8]))
        else:
            txt = doc.get("text")
            if isinstance(txt, str):
                parts.append(txt[:2000])

        # include evidence (already truncated by max_ctx_chars)
        if isinstance(evidence_ctx, str) and evidence_ctx:
            parts.append(evidence_ctx[:2000])

        return " ".join(parts).lower()


    def _prune_relation_types_for_prompt(
        self,
        doc: Dict[str, Any],
        relation_types: Sequence[str],
        evidence_ctx: str,
    ) -> List[str]:
        """
        Prune the (possibly huge) relation schema to a smaller list for the LLM prompt.
        We do NOT change the final output schema; we only reduce what the LLM sees.
        """
        # If already small, keep as-is
        if len(relation_types) <= self.params.max_relation_types_in_prompt:
            return list(relation_types)

        text = self._text_for_scoring(doc, evidence_ctx)

        # Score each relation by number of label tokens appearing in text
        scored: List[tuple[str, float]] = []
        for r in relation_types:
            toks = self._relation_label_tokens(r)
            if not toks:
                scored.append((r, 0.0))
                continue
            hits = sum(1.0 for t in toks if t in text)
            # mild prior that shorter labels are less informative
            score = hits + 0.05 * len(toks)
            scored.append((r, float(score)))

        # Sort by score desc, then stable by name
        scored.sort(key=lambda x: (x[1], x[0]), reverse=True)

        # Keep top-K but never fewer than min_relation_types_in_prompt
        k = max(self.params.min_relation_types_in_prompt, self.params.max_relation_types_in_prompt)
        kept = [r for (r, _s) in scored[:k]]

        return kept
