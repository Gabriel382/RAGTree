# ragtree/kg/services/document_kg.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass
class Triple:
    head: str
    rel: str
    tail: str


class InMemoryDocumentKG:
    """
    Simple in-memory KG built from a preprocessed JSONL:
      - nodes: entity IDs
      - edges: (head_id, relation_type, tail_id)

    This is deterministic, reproducible, and fast enough for research runs.
    Later, you can swap this out with a BYOKG GraphStore/Neptune adapter
    WITHOUT changing the agentic strategy.
    """

    def __init__(self) -> None:
        self._adj: Dict[str, List[Triple]] = {}
        self._ent_text_index: Dict[str, Set[str]] = {}  # normalized mention -> entity_ids
        self._built: bool = False

    @staticmethod
    def _norm_text(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower())

    def add_triple(self, h: str, r: str, t: str) -> None:
        self._adj.setdefault(h, []).append(Triple(h, r, t))

    def add_entity_mentions(self, ent_id: str, mentions: Sequence[Dict[str, Any]]) -> None:
        for m in mentions:
            if not isinstance(m, dict):
                continue
            texts: List[str] = []
            if m.get("trigger_word"):
                texts.append(str(m["trigger_word"]))
            if m.get("text"):
                texts.append(str(m["text"]))
            for txt in texts:
                key = self._norm_text(txt)
                if not key:
                    continue
                self._ent_text_index.setdefault(key, set()).add(ent_id)

    def build_from_jsonl(
        self,
        path: Path,
        *,
        doc_types: Sequence[str] | str = "train",
        skip: int = 0,
        limit: Optional[int] = None,
        require_gold_relations: bool = True,
    ) -> None:
        """
        Build KG from docs in JSONL.

        - doc_types: "all" or ["train","dev",...]
        - skip/limit apply AFTER doc type filtering
        - require_gold_relations: if True, only add triples when doc["relations"] is non-empty dict
        """
        if self._built:
            return

        def type_ok(doc: Dict[str, Any]) -> bool:
            if doc_types == "all":
                return True
            allowed = set(doc_types)
            return doc.get("type") in allowed

        kept = 0
        seen_after_filter = 0

        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)

                if not type_ok(doc):
                    continue

                # skip/limit after filtering
                if seen_after_filter < skip:
                    seen_after_filter += 1
                    continue
                seen_after_filter += 1

                if limit is not None and kept >= limit:
                    break

                entities = doc.get("entities") or {}
                if isinstance(entities, dict):
                    for ent_id, ent in entities.items():
                        if not isinstance(ent, dict):
                            continue
                        mentions = ent.get("mentions") or []
                        if isinstance(mentions, list):
                            self.add_entity_mentions(ent_id, mentions)

                rels = doc.get("relations")
                if require_gold_relations:
                    if not isinstance(rels, dict) or not rels:
                        continue

                if isinstance(rels, dict):
                    for rtype, pairs in rels.items():
                        if not isinstance(pairs, list):
                            continue
                        for p in pairs:
                            if not (isinstance(p, list) and len(p) == 2):
                                continue
                            h, t = p[0], p[1]
                            if isinstance(h, str) and isinstance(t, str):
                                self.add_triple(h, str(rtype), t)

                kept += 1

        self._built = True

    def retrieve(
        self,
        entity_ids: Sequence[str],
        *,
        allowed_relations: Optional[Set[str]] = None,
        max_triples: int = 40,
        hop: int = 1,
    ) -> List[Triple]:
        """
        Retrieve a compact set of triples around entity_ids.

        - hop=1: outgoing edges from given entities
        - hop=2: also expand one hop from tails (bounded)
        """
        out: List[Triple] = []
        seen: Set[Tuple[str, str, str]] = set()

        frontier = list(entity_ids)
        for _ in range(max(1, hop)):
            next_frontier: List[str] = []
            for eid in frontier:
                for tr in self._adj.get(eid, []):
                    if allowed_relations and tr.rel not in allowed_relations:
                        continue
                    key = (tr.head, tr.rel, tr.tail)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(tr)
                    next_frontier.append(tr.tail)
                    if len(out) >= max_triples:
                        return out
            frontier = next_frontier

        return out

    def map_literal_to_entity(self, literal: str) -> Optional[str]:
        """
        Map a literal mention to a unique entity ID, if unambiguous.
        """
        k = self._norm_text(literal)
        ids = self._ent_text_index.get(k)
        if not ids or len(ids) != 1:
            return None
        return next(iter(ids))
