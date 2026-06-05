# ragtree/processing/kg_rag/triple_kg_retriever.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ragtree.kg.local_graphstore import LocalGraphStore


def _tok(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _doc_text(doc: Dict[str, Any]) -> str:
    # Try common shapes used across your datasets
    sents = doc.get("sentences")
    if isinstance(sents, list) and all(isinstance(s, str) for s in sents):
        return " ".join(sents)

    # DocRED-style sometimes: sents = [["w1","w2"], ...]
    sents2 = doc.get("sents")
    if isinstance(sents2, list) and all(isinstance(x, list) for x in sents2):
        flat = []
        for s in sents2:
            flat.append(" ".join(str(w) for w in s))
        return " ".join(flat)

    t = doc.get("text")
    if isinstance(t, str):
        return t

    return ""


def _entity_alias_map(doc: Dict[str, Any], *, max_mentions: int = 2) -> Dict[str, str]:
    """
    Build a readable alias per entity_id from mentions.
    Expected mention shapes (your KG builder stores mentions verbatim):
      - {"trigger_word": "..."} or {"text": "..."} or {"name": "..."} ...
    """
    ents = doc.get("entities") or {}
    out: Dict[str, str] = {}

    if not isinstance(ents, dict):
        return out

    for ent_id, ent in ents.items():
        if not isinstance(ent_id, str) or not isinstance(ent, dict):
            continue
        mentions = ent.get("mentions", [])
        if not isinstance(mentions, list):
            mentions = [mentions]

        parts: List[str] = []
        for m in mentions:
            if not isinstance(m, dict):
                continue
            trig = m.get("trigger_word") or m.get("text") or m.get("name") or ""
            trig = str(trig).strip()
            if trig:
                parts.append(trig)
            if len(parts) >= max_mentions:
                break

        out[ent_id] = " / ".join(parts) if parts else ent.get("type", "") or ""

    return out


@dataclass
class TripleKGRetrieverParams:
    max_hops: int = 1
    max_triples: int = 80
    include_in_edges: bool = True
    scoring: str = "token_overlap"  # future: "bm25", "embedding", ...


class SimpleKGFragment:
    """
    Minimal fragment interface (mirrors what strategies need: to_dict(), to_text()).
    """
    def __init__(self, triples: List[Dict[str, Any]]) -> None:
        self.triples = triples

    def to_dict(self) -> Dict[str, Any]:
        return {"triples": self.triples}

    def to_text(self, *, max_lines: Optional[int] = None) -> str:
        lines: List[str] = []
        for i, tr in enumerate(self.triples):
            if max_lines is not None and i >= max_lines:
                break
            h = tr.get("h")
            r = tr.get("r")
            t = tr.get("t")
            score = tr.get("score")
            lines.append(f"- {h} --{r}--> {t} (score={score:.3f})")
        return "\n".join(lines) if lines else "(no triples retrieved)"


class TripleKGRetriever:
    """
    Simple triple retrieval on LocalGraphStore:
      1) seed nodes = doc entity IDs
      2) BFS up to max_hops collecting incident edges
      3) score triples vs doc text (token overlap)
      4) return top-K as fragment
    """
    def __init__(self, gs: LocalGraphStore, *, params: TripleKGRetrieverParams) -> None:
        self.gs = gs
        self.params = params

    def _seed_nodes(self, doc: Dict[str, Any]) -> List[str]:
        ents = doc.get("entities") or {}
        if isinstance(ents, dict):
            return [eid for eid in ents.keys() if isinstance(eid, str)]
        return []

    def _iter_incident_edge_ids(self, node_id: str) -> Iterable[int]:
        # LocalGraphStore stores adjacency in _out (and _in) as node_id -> [edge_id,...]
        out_adj = getattr(self.gs, "_out", {}) or {}
        for eid in out_adj.get(node_id, []) or []:
            yield eid

        if self.params.include_in_edges:
            in_adj = getattr(self.gs, "_in", {}) or {}
            for eid in in_adj.get(node_id, []) or []:
                yield eid

    def _edge_tuple(self, edge_id: int) -> Optional[Tuple[str, str, str, Dict[str, Any]]]:
        edges = getattr(self.gs, "_edges", None)
        if not isinstance(edges, list):
            return None
        if edge_id < 0 or edge_id >= len(edges):
            return None
        e = edges[edge_id]
        if not isinstance(e, tuple) or len(e) != 4:
            return None
        h, r, t, meta = e
        if not isinstance(h, str) or not isinstance(r, str) or not isinstance(t, str):
            return None
        if not isinstance(meta, dict):
            meta = {}
        return h, r, t, meta

    def retrieve(self, doc: Dict[str, Any]) -> SimpleKGFragment:
        seeds = self._seed_nodes(doc)
        if not seeds:
            return SimpleKGFragment([])

        qtext = _doc_text(doc)
        qtok = _tok(qtext)
        alias = _entity_alias_map(doc)

        # BFS to collect candidate edges
        seen_nodes: Set[str] = set(seeds)
        frontier: Set[str] = set(seeds)
        seen_edge_ids: Set[int] = set()

        for _hop in range(max(1, int(self.params.max_hops))):
            next_frontier: Set[str] = set()
            for nid in list(frontier):
                for eid in self._iter_incident_edge_ids(nid):
                    if not isinstance(eid, int) or eid in seen_edge_ids:
                        continue
                    seen_edge_ids.add(eid)
                    tup = self._edge_tuple(eid)
                    if not tup:
                        continue
                    h, r, t, _meta = tup
                    if h not in seen_nodes:
                        next_frontier.add(h)
                    if t not in seen_nodes:
                        next_frontier.add(t)
            seen_nodes |= next_frontier
            frontier = next_frontier
            if not frontier:
                break

        # Score triples
        scored: List[Dict[str, Any]] = []
        for eid in seen_edge_ids:
            tup = self._edge_tuple(eid)
            if not tup:
                continue
            h, r, t, meta = tup

            # textualize with aliases (helps overlap a lot)
            ha = alias.get(h, "")
            ta = alias.get(t, "")
            triple_text = f"{h} {ha} {r} {t} {ta}"
            ttok = _tok(triple_text)

            overlap = float(len(ttok & qtok))
            bonus = 0.0
            # strong bias for within-document entity pairs
            if h in alias:
                bonus += 1.0
            if t in alias:
                bonus += 1.0
            # slight bias if relation token appears in doc
            if r.lower() in qtok:
                bonus += 0.5

            score = overlap + bonus

            scored.append(
                {
                    "h": h,
                    "r": r,
                    "t": t,
                    "score": float(score),
                    "meta": meta,
                }
            )

        scored.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        top = scored[: int(self.params.max_triples)]
        return SimpleKGFragment(top)
