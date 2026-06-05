# ragtree/kg/local_graphstore.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Edge:
    head: str
    rel: str
    tail: str
    meta: Dict[str, Any]


class LocalGraphStore:
    """
    Minimal local graph store (in-memory) that is:
      - deterministic
      - serializable
      - enough for retrieval / traversal

    Nodes: {node_id: {attrs...}}
    Edges: list[Edge]
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._edges: List[Edge] = []

        # adjacency for fast traversal
        self._out: Dict[str, List[int]] = {}  # node_id -> edge indices
        self._in: Dict[str, List[int]] = {}   # node_id -> edge indices

    def upsert_node(self, node_id: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        if node_id not in self._nodes:
            self._nodes[node_id] = {}
        if attrs:
            self._nodes[node_id].update(attrs)

    def add_edge(self, head: str, rel: str, tail: str, meta: Optional[Dict[str, Any]] = None) -> None:
        meta = meta or {}
        idx = len(self._edges)
        self._edges.append(Edge(head=head, rel=rel, tail=tail, meta=meta))

        self._out.setdefault(head, []).append(idx)
        self._in.setdefault(tail, []).append(idx)

        # Ensure nodes exist
        self.upsert_node(head, {})
        self.upsert_node(tail, {})

    def neighbors(self, node_id: str) -> List[Tuple[str, str, str]]:
        """
        Return outgoing (head, rel, tail) triples.
        """
        out = []
        for ei in self._out.get(node_id, []):
            e = self._edges[ei]
            out.append((e.head, e.rel, e.tail))
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": self._nodes,
            "edges": [
                {"head": e.head, "rel": e.rel, "tail": e.tail, "meta": e.meta}
                for e in self._edges
            ],
            "adj": {
                "out": self._out,
                "in": self._in,
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LocalGraphStore":
        gs = cls()
        gs._nodes = d.get("nodes", {}) or {}
        edges = d.get("edges", []) or []
        for e in edges:
            gs.add_edge(
                str(e["head"]),
                str(e["rel"]),
                str(e["tail"]),
                dict(e.get("meta", {}) or {}),
            )
        return gs
