# ragtree/integrations/graphstores/local.py
"""GraphStore adapter over the existing local research graph store.

Wraps ``ragtree.kg.local_graphstore.LocalGraphStore`` (kept unchanged for
the benchmark scripts) behind the core GraphStore protocol.
"""

from __future__ import annotations

from typing import Any

from ragtree.kg.local_graphstore import LocalGraphStore as _LegacyLocalGraphStore

__all__ = ["LocalGraphStore"]


class LocalGraphStore:
    """In-process GraphStore; supports the queries ``nodes``, ``edges`` and
    ``neighbors:<node_id>``."""

    def __init__(self, store: _LegacyLocalGraphStore | None = None) -> None:
        self._store = store or _LegacyLocalGraphStore()
        self._edge_keys: set[tuple[str, str, str]] = {
            (e.head, e.rel, e.tail) for e in self._store._edges
        }

    def upsert_nodes(self, nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            node_id = str(node["id"])
            attrs = {k: v for k, v in node.items() if k != "id"}
            self._store.upsert_node(node_id, attrs)

    def upsert_edges(self, edges: list[dict[str, Any]]) -> None:
        for edge in edges:
            head = str(edge.get("source") or edge.get("head") or "")
            tail = str(edge.get("target") or edge.get("tail") or "")
            rel = str(edge.get("type") or edge.get("rel") or "RELATED")
            key = (head, rel, tail)
            if key in self._edge_keys:
                continue
            meta = {
                k: v
                for k, v in edge.items()
                if k not in ("source", "target", "head", "tail", "type", "rel")
            }
            self._store.add_edge(head, rel, tail, meta=meta)
            self._edge_keys.add(key)

    def query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        data = self._store.to_dict()
        if query == "nodes":
            return [{"id": node_id, **attrs} for node_id, attrs in data["nodes"].items()]
        if query == "edges":
            return [
                {"source": e["head"], "type": e["rel"], "target": e["tail"], **e["meta"]}
                for e in data["edges"]
            ]
        if query.startswith("neighbors:"):
            node_id = query.split(":", 1)[1]
            return [
                {"source": head, "type": rel, "target": tail}
                for head, rel, tail in self._store.neighbors(node_id)
            ]
        raise ValueError(
            f"Unsupported query {query!r}: LocalGraphStore answers 'nodes', "
            "'edges' and 'neighbors:<node_id>'."
        )
