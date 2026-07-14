# ragtree/integrations/graphstores/neo4j.py
"""Neo4j GraphStore adapter."""

from __future__ import annotations

import os
from typing import Any

from ragtree.core.errors import require_extra

__all__ = ["Neo4jGraphStore"]

_PRIMITIVES = (str, int, float, bool)


def _props(mapping: dict[str, Any], exclude: tuple[str, ...]) -> dict[str, Any]:
    return {
        k: v
        for k, v in mapping.items()
        if k not in exclude and (v is None or isinstance(v, _PRIMITIVES))
    }


class Neo4jGraphStore:
    """GraphStore over Neo4j (extra: ``neo4j``).

    Nodes are ``(:Entity {id, ...})`` merged on ``id``; edges are
    ``[:REL {type, ...}]`` merged on (source, target, type) — a fixed
    relationship label with a ``type`` property avoids requiring APOC for
    dynamic relationship types. ``query`` runs raw Cypher.
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ) -> None:
        require_extra("neo4j", "neo4j")
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(
            uri or os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                user or os.getenv("NEO4J_USER", "neo4j"),
                password or os.getenv("NEO4J_PASSWORD", "password"),
            ),
        )
        self._database = database or os.getenv("NEO4J_DATABASE") or None

    def close(self) -> None:
        self._driver.close()

    def _run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        with self._driver.session(database=self._database) as session:
            return [dict(record) for record in session.run(cypher, **params)]

    def upsert_nodes(self, nodes: list[dict[str, Any]]) -> None:
        payload = [
            {"id": str(node["id"]), "props": _props(node, exclude=("id",))} for node in nodes
        ]
        self._run(
            "UNWIND $nodes AS n MERGE (x:Entity {id: n.id}) SET x += n.props",
            nodes=payload,
        )

    def upsert_edges(self, edges: list[dict[str, Any]]) -> None:
        payload = [
            {
                "source": str(edge.get("source") or edge.get("head") or ""),
                "target": str(edge.get("target") or edge.get("tail") or ""),
                "type": str(edge.get("type") or edge.get("rel") or "RELATED"),
                "props": _props(edge, exclude=("source", "target", "head", "tail", "type", "rel")),
            }
            for edge in edges
        ]
        self._run(
            "UNWIND $edges AS e "
            "MERGE (a:Entity {id: e.source}) "
            "MERGE (b:Entity {id: e.target}) "
            "MERGE (a)-[r:REL {type: e.type}]->(b) "
            "SET r += e.props",
            edges=payload,
        )

    def query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return self._run(query, **(params or {}))
