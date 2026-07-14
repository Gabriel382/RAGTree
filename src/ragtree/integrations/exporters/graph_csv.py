# ragtree/integrations/exporters/graph_csv.py
"""Graph-ready CSV exporter for relation outputs.

Writes the EDGES table at ``output_path`` (source,type,target,document_id)
and a companion nodes file at ``<stem>_nodes.csv``. Both import directly
into Neo4j (LOAD CSV), Gephi or pandas.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ragtree.core.schemas import RAGResult

__all__ = ["GraphCsvExporter"]


class GraphCsvExporter:
    def export(self, result: RAGResult, output_path: str) -> None:
        document_id = str(result.metadata.get("document_id", ""))

        edges: list[dict[str, str]] = []
        if isinstance(result.artifacts.get("edges"), list):
            for edge in result.artifacts["edges"]:
                edges.append(
                    {
                        "source": str(edge.get("source", "")),
                        "type": str(edge.get("type", "RELATED")),
                        "target": str(edge.get("target", "")),
                        "document_id": str(edge.get("document_id", document_id)),
                    }
                )
        elif isinstance(result.output, dict):
            for rel_type, pairs in result.output.items():
                if not isinstance(pairs, list):
                    continue
                for pair in pairs:
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        edges.append(
                            {
                                "source": str(pair[0]),
                                "type": str(rel_type),
                                "target": str(pair[1]),
                                "document_id": document_id,
                            }
                        )

        edge_path = Path(output_path)
        with edge_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["source", "type", "target", "document_id"]
            )
            writer.writeheader()
            writer.writerows(edges)

        node_ids = sorted({e["source"] for e in edges} | {e["target"] for e in edges})
        nodes_path = edge_path.with_name(f"{edge_path.stem}_nodes.csv")
        with nodes_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "document_id"])
            writer.writeheader()
            writer.writerows({"id": nid, "document_id": document_id} for nid in node_ids)
