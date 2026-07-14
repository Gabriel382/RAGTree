# ragtree/integrations/exporters/csv.py
"""CSV exporter: one flat row per result (appends, writes header once)."""

from __future__ import annotations

import csv
import json
import os

from ragtree.core.schemas import RAGResult

__all__ = ["CsvExporter"]


class CsvExporter:
    def export(self, result: RAGResult, output_path: str) -> None:
        row = {
            "task_type": result.task_type,
            "output": result.output
            if isinstance(result.output, str)
            else json.dumps(result.output, ensure_ascii=False),
            "n_evidence": len(result.evidence),
            **{f"metric_{k}": v for k, v in sorted(result.metrics.items())},
            "document_id": result.metadata.get("document_id", ""),
        }
        write_header = not os.path.exists(output_path) or os.path.getsize(output_path) == 0
        with open(output_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
