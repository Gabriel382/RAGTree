# ragtree/integrations/exporters/jsonl.py
"""JSONL exporter: appends one line per result (multi-document runs)."""

from __future__ import annotations

import json

from ragtree.core.schemas import RAGResult

__all__ = ["JsonlExporter"]


class JsonlExporter:
    def export(self, result: RAGResult, output_path: str) -> None:
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.model_dump(mode="json"), ensure_ascii=False) + "\n")
