# ragtree/integrations/exporters/json.py
"""Pretty single-file JSON exporter."""

from __future__ import annotations

import json

from ragtree.core.schemas import RAGResult

__all__ = ["JsonExporter"]


class JsonExporter:
    def export(self, result: RAGResult, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result.model_dump(mode="json"), handle, ensure_ascii=False, indent=2)
