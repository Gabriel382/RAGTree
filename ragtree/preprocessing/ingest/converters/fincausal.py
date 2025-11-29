# ragtree/preprocessing/ingest/converters/fincausal.py
import csv
from pathlib import Path
from typing import Iterator, Dict, Any
from ragtree.preprocessing.ingest.convert_registry import BaseConverter, register

@register("fincausal")
class FinCausalConverter(BaseConverter):
    def iter_docs(self) -> Iterator[Dict[str, Any]]:
        files = list(Path(self.raw_dir).glob("**/*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV found under {self.raw_dir}")
        for fp in files:
            with fp.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    cause = (row.get("Cause") or row.get("cause") or "").strip()
                    effect = (row.get("Effect") or row.get("effect") or "").strip()
                    text = f"{cause} -> {effect}".strip()
                    yield {
                        "doc_id": f"fincausal::{fp.stem}::{i}",
                        "text": text,
                        "entities": [
                            {"id": "e1", "text": cause, "start": None, "end": None, "label": "Cause"},
                            {"id": "e2", "text": effect, "start": None, "end": None, "label": "Effect"},
                        ],
                        "relations": [{"id": "r1", "type": "causal", "head": "e1", "tail": "e2", "evidence": text}],
                        "meta": {"source": "FinCausal", "file": str(fp)},
                    }
