# ragtree/preprocessing/ingest/converters/causalbank.py
from pathlib import Path
import csv
from typing import Dict, Any, Iterator
from ragtree.preprocessing.ingest.convert_registry import BaseConverter, register

@register("causalbank")
class CausalBankConverter(BaseConverter):
    def iter_docs(self) -> Iterator[Dict[str, Any]]:
        # Expect a TSV somewhere inside data/raw/CausalBank
        tsvs = list(Path(self.raw_dir).glob("**/*.tsv"))
        if not tsvs:
            raise FileNotFoundError(f"No TSV found under {self.raw_dir}")

        for tsv in tsvs:
            with tsv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for i, row in enumerate(reader):
                    cause = (row.get("cause") or row.get("Cause") or "").strip()
                    effect = (row.get("effect") or row.get("Effect") or "").strip()
                    text = f"{cause} -> {effect}".strip()
                    yield {
                        "doc_id": f"causalbank::{tsv.stem}::{i}",
                        "text": text,
                        "entities": [
                            {"id": "e1", "text": cause, "start": None, "end": None, "label": "Cause"},
                            {"id": "e2", "text": effect, "start": None, "end": None, "label": "Effect"},
                        ],
                        "relations": [{"id": "r1", "type": "causal", "head": "e1", "tail": "e2", "evidence": text}],
                        "meta": {"source": "CausalBank", "file": str(tsv)},
                    }
