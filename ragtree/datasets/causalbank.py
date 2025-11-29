# ragtree/datasets/causalbank.py
from pathlib import Path
import csv
from typing import Dict, Any, Iterable
from .base import BaseDataset

class CausalBankDataset(BaseDataset):
    name = "causalbank"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> Iterable[Dict[str, Any]]:
        if not self.path.exists():
            raise FileNotFoundError(
                f"CausalBank file not found at {self.path}. "
                "Download it and place it there (configs.paths.data_raw)."
            )
        with self.path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for i, row in enumerate(reader):
                # adjust to your real columns — many releases have 'cause' and 'effect'
                yield {
                    "doc_id": f"causalbank_{i}",
                    "text": f"{row.get('cause','')} -> {row.get('effect','')}",
                    "entities": [
                        {"text": row.get("cause",""), "role": "cause"},
                        {"text": row.get("effect",""), "role": "effect"},
                    ],
                    "relations": [
                        {
                            "type": "causal",
                            "head": row.get("cause",""),
                            "tail": row.get("effect",""),
                        }
                    ],
                }
