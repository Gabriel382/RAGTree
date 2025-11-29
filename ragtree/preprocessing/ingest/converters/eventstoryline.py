# ragtree/preprocessing/ingest/converters/eventstoryline.py
import json
from pathlib import Path
from typing import Iterator, Dict, Any
from ragtree.preprocessing.ingest.convert_registry import BaseConverter, register

@register("eventstoryline")
class EventStorylineConverter(BaseConverter):
    def iter_docs(self) -> Iterator[Dict[str, Any]]:
        files = list(Path(self.raw_dir).glob("**/*.json"))
        if not files:
            raise FileNotFoundError(f"No JSON found under {self.raw_dir}")
        for fp in files:
            obj = json.loads(fp.read_text(encoding="utf-8"))
            # adapt to exact ESL format
            yield {
                "doc_id": obj.get("id", fp.stem),
                "text": obj.get("text", ""),
                "entities": obj.get("entities", []),
                "relations": obj.get("relations", []),
                "meta": {"source": "EventStoryLine", "file": str(fp)},
            }
