# ragtree/preprocessing/ingest/converters/docred_causal.py
import json
from pathlib import Path
from typing import Iterator, Dict, Any
from ragtree.preprocessing.ingest.convert_registry import BaseConverter, register

CAUSAL_REL_LABELS = {"causal", "cause-effect", "causes"}

@register("docred_causal")
class DocREDCausalConverter(BaseConverter):
    def iter_docs(self) -> Iterator[Dict[str, Any]]:
        files = list(Path(self.raw_dir).glob("**/*.json"))
        if not files:
            raise FileNotFoundError(f"No JSON found under {self.raw_dir}")
        for fp in files:
            data = json.loads(fp.read_text(encoding="utf-8"))
            # if file is an array of docs:
            docs = data if isinstance(data, list) else [data]
            for i, doc in enumerate(docs):
                text = "\n".join(" ".join(sent) for sent in doc.get("sents", []))
                rels = [r for r in doc.get("labels", []) if r.get("r") in CAUSAL_REL_LABELS]
                yield {
                    "doc_id": doc.get("doc_id", f"{fp.stem}::{i}"),
                    "text": text,
                    "entities": doc.get("vertexSet", []),
                    "relations": rels,
                    "meta": {"source": "DocRED", "file": str(fp)},
                }
