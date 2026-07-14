# ragtree/evaluation/relations/io.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def extract_doc_id(doc: Dict[str, Any]) -> Optional[str]:
    """
    Try to extract a document identifier in a robust way.
    Prefer 'document_id', fallback to 'id'.
    """
    doc_id = doc.get("document_id")
    if isinstance(doc_id, str) and doc_id:
        return doc_id
    doc_id = doc.get("id")
    if isinstance(doc_id, str) and doc_id:
        return doc_id
    return None


def load_gold_relations_by_id(gold_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load all gold relations from a preprocessed JSONL file, indexed by document id.

    We assume each line is a JSON object with:
      - a doc id ('document_id' or 'id')
      - a 'relations' dict: relation_type -> [[head_id, tail_id], ...]

    Returns:
      dict: doc_id -> relations_dict
    """
    mapping: Dict[str, Dict[str, Any]] = {}

    with gold_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            doc_id = extract_doc_id(doc)
            if not doc_id:
                continue

            rels = doc.get("relations")
            if isinstance(rels, dict):
                mapping[doc_id] = rels

    return mapping
