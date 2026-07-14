# ragtree/processing/kg_rag/kg_loader.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ragtree.kg.local_graphstore import LocalGraphStore


def load_local_kg(kg_path: Path) -> Dict[str, Any]:
    """
    Load the JSON produced by scripts/build_kg_from_preprocessed.py.
    Returns the full payload, including payload["graph"].
    """
    payload = json.loads(kg_path.read_text(encoding="utf-8"))
    return payload


def load_local_graphstore(kg_path: Path) -> LocalGraphStore:
    payload = load_local_kg(kg_path)
    graph = payload.get("graph", {}) or {}
    return LocalGraphStore.from_dict(graph)
