from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Literal, Protocol
from datetime import datetime, timezone

TargetKind = Literal["class", "datatype", "property"]

@dataclass(frozen=True)
class OntologyLinkCandidate:
    target_kind: TargetKind
    concept_uri: str
    label: str
    score: float
    source: str = "unknown"
    evidence: Optional[Dict[str, Any]] = None

@dataclass(frozen=True)
class OntologyLinksMeta:
    schema_version: str
    method: str
    ontology_key: str
    params: Dict[str, Any]
    created_at: str

def now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def make_links_v1(
    *,
    method: str,
    ontology_key: str,
    params: Dict[str, Any],
    by_entity: Dict[str, List[OntologyLinkCandidate]],
    selected_top1: bool = False,
) -> Dict[str, Any]:
    """Create schema v1 payload for doc['ontology_links'].

    Returns:
      {
        "_meta": {...},
        "by_entity": {
          "<ENTITY_ID>": {"candidates": [...], "selected": {...}|null},
          ...
        }
      }
    """
    meta = OntologyLinksMeta(
        schema_version="1.0",
        method=method,
        ontology_key=ontology_key,
        params=params,
        created_at=now_iso_utc(),
    )

    out: Dict[str, Any] = {"_meta": asdict(meta), "by_entity": {}}
    for ent_id, cands in by_entity.items():
        cands_sorted = sorted(cands, key=lambda c: c.score, reverse=True)
        selected = asdict(cands_sorted[0]) if (selected_top1 and cands_sorted) else None
        out["by_entity"][ent_id] = {
            "candidates": [asdict(c) for c in cands_sorted],
            "selected": selected,
        }
    return out

def is_links_v1(obj: Any) -> bool:
    return isinstance(obj, dict) and "_meta" in obj and "by_entity" in obj

def upgrade_legacy_links(legacy: Dict[str, List[Dict[str, Any]]], *, method: str, ontology_key: str) -> Dict[str, Any]:
    """Upgrade legacy {ent_id: [{concept_uri,label,score}, ...]} to schema v1."""
    by_entity: Dict[str, List[OntologyLinkCandidate]] = {}
    for ent_id, arr in (legacy or {}).items():
        cands: List[OntologyLinkCandidate] = []
        for item in arr or []:
            cands.append(
                OntologyLinkCandidate(
                    target_kind="class",
                    concept_uri=str(item.get("concept_uri", "")),
                    label=str(item.get("label", "")),
                    score=float(item.get("score", 0.0)),
                    source="legacy",
                    evidence=None,
                )
            )
        if cands:
            by_entity[ent_id] = cands
    return make_links_v1(method=method, ontology_key=ontology_key, params={"upgraded_from": "legacy"}, by_entity=by_entity)

class BaseOntologyLinker(Protocol):
    """Abstract interface for ontology linking implementations."""
    def link_document(self, doc: Dict[str, Any], *, top_k: int = 3, min_score: float = 0.3) -> Dict[str, Any]:
        ...
