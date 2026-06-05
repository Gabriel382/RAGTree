# scripts/build_kg_from_preprocessed.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ragtree.core.config import load_config
from ragtree.kg.local_graphstore import LocalGraphStore


def _parse_doc_types(arg: str) -> Sequence[str] | str:
    """
    "--doc-types train,dev" -> ["train","dev"]
    "--doc-types all"       -> "all"
    """
    if not arg or arg.strip() == "all":
        return "all"
    items = [x.strip() for x in arg.split(",")]
    items = [x for x in items if x]
    return items or "all"


def _iter_docs(
    path: Path,
    *,
    doc_types: Sequence[str] | str,
    skip: int,
    limit: Optional[int],
) -> Iterable[Dict[str, Any]]:
    kept = 0
    skipped = 0

    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)

            # 1) doc-type filter
            if doc_types != "all":
                if doc.get("type") not in set(doc_types):
                    continue

            # 2) skip AFTER doc-type filtering
            if skipped < skip:
                skipped += 1
                continue

            # 3) limit AFTER doc-type filtering
            if limit is not None and kept >= limit:
                break

            kept += 1
            yield doc


def _extract_triples_from_doc(doc: Dict[str, Any]) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    """
    Convert doc relations -> KG edges.

    Expected doc format:
      doc["relations"] = { "REL": [[head_id, tail_id], ...], ... }
    """
    triples: List[Tuple[str, str, str, Dict[str, Any]]] = []
    rels = doc.get("relations")

    if not isinstance(rels, dict) or not rels:
        return triples

    doc_id = doc.get("document_id", "")
    for rel, pairs in rels.items():
        if not isinstance(pairs, list):
            continue
        for p in pairs:
            if not isinstance(p, list) or len(p) != 2:
                continue
            h, t = p[0], p[1]
            if not isinstance(h, str) or not isinstance(t, str):
                continue
            meta = {
                "source": "gold",
                "document_id": doc_id,
            }
            triples.append((h, rel, t, meta))
    return triples


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a local KG from preprocessed JSONL (gold relations).")
    ap.add_argument("--config", type=Path, default=None, help="Path to default.yaml (optional).")
    ap.add_argument("--dataset-key", required=True, help="Key under cfg['datasets']['preprocessed'].")
    ap.add_argument("--doc-types", type=str, default="train", help="all or comma-separated (train,dev,test). Default=train.")
    ap.add_argument("--skip", type=int, default=0, help="Skip first N docs AFTER doc-type filter.")
    ap.add_argument("--limit", type=int, default=None, help="Take at most K docs AFTER doc-type filter. Default=None (all).")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional override output path. Default: <paths.kg>/<dataset>_kg.json",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Resolve input
    ds_pre = cfg["datasets"]["preprocessed"]
    if args.dataset_key not in ds_pre:
        available = ", ".join(sorted(ds_pre.keys()))
        raise KeyError(f"Unknown dataset key '{args.dataset_key}'. Available: {available}")
    input_path = Path(ds_pre[args.dataset_key])

    # Resolve output root
    paths = cfg.get("paths", {}) or {}
    kg_root = Path(paths.get("kg", "data/kg"))
    kg_root.mkdir(parents=True, exist_ok=True)

    doc_types = _parse_doc_types(args.doc_types)
    suffix = f"types={args.doc_types}_skip={args.skip}_limit={args.limit}"
    default_out = kg_root / f"{args.dataset_key}__{suffix}__kg.json"
    out_path = args.out or default_out

    # Build KG
    gs = LocalGraphStore()

    num_docs = 0
    num_triples = 0

    for doc in _iter_docs(input_path, doc_types=doc_types, skip=args.skip, limit=args.limit):
        num_docs += 1

        # Add entity nodes (optional but helpful)
        ents = doc.get("entities") or {}
        if isinstance(ents, dict):
            for ent_id, ent in ents.items():
                if not isinstance(ent_id, str):
                    continue
                gs.upsert_node(
                    ent_id,
                    {
                        "type": ent.get("type"),
                        "mentions": ent.get("mentions", []),
                    },
                )

        # Add gold relation edges
        triples = _extract_triples_from_doc(doc)
        for (h, r, t, meta) in triples:
            gs.add_edge(h, r, t, meta)
            num_triples += 1

    # Save
    payload = {
        "dataset_key": args.dataset_key,
        "source_path": str(input_path),
        "params": {
            "doc_types": args.doc_types,
            "skip": args.skip,
            "limit": args.limit,
        },
        "stats": {
            "num_docs": num_docs,
            "num_triples": num_triples,
            "num_nodes": len(gs._nodes),
            "num_edges": len(gs._edges),
        },
        "graph": gs.to_dict(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[kg] wrote:", out_path)
    print("[kg] stats:", payload["stats"])


if __name__ == "__main__":
    main()
