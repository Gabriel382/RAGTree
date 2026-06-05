#!/usr/bin/env python3
# scripts/build_community_kgrag_index.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm

# Louvain community detection (python-louvain)
try:
    import community as community_louvain  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: python-louvain (pip install python-louvain)") from e

# FAISS index
try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: faiss (pip install faiss-cpu or faiss-gpu)") from e

# Sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError("Missing dependency: sentence-transformers (pip install sentence-transformers)") from e


# ============================================================
# Metadata written to meta.json
# ============================================================

@dataclass
class BuilderMeta:
    # Dataset / inputs
    dataset_key: str
    kg_path: str
    output_dir: str

    # Models
    node_embed_model: str
    query_embed_model: str
    sentence_embed_model: str

    # Index settings
    community_index_metric: str
    sentence_index_metric: str

    # Build options
    build_sentence_index: bool
    device: str
    sentence_device: str
    batch_size: int

    # Stats
    num_nodes: int
    num_edges: int
    num_sentences: int
    num_communities: int
    dim_node: int
    dim_sentence: int


# ============================================================
# IO helpers
# ============================================================

def _read_json_or_jsonl(path: Path) -> Any:
    """
    Read JSON (dict/list) or JSONL (one JSON per line).
    """
    txt = path.read_text(encoding="utf-8")
    txt_strip = txt.lstrip()
    if txt_strip.startswith("{") or txt_strip.startswith("["):
        return json.loads(txt)

    # JSONL fallback
    items: List[Any] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _ensure_dir(p: Path) -> None:
    """
    Ensure directory exists.
    """
    p.mkdir(parents=True, exist_ok=True)


def _save_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """
    Save iterable of dicts to JSONL.
    """
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ============================================================
# KG parsing helpers
# ============================================================

def _normalize_edge(e: Dict[str, Any], fallback_idx: int) -> Dict[str, Any]:
    """
    Normalize dict-edges into the unified schema expected downstream.
    """
    head = e.get("head") or e.get("h") or e.get("source")
    tail = e.get("tail") or e.get("t") or e.get("target")
    rel = e.get("rel") or e.get("relation") or e.get("r") or "related_to"

    if head is None or tail is None:
        raise ValueError(f"Edge missing head/tail: {e}")

    doc_id = e.get("document_id") or e.get("doc_id") or e.get("docId")
    sent_id = e.get("sentence_id") or e.get("sent_id") or e.get("sid")
    sent_txt = e.get("sentence_text") or e.get("sentence") or e.get("text")

    # Deterministic sentence_id if missing
    if sent_id is None:
        if doc_id is None:
            doc_id = "unknown_doc"
        sent_id = f"{doc_id}::edge_{fallback_idx}"

    return {
        "head": str(head),
        "tail": str(tail),
        "rel": str(rel),
        "document_id": str(doc_id) if doc_id is not None else None,
        "sentence_id": str(sent_id),
        "sentence_text": str(sent_txt) if sent_txt is not None else None,
        # Optional labels if present
        "head_label": e.get("head_label") or e.get("h_label") or e.get("source_label"),
        "tail_label": e.get("tail_label") or e.get("t_label") or e.get("target_label"),
    }


def _collect_from_kg(kg_obj: Any) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, Dict[str, Any]]]:
    """
    Supports YOUR KG schema produced by build_kg_from_preprocessed.py:

    Top-level example:
      {
        "dataset_key": "...",
        "graph": {
          "nodes": { node_id: {"type":..., "mentions":[{"trigger_word":...}, ...]}, ... },
          "edges":  [...],   # OR
          "triples":[...]
        }
      }

    Notes:
      - nodes is a DICT (not list)
      - edges/triples can be:
          * dict edges
          * tuple/list edges: (head, rel, tail, meta_dict)

    Returns:
      edges: normalized dict edges
      node_labels: node_id -> readable label (from mentions trigger_word if available else node_id)
      sent_map: sentence_id -> {"sentence_id","document_id","text"}
    """

    # Unwrap graph key if present
    if isinstance(kg_obj, dict) and "graph" in kg_obj and isinstance(kg_obj["graph"], dict):
        kg_obj = kg_obj["graph"]

    edges_raw = None
    nodes_raw = None

    # Determine where edges and nodes are
    if isinstance(kg_obj, dict):
        nodes_raw = kg_obj.get("nodes")
        edges_raw = kg_obj.get("edges") or kg_obj.get("triples") or kg_obj.get("relations")

        # Fallback variant if ever stored differently
        if edges_raw is None and "_edges" in kg_obj:
            edges_raw = kg_obj.get("_edges")

        if edges_raw is None:
            raise ValueError(
                "KG JSON must contain graph['edges'] or graph['triples'].\n"
                f"Keys in graph: {sorted(list(kg_obj.keys()))}"
            )
    elif isinstance(kg_obj, list):
        # If user ever passes a list of edges directly
        edges_raw = kg_obj
        nodes_raw = None
    else:
        raise ValueError(f"Unsupported KG format: {type(kg_obj)}")

    # Build node labels from nodes dict:
    # node_labels[node_id] = first mention trigger_word if possible, else node_id
    node_labels: Dict[str, str] = {}
    if isinstance(nodes_raw, dict):
        for nid, info in nodes_raw.items():
            nid = str(nid)
            label = nid
            if isinstance(info, dict):
                mentions = info.get("mentions") or []
                if isinstance(mentions, list):
                    for m in mentions:
                        if isinstance(m, dict):
                            tw = m.get("trigger_word") or m.get("text") or m.get("name")
                            if tw:
                                label = str(tw)
                                break
            node_labels[nid] = label

    edges: List[Dict[str, Any]] = []
    sent_map: Dict[str, Dict[str, Any]] = {}

    def _register_sentence(sid: str, doc_id: Any, sent_txt: Any) -> None:
        """
        Store sentence metadata (if available).
        """
        if sid not in sent_map:
            sent_map[sid] = {
                "sentence_id": sid,
                "document_id": str(doc_id) if doc_id is not None else None,
                "text": str(sent_txt) if sent_txt is not None else "",
            }
        else:
            # Fill missing text if later edge provides it
            if not sent_map[sid].get("text") and sent_txt:
                sent_map[sid]["text"] = str(sent_txt)

    # Normalize edges in either dict or tuple form
    for i, e in enumerate(edges_raw):
        # Dict edges
        if isinstance(e, dict):
            ne = _normalize_edge(e, i)
            edges.append(ne)

            # Ensure labels exist for head/tail
            node_labels.setdefault(ne["head"], ne.get("head_label") or ne["head"])
            node_labels.setdefault(ne["tail"], ne.get("tail_label") or ne["tail"])

            _register_sentence(ne["sentence_id"], ne.get("document_id"), ne.get("sentence_text"))
            continue

        # Tuple/list edges: (head, rel, tail, meta_dict)
        if isinstance(e, (list, tuple)) and len(e) == 4:
            h, r, t, meta = e
            if meta is None or not isinstance(meta, dict):
                meta = {}

            doc_id = meta.get("document_id") or meta.get("doc_id")
            sid = meta.get("sentence_id") or meta.get("sent_id") or f"unknown_doc::edge_{i}"
            sent_txt = meta.get("sentence_text") or meta.get("sentence") or meta.get("text")

            ne = {
                "head": str(h),
                "tail": str(t),
                "rel": str(r),
                "document_id": str(doc_id) if doc_id is not None else None,
                "sentence_id": str(sid),
                "sentence_text": str(sent_txt) if sent_txt is not None else None,
                "head_label": meta.get("head_label"),
                "tail_label": meta.get("tail_label"),
            }
            edges.append(ne)

            node_labels.setdefault(ne["head"], ne.get("head_label") or ne["head"])
            node_labels.setdefault(ne["tail"], ne.get("tail_label") or ne["tail"])

            _register_sentence(ne["sentence_id"], ne.get("document_id"), ne.get("sentence_text"))
            continue

        # Ignore unknown shapes safely
        continue

    return edges, node_labels, sent_map


# ============================================================
# Graph + embeddings + FAISS
# ============================================================

def _build_graph(edges: List[Dict[str, Any]]) -> nx.Graph:
    """
    Undirected graph for Louvain clustering.
    Edge weights = frequency of the pair in extracted triples.
    """
    G = nx.Graph()
    for e in edges:
        h, t = e["head"], e["tail"]
        if h == t:
            continue
        if G.has_edge(h, t):
            G[h][t]["weight"] += 1.0
        else:
            G.add_edge(h, t, weight=1.0)
    return G


def _embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    """
    Encode a list of texts with normalization and return float32 matrix.
    """
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return emb.astype("float32")


def _faiss_index(metric: str, dim: int) -> faiss.Index:
    """
    Build a Flat FAISS index.
    """
    metric = metric.lower()
    if metric == "ip":
        return faiss.IndexFlatIP(dim)
    if metric == "l2":
        return faiss.IndexFlatL2(dim)
    raise ValueError(f"Unsupported metric: {metric}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser("Build CommunityKG-RAG artifacts (communities + FAISS indexes).")

    # Input
    ap.add_argument("--dataset-key", required=True)
    ap.add_argument("--kg-path", type=Path, required=True, help="KG JSON produced by build_kg_from_preprocessed.py")
    ap.add_argument("--output-dir", type=Path, required=True, help="Root output dir (e.g., data/kg_community).")

    # Models
    ap.add_argument("--node-embed-model", default="BAAI/bge-m3")
    ap.add_argument("--query-embed-model", default="BAAI/bge-m3")
    ap.add_argument("--sentence-embed-model", default="BAAI/bge-m3")

    # Index metrics
    ap.add_argument("--community-index-metric", default="ip", choices=["ip", "l2"])
    ap.add_argument("--sentence-index-metric", default="ip", choices=["ip", "l2"])

    # IMPORTANT: defaults to CPU to avoid CUDA OOM when GPU is busy
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for node/community embeddings.")
    ap.add_argument("--sentence-device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for sentence embeddings.")

    # Optional sentence index
    ap.add_argument("--build-sentence-index", action="store_true", help="If set, pre-embed sentences + FAISS.")

    # Performance
    ap.add_argument("--batch-size", type=int, default=32)

    args = ap.parse_args()

    out_root = args.output_dir / args.dataset_key
    _ensure_dir(out_root)

    # 1) Load KG
    kg_obj = _read_json_or_jsonl(args.kg_path)

    # 2) Parse into edges + labels + sentences
    edges, node_labels, sent_map = _collect_from_kg(kg_obj)

    # 3) Persist normalized material (debuggable + reusable)
    _save_jsonl(out_root / "edges.jsonl", edges)
    _save_jsonl(out_root / "nodes.jsonl", ({"node_id": k, "label": v} for k, v in sorted(node_labels.items())))
    _save_jsonl(out_root / "sentences.jsonl", (sent_map[k] for k in sorted(sent_map.keys())))

    # 4) Build graph + Louvain
    print("[communitykgrag] Building graph + Louvain communities...")
    G = _build_graph(edges)
    partition = community_louvain.best_partition(G, weight="weight")  # node_id -> community_id (int)

    # Build maps
    comm_to_nodes: Dict[int, List[str]] = {}
    for nid, cid in partition.items():
        comm_to_nodes.setdefault(int(cid), []).append(str(nid))
    for cid in comm_to_nodes:
        comm_to_nodes[cid].sort()

    node_to_comm = {str(nid): int(cid) for nid, cid in partition.items()}

    _save_jsonl(
        out_root / "communities" / "node_to_comm.jsonl",
        ({"node_id": nid, "community_id": cid} for nid, cid in sorted(node_to_comm.items())),
    )
    _save_jsonl(
        out_root / "communities" / "comm_to_nodes.jsonl",
        ({"community_id": cid, "node_ids": nids} for cid, nids in sorted(comm_to_nodes.items())),
    )

    # 5) Map sentences to communities (simple, robust):
    # assign each sentence_id to the community of the head (or tail) node
    comm_to_sent: Dict[int, List[str]] = {}
    for e in edges:
        sid = e["sentence_id"]
        h = e["head"]
        cid = node_to_comm.get(h)
        if cid is None:
            cid = node_to_comm.get(e["tail"])
        if cid is None:
            continue
        comm_to_sent.setdefault(int(cid), []).append(sid)

    # de-dup sentence ids per community
    for cid, sids in comm_to_sent.items():
        seen = set()
        uniq: List[str] = []
        for s in sids:
            if s in seen:
                continue
            seen.add(s)
            uniq.append(s)
        comm_to_sent[cid] = uniq

    _save_jsonl(
        out_root / "communities" / "comm_to_sentences.jsonl",
        ({"community_id": cid, "sentence_ids": sids} for cid, sids in sorted(comm_to_sent.items())),
    )

    # 6) Embed nodes
    print(f"[communitykgrag] Embedding nodes on device={args.device} ...")
    node_model = SentenceTransformer(args.node_embed_model, device=args.device)

    node_ids = sorted(node_labels.keys())
    node_texts = [node_labels[nid] for nid in node_ids]
    node_vecs = _embed_texts(node_model, node_texts, batch_size=args.batch_size)
    dim_node = int(node_vecs.shape[1])

    # 7) Community vectors = mean(node vectors) for members
    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    comm_ids = sorted(comm_to_nodes.keys())
    comm_vecs = np.zeros((len(comm_ids), dim_node), dtype="float32")

    for j, cid in enumerate(comm_ids):
        members = comm_to_nodes[cid]
        idxs = [node_id_to_idx[m] for m in members if m in node_id_to_idx]
        if not idxs:
            continue
        v = node_vecs[idxs].mean(axis=0)
        # Re-normalize
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v = v / norm
        comm_vecs[j] = v.astype("float32")

    np.save(out_root / "embeddings.node_vectors.npy", node_vecs)
    np.save(out_root / "embeddings.community_vectors.npy", comm_vecs)

    # 8) Build community FAISS index
    comm_index = _faiss_index(args.community_index_metric, dim_node)
    comm_index.add(comm_vecs)
    faiss.write_index(comm_index, str(out_root / "faiss.community.index"))
    (out_root / "faiss.community.ids.json").write_text(
        json.dumps(comm_ids, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 9) Optional: sentence embeddings + sentence FAISS index
    dim_sent = 0
    if args.build_sentence_index:
        print(f"[communitykgrag] Embedding sentences on device={args.sentence_device} ...")
        sent_model = SentenceTransformer(args.sentence_embed_model, device=args.sentence_device)

        sent_ids = sorted(sent_map.keys())
        sent_texts = [sent_map[sid].get("text") or "" for sid in sent_ids]
        sent_vecs = _embed_texts(sent_model, sent_texts, batch_size=args.batch_size)
        dim_sent = int(sent_vecs.shape[1])

        np.save(out_root / "embeddings.sentence_vectors.npy", sent_vecs)

        sent_index = _faiss_index(args.sentence_index_metric, dim_sent)
        sent_index.add(sent_vecs)
        faiss.write_index(sent_index, str(out_root / "faiss.sentence.index"))
        (out_root / "faiss.sentence.ids.json").write_text(
            json.dumps(sent_ids, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 10) Write meta.json
    meta = BuilderMeta(
        dataset_key=args.dataset_key,
        kg_path=str(args.kg_path),
        output_dir=str(out_root),
        node_embed_model=args.node_embed_model,
        query_embed_model=args.query_embed_model,
        sentence_embed_model=args.sentence_embed_model,
        community_index_metric=args.community_index_metric,
        sentence_index_metric=args.sentence_index_metric,
        build_sentence_index=bool(args.build_sentence_index),
        device=args.device,
        sentence_device=args.sentence_device,
        batch_size=int(args.batch_size),
        num_nodes=len(node_ids),
        num_edges=len(edges),
        num_sentences=len(sent_map),
        num_communities=len(comm_ids),
        dim_node=int(dim_node),
        dim_sentence=int(dim_sent),
    )
    (out_root / "meta.json").write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8")

    # 11) Print summary
    print(f"[communitykgrag] built at: {out_root}")
    print(f"  nodes={meta.num_nodes} edges={meta.num_edges} sentences={meta.num_sentences} communities={meta.num_communities}")
    print(f"  community index: {out_root / 'faiss.community.index'}")
    if args.build_sentence_index:
        print(f"  sentence  index: {out_root / 'faiss.sentence.index'}")


if __name__ == "__main__":
    main()
