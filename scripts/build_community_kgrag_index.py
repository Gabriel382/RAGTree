#!/usr/bin/env python3
# scripts/build_community_kgrag_index.py
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm

try:
    import community as community_louvain  # python-louvain
except Exception as e:
    raise RuntimeError("Missing dependency: python-louvain (pip install python-louvain)") from e

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: faiss (pip install faiss-cpu or faiss-gpu)") from e

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError("Missing dependency: sentence-transformers (pip install sentence-transformers)") from e


# -------------------------
# Expected KG formats
# -------------------------
# Accept:
#  A) JSON with keys: edges (list), optional nodes (list)
#     edge: {"head": "...", "tail": "...", "rel": "...", "sentence_id": "...", "sentence_text": "...", "document_id": "..."}
#  B) JSONL edges, one per line with same schema


@dataclass
class BuilderMeta:
    dataset_key: str
    kg_path: str
    output_dir: str
    node_embed_model: str
    query_embed_model: str
    sentence_embed_model: str
    community_index_metric: str
    sentence_index_metric: str
    build_sentence_index: bool
    num_nodes: int
    num_edges: int
    num_sentences: int
    num_communities: int
    dim_node: int
    dim_sentence: int


def _read_json_or_jsonl(path: Path) -> Any:
    txt = path.read_text(encoding="utf-8")
    txt_strip = txt.lstrip()
    if txt_strip.startswith("{") or txt_strip.startswith("["):
        return json.loads(txt)
    # jsonl
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _normalize_edge(e: Dict[str, Any], fallback_idx: int) -> Dict[str, Any]:
    head = e.get("head") or e.get("h") or e.get("source")
    tail = e.get("tail") or e.get("t") or e.get("target")
    rel = e.get("rel") or e.get("relation") or e.get("r") or "related_to"

    if head is None or tail is None:
        raise ValueError(f"Edge missing head/tail: {e}")

    doc_id = e.get("document_id") or e.get("doc_id") or e.get("docId")
    sent_id = e.get("sentence_id") or e.get("sent_id") or e.get("sid")
    sent_txt = e.get("sentence_text") or e.get("sentence") or e.get("text")

    # deterministic sentence_id if missing but sentence text exists
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
        # optional labels
        "head_label": e.get("head_label") or e.get("h_label") or e.get("source_label"),
        "tail_label": e.get("tail_label") or e.get("t_label") or e.get("target_label"),
    }


def _collect_from_kg(kg_obj: Any) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, Dict[str, Any]]]:
    """
    Returns:
      edges: normalized edges
      node_id_to_label
      sentence_id_to_obj: {sentence_id: {"sentence_id","document_id","text"}}
    """
    edges_raw = None
    nodes_raw = None

    if isinstance(kg_obj, dict):
        edges_raw = kg_obj.get("edges") or kg_obj.get("relations") or kg_obj.get("triples")
        nodes_raw = kg_obj.get("nodes") or kg_obj.get("entities")
        if edges_raw is None:
            # maybe it's already an edge list dict?
            raise ValueError("KG JSON must contain 'edges' (or relations/triples).")
    elif isinstance(kg_obj, list):
        edges_raw = kg_obj
    else:
        raise ValueError("Unsupported KG format")

    node_labels: Dict[str, str] = {}
    if isinstance(nodes_raw, list):
        for n in nodes_raw:
            if not isinstance(n, dict):
                continue
            nid = n.get("id") or n.get("node_id") or n.get("name")
            lab = n.get("label") or n.get("text") or n.get("name") or nid
            if nid is not None:
                node_labels[str(nid)] = str(lab)

    edges: List[Dict[str, Any]] = []
    sent_map: Dict[str, Dict[str, Any]] = {}

    for i, e in enumerate(edges_raw):
        if not isinstance(e, dict):
            continue
        ne = _normalize_edge(e, i)
        edges.append(ne)

        # update node labels opportunistically
        if ne.get("head_label"):
            node_labels.setdefault(ne["head"], str(ne["head_label"]))
        if ne.get("tail_label"):
            node_labels.setdefault(ne["tail"], str(ne["tail_label"]))
        node_labels.setdefault(ne["head"], ne["head"])
        node_labels.setdefault(ne["tail"], ne["tail"])

        sid = ne["sentence_id"]
        if sid not in sent_map:
            sent_map[sid] = {
                "sentence_id": sid,
                "document_id": ne.get("document_id"),
                "text": ne.get("sentence_text") or "",
            }
        else:
            # fill missing text if later edge provides it
            if not sent_map[sid].get("text") and ne.get("sentence_text"):
                sent_map[sid]["text"] = ne["sentence_text"]

    return edges, node_labels, sent_map


def _build_graph(edges: List[Dict[str, Any]]) -> nx.Graph:
    """
    Undirected graph for Louvain (common choice).
    If you want directed, you can keep directed separately, but Louvain here uses undirected.
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


def _embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return emb.astype("float32")


def _faiss_index(metric: str, dim: int) -> faiss.Index:
    metric = metric.lower()
    if metric == "ip":
        return faiss.IndexFlatIP(dim)
    if metric == "l2":
        return faiss.IndexFlatL2(dim)
    raise ValueError(f"Unsupported metric: {metric}")


def main() -> None:
    ap = argparse.ArgumentParser("Build CommunityKG-RAG artifacts (communities + FAISS indexes).")
    ap.add_argument("--dataset-key", required=True)
    ap.add_argument("--kg-path", type=Path, required=True, help="KG edges as JSON or JSONL.")
    ap.add_argument("--output-dir", type=Path, required=True, help="Root output dir (e.g., data/kg_community).")

    ap.add_argument("--node-embed-model", default="BAAI/bge-m3")
    ap.add_argument("--query-embed-model", default="BAAI/bge-m3")
    ap.add_argument("--sentence-embed-model", default="BAAI/bge-m3")

    ap.add_argument("--community-index-metric", default="ip", choices=["ip", "l2"])
    ap.add_argument("--sentence-index-metric", default="ip", choices=["ip", "l2"])

    ap.add_argument("--build-sentence-index", action="store_true", help="If set, pre-embed sentences + FAISS.")
    ap.add_argument("--batch-size", type=int, default=64)

    args = ap.parse_args()

    out_root = args.output_dir / args.dataset_key
    _ensure_dir(out_root)

    kg_obj = _read_json_or_jsonl(args.kg_path)
    edges, node_labels, sent_map = _collect_from_kg(kg_obj)

    # Save normalized edges / nodes / sentences
    _save_jsonl(out_root / "edges.jsonl", edges)
    _save_jsonl(out_root / "nodes.jsonl", ({"node_id": k, "label": v} for k, v in sorted(node_labels.items())))
    _save_jsonl(out_root / "sentences.jsonl", (sent_map[k] for k in sorted(sent_map.keys())))

    # Build graph + Louvain
    G = _build_graph(edges)
    partition = community_louvain.best_partition(G, weight="weight")  # node_id -> community_id (int)

    # comm_to_nodes
    comm_to_nodes: Dict[int, List[str]] = {}
    for nid, cid in partition.items():
        comm_to_nodes.setdefault(int(cid), []).append(str(nid))
    for cid in comm_to_nodes:
        comm_to_nodes[cid].sort()

    # node_to_comm
    node_to_comm = {str(nid): int(cid) for nid, cid in partition.items()}

    _save_jsonl(out_root / "communities" / "node_to_comm.jsonl",
                ({"node_id": nid, "community_id": cid} for nid, cid in sorted(node_to_comm.items())))
    _save_jsonl(out_root / "communities" / "comm_to_nodes.jsonl",
                ({"community_id": cid, "node_ids": nids} for cid, nids in sorted(comm_to_nodes.items())))

    # Map sentences to communities (simple, robust):
    # If an edge has sentence_id, assign it to community of head (or tail if head missing).
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
        uniq = []
        for s in sids:
            if s in seen:
                continue
            seen.add(s)
            uniq.append(s)
        comm_to_sent[cid] = uniq

    _save_jsonl(out_root / "communities" / "comm_to_sentences.jsonl",
                ({"community_id": cid, "sentence_ids": sids} for cid, sids in sorted(comm_to_sent.items())))

    # Embeddings
    node_model = SentenceTransformer(args.node_embed_model)
    node_ids = sorted(node_labels.keys())
    node_texts = [node_labels[nid] for nid in node_ids]
    node_vecs = _embed_texts(node_model, node_texts, batch_size=args.batch_size)
    dim_node = node_vecs.shape[1]

    # community vectors = mean of member node vectors
    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    comm_ids = sorted(comm_to_nodes.keys())
    comm_vecs = np.zeros((len(comm_ids), dim_node), dtype="float32")

    for j, cid in enumerate(comm_ids):
        members = comm_to_nodes[cid]
        idxs = [node_id_to_idx[m] for m in members if m in node_id_to_idx]
        if not idxs:
            continue
        v = node_vecs[idxs].mean(axis=0)
        # already normalized-ish, but re-normalize
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        comm_vecs[j] = v.astype("float32")

    np.save(out_root / "embeddings.node_vectors.npy", node_vecs)
    np.save(out_root / "embeddings.community_vectors.npy", comm_vecs)

    # FAISS community index
    comm_index = _faiss_index(args.community_index_metric, dim_node)
    comm_index.add(comm_vecs)
    faiss.write_index(comm_index, str(out_root / "faiss.community.index"))

    # Sentence index (optional)
    dim_sent = 0
    if args.build_sentence_index:
        sent_model = SentenceTransformer(args.sentence_embed_model)
        sent_ids = sorted(sent_map.keys())
        sent_texts = [sent_map[sid].get("text") or "" for sid in sent_ids]
        sent_vecs = _embed_texts(sent_model, sent_texts, batch_size=args.batch_size)
        dim_sent = sent_vecs.shape[1]
        np.save(out_root / "embeddings.sentence_vectors.npy", sent_vecs)

        sent_index = _faiss_index(args.sentence_index_metric, dim_sent)
        sent_index.add(sent_vecs)
        faiss.write_index(sent_index, str(out_root / "faiss.sentence.index"))

        # persist sentence id order (align with FAISS rows)
        (out_root / "faiss.sentence.ids.json").write_text(json.dumps(sent_ids, ensure_ascii=False, indent=2), encoding="utf-8")

    # persist community id order (align with FAISS rows)
    (out_root / "faiss.community.ids.json").write_text(json.dumps(comm_ids, ensure_ascii=False, indent=2), encoding="utf-8")

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
        num_nodes=len(node_ids),
        num_edges=len(edges),
        num_sentences=len(sent_map),
        num_communities=len(comm_ids),
        dim_node=int(dim_node),
        dim_sentence=int(dim_sent),
    )
    (out_root / "meta.json").write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[communitykgrag] built at: {out_root}")
    print(f"  nodes={meta.num_nodes} edges={meta.num_edges} sentences={meta.num_sentences} communities={meta.num_communities}")
    print(f"  community index: {out_root / 'faiss.community.index'}")
    if args.build_sentence_index:
        print(f"  sentence  index: {out_root / 'faiss.sentence.index'}")


if __name__ == "__main__":
    main()
