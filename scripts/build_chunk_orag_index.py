#!/usr/bin/env python3
# scripts/build_chunk_orag_index.py
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Optional deps
try:
    import rdflib
    from rdflib.namespace import RDF, RDFS, OWL, SKOS
except Exception as e:
    rdflib = None  # type: ignore

try:
    import numpy as np
except Exception as e:
    np = None  # type: ignore

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore


@dataclass
class ChunkMeta:
    chunk_id: str
    level: int                 # 0=big, 1=medium, 2=small (leaf)
    parent_id: Optional[str]
    subject_uri: str
    subject_label: str
    text: str


@dataclass
class IndexMeta:
    ontology_key: str
    ttl_path: str
    embed_model: str
    chunk_sizes: List[int]
    chunk_overlap: int
    leaf_level: int
    num_subjects: int
    num_chunks: int
    dim: int
    index_type: str
    sha: str


def _sha_of_params(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()[:16]


def _ensure_deps() -> None:
    if rdflib is None:
        raise RuntimeError("Missing dependency: rdflib. Install: pip install rdflib")
    if np is None:
        raise RuntimeError("Missing dependency: numpy. Install: pip install numpy")
    if faiss is None:
        raise RuntimeError("Missing dependency: faiss. Install: pip install faiss-cpu (or faiss-gpu)")
    if SentenceTransformer is None:
        raise RuntimeError("Missing dependency: sentence-transformers. Install: pip install sentence-transformers")


def _load_ontology_graph(ttl_path: Path) -> "rdflib.Graph":
    g = rdflib.Graph()
    # format auto-detected by rdflib; TTL should parse
    g.parse(str(ttl_path))
    return g


def _uri_to_label(g: "rdflib.Graph", uri: "rdflib.term.Identifier") -> str:
    # Prefer SKOS prefLabel / RDFS label
    for p in (SKOS.prefLabel, RDFS.label):
        for o in g.objects(uri, p):
            try:
                return str(o)
            except Exception:
                continue
    # fallback to local name
    s = str(uri)
    if "#" in s:
        return s.split("#")[-1]
    if "/" in s:
        return s.rstrip("/").split("/")[-1]
    return s


def _collect_subject_records(g: "rdflib.Graph") -> List[Tuple[str, str, str]]:
    """
    Create a textual record per subject:
      - label(s), comment(s), type(s), and selected outgoing edges
    Returns: [(subject_uri, subject_label, record_text), ...]
    """
    subjects = set(g.subjects())
    # filter out blank nodes for indexing
    subjects = {s for s in subjects if not isinstance(s, rdflib.BNode)}

    records: List[Tuple[str, str, str]] = []

    # Predicates to prioritize in text
    preferred_text_preds = {RDFS.comment, SKOS.definition, SKOS.scopeNote}
    preferred_label_preds = {RDFS.label, SKOS.prefLabel, SKOS.altLabel}

    for s in sorted(subjects, key=lambda x: str(x)):
        s_uri = str(s)
        s_label = _uri_to_label(g, s)

        labels: List[str] = []
        comments: List[str] = []
        types: List[str] = []

        for p in preferred_label_preds:
            for o in g.objects(s, p):
                labels.append(str(o))

        for p in preferred_text_preds:
            for o in g.objects(s, p):
                comments.append(str(o))

        for o in g.objects(s, RDF.type):
            if isinstance(o, rdflib.BNode):
                continue
            types.append(_uri_to_label(g, o))

        # Outgoing edges summary (compact)
        # Keep only non-bnode objects, and cap to avoid huge subjects
        edge_lines: List[str] = []
        cap_edges = 60
        count = 0
        for p, o in g.predicate_objects(s):
            if count >= cap_edges:
                break
            if isinstance(o, rdflib.BNode):
                continue
            p_str = _uri_to_label(g, p)
            o_str = _uri_to_label(g, o) if isinstance(o, rdflib.URIRef) else str(o)
            # skip trivial label/comment edges to avoid repetition
            if p in preferred_label_preds or p in preferred_text_preds:
                continue
            edge_lines.append(f"{p_str}: {o_str}")
            count += 1

        parts: List[str] = []
        parts.append(f"SUBJECT: {s_label}")
        parts.append(f"URI: {s_uri}")

        if types:
            parts.append("TYPES: " + "; ".join(sorted(set(types))[:10]))
        if labels:
            parts.append("LABELS: " + "; ".join(sorted(set(labels))[:10]))
        if comments:
            # keep comments short-ish
            compact = " ".join(comments)
            compact = compact.strip()
            if len(compact) > 1200:
                compact = compact[:1200] + "..."
            parts.append("DESCRIPTION: " + compact)
        if edge_lines:
            parts.append("PROPERTIES:\n- " + "\n- ".join(edge_lines))

        record_text = "\n".join(parts).strip()
        records.append((s_uri, s_label, record_text))

    return records


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    # Simple character-based chunking (robust, no tokenizer dependency)
    text = text.strip()
    if not text:
        return []
    if chunk_size <= 0:
        return [text]

    out: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        out.append(text[start:end].strip())
        if end >= n:
            break
        start = max(0, end - overlap)
        if start == end:
            start = end
    return [c for c in out if c]


def _build_hierarchical_chunks(
    subject_uri: str,
    subject_label: str,
    record_text: str,
    chunk_sizes: Sequence[int],
    overlap: int,
) -> List[ChunkMeta]:
    """
    Levels follow chunk_sizes order:
      level 0 = chunk_sizes[0] (largest)
      ...
      leaf level = last
    We create a simple parent pointer:
      each chunk at level L+1 points to the chunk at level L that contains its start offset.
    For simplicity, we approximate containment by ordering chunks and mapping by position.
    """
    # We chunk independently per level, then map children to parents by index proportion.
    # This is good enough for "AutoMerge" behavior.
    level_chunks: List[List[str]] = []
    for sz in chunk_sizes:
        level_chunks.append(_chunk_text(record_text, sz, overlap))

    metas: List[ChunkMeta] = []
    # stable id base per subject
    base = hashlib.md5(f"{subject_uri}".encode("utf-8")).hexdigest()[:10]

    # Create IDs per level
    ids_by_level: List[List[str]] = []
    for lvl, chunks in enumerate(level_chunks):
        ids = [f"ch_{base}_L{lvl}_{i}" for i in range(len(chunks))]
        ids_by_level.append(ids)

    # Parent assignment: child index maps to parent index by ratio
    for lvl, chunks in enumerate(level_chunks):
        for i, ctext in enumerate(chunks):
            parent_id: Optional[str] = None
            if lvl > 0 and len(ids_by_level[lvl - 1]) > 0:
                parent_idx = min(
                    len(ids_by_level[lvl - 1]) - 1,
                    int(math.floor(i * len(ids_by_level[lvl - 1]) / max(1, len(ids_by_level[lvl])))),
                )
                parent_id = ids_by_level[lvl - 1][parent_idx]

            metas.append(
                ChunkMeta(
                    chunk_id=ids_by_level[lvl][i],
                    level=lvl,
                    parent_id=parent_id,
                    subject_uri=subject_uri,
                    subject_label=subject_label,
                    text=ctext,
                )
            )

    return metas


def _embed_texts(model: "SentenceTransformer", texts: List[str], batch_size: int = 32) -> "np.ndarray":
    # returns float32 matrix
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return emb.astype("float32")


def _save_jsonl(path: Path, items: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main() -> None:
    _ensure_deps()

    ap = argparse.ArgumentParser("Build Chunk-O-RAG ontology index (FAISS + chunks.jsonl).")
    ap.add_argument("--ontology-key", required=True)
    ap.add_argument("--ttl-path", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)

    ap.add_argument("--embed-model", default="BAAI/bge-m3")
    ap.add_argument("--chunk-sizes", default="2048,512,128", help="Comma-separated. First is largest; last is leaf.")
    ap.add_argument("--chunk-overlap", type=int, default=128)
    ap.add_argument("--leaf-level", type=int, default=None, help="Default: last level.")
    ap.add_argument("--index-type", default="ip", choices=["ip", "l2"], help="FAISS index metric. Use ip with normalized embeddings.")
    ap.add_argument("--batch-size", type=int, default=32)

    args = ap.parse_args()

    chunk_sizes = [int(x.strip()) for x in args.chunk_sizes.split(",") if x.strip()]
    if not chunk_sizes:
        raise ValueError("chunk_sizes empty")

    leaf_level = args.leaf_level if args.leaf_level is not None else (len(chunk_sizes) - 1)
    if leaf_level < 0 or leaf_level >= len(chunk_sizes):
        raise ValueError(f"leaf_level {leaf_level} invalid for {len(chunk_sizes)} sizes")

    # Build a deterministic directory name from params (good for many ontologies)
    sha = _sha_of_params(
        args.ontology_key,
        str(args.ttl_path.resolve()),
        args.embed_model,
        ",".join(map(str, chunk_sizes)),
        str(args.chunk_overlap),
        str(leaf_level),
        args.index_type,
    )
    out_dir = args.output_dir / args.ontology_key / "bge-m3" / f"hier_sizes={'-'.join(map(str, chunk_sizes))}_leaf={leaf_level}_sha={sha}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load ontology & records
    g = _load_ontology_graph(args.ttl_path)
    records = _collect_subject_records(g)

    all_chunks: List[ChunkMeta] = []
    for s_uri, s_label, rec_text in records:
        metas = _build_hierarchical_chunks(s_uri, s_label, rec_text, chunk_sizes, args.chunk_overlap)
        all_chunks.extend(metas)

    # Prepare embeddings ONLY for leaf level (fast + matches AutoMerge design)
    leaf_chunks = [c for c in all_chunks if c.level == leaf_level]
    leaf_texts = [c.text for c in leaf_chunks]

    model = SentenceTransformer(args.embed_model)
    emb = _embed_texts(model, leaf_texts, batch_size=args.batch_size)
    dim = emb.shape[1]

    # Build FAISS index
    if args.index_type == "ip":
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)

    index.add(emb)

    # Persist
    faiss_path = out_dir / "leaf.index.faiss"
    faiss.write_index(index, str(faiss_path))

    chunks_path = out_dir / "chunks.jsonl"
    _save_jsonl(
        chunks_path,
        (
            asdict(c)
            for c in all_chunks
        ),
    )

    meta = IndexMeta(
        ontology_key=args.ontology_key,
        ttl_path=str(args.ttl_path),
        embed_model=args.embed_model,
        chunk_sizes=list(chunk_sizes),
        chunk_overlap=args.chunk_overlap,
        leaf_level=leaf_level,
        num_subjects=len(records),
        num_chunks=len(all_chunks),
        dim=dim,
        index_type=args.index_type,
        sha=sha,
    )
    (out_dir / "meta.json").write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[chunk-orag] built index at: {out_dir}")
    print(f"  leaf vectors: {len(leaf_chunks)} | dim: {dim}")
    print(f"  chunks.jsonl: {chunks_path}")
    print(f"  faiss:        {faiss_path}")


if __name__ == "__main__":
    main()
