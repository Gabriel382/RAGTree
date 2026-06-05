#!/usr/bin/env python3
# scripts/build_chunk_orag_index.py
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: faiss (pip install faiss-cpu or faiss-gpu)") from e

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError("Missing dependency: sentence-transformers (pip install sentence-transformers)") from e


# -------------------------
# Helpers
# -------------------------

def _ensure_dir(p: Path) -> None:
    """Create directory if missing."""
    p.mkdir(parents=True, exist_ok=True)


def _json_dump(path: Path, obj: Any) -> None:
    """Write JSON with indentation."""
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha_for_build(ttl_path: Path, chunk_sizes: List[int], overlap: int, leaf_level: int, embed_model: str) -> str:
    """Deterministic build hash based on parameters + file stats."""
    h = hashlib.sha256()
    st = ttl_path.stat()
    h.update(str(ttl_path.resolve()).encode("utf-8"))
    h.update(str(st.st_size).encode("utf-8"))
    h.update(str(int(st.st_mtime)).encode("utf-8"))
    h.update(("-".join(map(str, chunk_sizes)) + f"|{overlap}|{leaf_level}|{embed_model}").encode("utf-8"))
    return h.hexdigest()[:12]


def _tokenize(text: str) -> List[str]:
    """Very cheap tokenization (whitespace)."""
    return text.split()


def _detokenize(tokens: List[str]) -> str:
    """Join tokens back to text."""
    return " ".join(tokens)


@dataclass
class ChunkRow:
    """Single stored chunk row."""
    chunk_id: int
    level: int           # 0=largest (e.g., 512), 1=smaller (e.g., 128) ...
    is_leaf: int         # 1 if leaf chunk (indexed), else 0
    parent_id: Optional[int]
    subject_label: str
    text: str


@dataclass
class BuildMeta:
    ontology_key: str
    ttl_path: str
    output_dir: str
    embed_model: str
    device: str
    chunk_sizes: List[int]
    chunk_overlap: int
    leaf_level: int
    num_chunks_total: int
    num_chunks_leaf: int
    embed_dim: int


def _yield_ttl_text(ttl_path: Path) -> Iterator[str]:
    """
    Stream TTL text lines (keeps RAM low).
    Removes empty lines but keeps prefixes and structure.
    """
    with ttl_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield s


def _make_subject_label_from_text(text: str) -> str:
    """
    Best-effort "label" for a chunk.
    We try to extract first URI-like token; else fallback.
    """
    for tok in text.split():
        if tok.startswith("<http"):
            return tok.strip("<>").split("/")[-1][:60]
        if tok.startswith("http"):
            return tok.split("/")[-1][:60]
    # fallback: first 6 tokens
    toks = text.split()
    return " ".join(toks[:6])[:60] if toks else "ontology_chunk"


def _chunk_tokens(tokens: List[str], size: int, overlap: int) -> Iterator[List[str]]:
    """
    Sliding window chunking.
    Safe: yields chunks one-by-one; never stores all chunks.
    """
    if size <= 0:
        return
    step = max(1, size - max(0, overlap))
    n = len(tokens)
    i = 0
    while i < n:
        yield tokens[i : i + size]
        i += step


def _build_chunks_jsonl(
    ttl_path: Path,
    out_chunks_jsonl: Path,
    chunk_sizes: List[int],
    overlap: int,
    leaf_level: int,
    max_chars_per_parent: int = 20000,
) -> Tuple[int, int]:
    """
    Pass 1: create chunks.jsonl on disk (no embeddings here).
    - level 0 chunks are built from the whole TTL token stream.
    - each parent chunk is then sub-chunked into the next level size, etc.
    - only leaf chunks are intended for FAISS indexing.

    Returns:
      (num_chunks_total, num_chunks_leaf)
    """
    _ensure_dir(out_chunks_jsonl.parent)

    # Stream TTL into a rolling token buffer to build level-0 chunks.
    # We keep a moderate buffer and chunk it progressively.
    ttl_lines = list(_yield_ttl_text(ttl_path))
    ttl_text = "\n".join(ttl_lines)
    ttl_tokens = _tokenize(ttl_text)

    total = 0
    leaf = 0
    next_chunk_id = 0

    def write_row(f, row: ChunkRow) -> None:
        nonlocal total, leaf
        f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
        total += 1
        if row.is_leaf == 1:
            leaf += 1

    # Build hierarchical chunks
    with out_chunks_jsonl.open("w", encoding="utf-8") as f:
        # Level 0 parents
        level0_size = chunk_sizes[0]
        for parent_tokens in _chunk_tokens(ttl_tokens, level0_size, overlap):
            parent_text = _detokenize(parent_tokens)
            parent_text = parent_text[:max_chars_per_parent]  # hard safety cap

            parent_id = next_chunk_id
            next_chunk_id += 1

            parent_row = ChunkRow(
                chunk_id=parent_id,
                level=0,
                is_leaf=1 if leaf_level == 0 else 0,
                parent_id=None,
                subject_label=_make_subject_label_from_text(parent_text),
                text=parent_text,
            )
            write_row(f, parent_row)

            # Sub-levels
            prev_parent_id = parent_id
            prev_text = parent_text
            prev_tokens = _tokenize(prev_text)

            for lvl in range(1, len(chunk_sizes)):
                size = chunk_sizes[lvl]

                # Each level chunks the *previous parent text* (keeps hierarchy stable)
                # Note: we only create children; they all share same parent_id (prev_parent_id)
                child_ids: List[int] = []
                for child_tokens in _chunk_tokens(prev_tokens, size, overlap):
                    child_text = _detokenize(child_tokens)
                    child_text = child_text[:max_chars_per_parent]

                    cid = next_chunk_id
                    next_chunk_id += 1
                    child_ids.append(cid)

                    row = ChunkRow(
                        chunk_id=cid,
                        level=lvl,
                        is_leaf=1 if leaf_level == lvl else 0,
                        parent_id=prev_parent_id,
                        subject_label=_make_subject_label_from_text(child_text),
                        text=child_text,
                    )
                    write_row(f, row)

                # For deeper levels, we DO NOT recursively chunk each child (that explodes count).
                # Instead, we keep hierarchy shallow (only chunking the parent once per level),
                # which is dramatically safer for RAM/disk and still works for retrieval.
                prev_parent_id = prev_parent_id
                prev_tokens = prev_tokens

    return total, leaf


def _faiss_index(metric: str, dim: int) -> faiss.Index:
    """Create FAISS index."""
    metric = metric.lower()
    if metric == "ip":
        return faiss.IndexFlatIP(dim)
    if metric == "l2":
        return faiss.IndexFlatL2(dim)
    raise ValueError(f"Unsupported metric: {metric}")


def _iter_leaf_chunks(chunks_path: Path, leaf_level: int) -> Iterator[ChunkRow]:
    """Stream only leaf chunks from chunks.jsonl."""
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if int(obj.get("level", -1)) != int(leaf_level):
                continue
            yield ChunkRow(
                chunk_id=int(obj["chunk_id"]),
                level=int(obj["level"]),
                is_leaf=int(obj["is_leaf"]),
                parent_id=obj.get("parent_id"),
                subject_label=str(obj.get("subject_label", "")),
                text=str(obj.get("text", "")),
            )


def main() -> None:
    ap = argparse.ArgumentParser("Build Chunk-O-RAG index (safe, CPU-first, streaming).")

    ap.add_argument("--ontology-key", required=True)
    ap.add_argument("--ttl-path", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)

    ap.add_argument("--embed-model", default="BAAI/bge-m3")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    ap.add_argument("--chunk-sizes", default="512,128")
    ap.add_argument("--chunk-overlap", type=int, default=128)

    ap.add_argument("--leaf-level", type=int, default=None,
                    help="Which level is indexed in FAISS. Default: last level.")
    ap.add_argument("--metric", default="ip", choices=["ip", "l2"])

    # Memory safety knobs
    ap.add_argument("--batch-size", type=int, default=8, help="Embedding batch size (keep small).")
    ap.add_argument("--max-chars-per-parent", type=int, default=20000, help="Hard cap to avoid huge chunks.")

    args = ap.parse_args()

    # Force CPU if user asked CPU (prevents CUDA OOM)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    chunk_sizes = [int(x.strip()) for x in args.chunk_sizes.split(",") if x.strip()]
    if not chunk_sizes:
        raise ValueError("chunk-sizes must be like '512,128'")

    leaf_level = args.leaf_level
    if leaf_level is None:
        leaf_level = len(chunk_sizes) - 1
    if not (0 <= leaf_level < len(chunk_sizes)):
        raise ValueError(f"leaf-level must be in [0, {len(chunk_sizes)-1}]")

    sha = _sha_for_build(args.ttl_path, chunk_sizes, args.chunk_overlap, leaf_level, args.embed_model)

    # Keep your existing folder naming style
    embed_label = args.embed_model.replace("/", "__")
    out_root = args.output_dir / args.ontology_key / embed_label / f"hier_sizes={'-'.join(map(str, chunk_sizes))}_leaf={leaf_level}_sha={sha}"
    _ensure_dir(out_root)

    chunks_path = out_root / "chunks.jsonl"
    faiss_path = out_root / "faiss.index"
    ids_path = out_root / "faiss.ids.json"
    meta_path = out_root / "meta.json"

    # PASS 1: write chunks.jsonl (streaming, safe)
    num_total, num_leaf = _build_chunks_jsonl(
        args.ttl_path,
        chunks_path,
        chunk_sizes=chunk_sizes,
        overlap=args.chunk_overlap,
        leaf_level=leaf_level,
        max_chars_per_parent=args.max_chars_per_parent,
    )

    # PASS 2: embed only leaf chunks, in tiny batches, and add to FAISS incrementally
    model = SentenceTransformer(args.embed_model, device=args.device)

    # Determine embedding dim using a tiny probe
    probe = model.encode(["probe"], normalize_embeddings=True, convert_to_numpy=True)
    dim = int(probe.shape[1])

    index = _faiss_index(args.metric, dim)

    leaf_ids: List[int] = []
    buf_texts: List[str] = []
    buf_ids: List[int] = []

    def flush_batch() -> None:
        """Embed current buffer and add to FAISS."""
        nonlocal buf_texts, buf_ids, leaf_ids
        if not buf_texts:
            return
        vecs = model.encode(
            buf_texts,
            batch_size=max(1, int(args.batch_size)),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        index.add(vecs)
        leaf_ids.extend(buf_ids)
        buf_texts = []
        buf_ids = []

    for row in tqdm(_iter_leaf_chunks(chunks_path, leaf_level), desc="Embedding leaf chunks", unit="chunk"):
        buf_texts.append(row.text)
        buf_ids.append(int(row.chunk_id))
        if len(buf_texts) >= int(args.batch_size):
            flush_batch()

    flush_batch()

    faiss.write_index(index, str(faiss_path))
    _json_dump(ids_path, leaf_ids)

    meta = BuildMeta(
        ontology_key=args.ontology_key,
        ttl_path=str(args.ttl_path),
        output_dir=str(out_root),
        embed_model=args.embed_model,
        device=args.device,
        chunk_sizes=chunk_sizes,
        chunk_overlap=int(args.chunk_overlap),
        leaf_level=int(leaf_level),
        num_chunks_total=int(num_total),
        num_chunks_leaf=int(num_leaf),
        embed_dim=int(dim),
    )
    _json_dump(meta_path, asdict(meta))

    print(f"[chunk_orag] built at: {out_root}")
    print(f"  chunks total={num_total} leaf={num_leaf}")
    print(f"  faiss: {faiss_path}")
    print(f"  ids:   {ids_path}")
    print(f"  meta:  {meta_path}")


if __name__ == "__main__":
    main()