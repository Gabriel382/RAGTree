# ragtree/ontologies/retrieval/chunk_orag_retriever.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception as e:
    np = None  # type: ignore

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except Exception:
    SentenceTransformer = None  # type: ignore
    CrossEncoder = None  # type: ignore


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    level: int
    parent_id: Optional[str]
    subject_uri: str
    subject_label: str


class ChunkORAGRetriever:
    """
    Chunk-O-RAG retriever:
      - Load persisted FAISS leaf index + chunks.jsonl (hierarchical chunk tree)
      - Retrieve leaf chunks by vector similarity
      - Optionally auto-merge to parents (like LlamaIndex AutoMergingRetriever)
      - Optionally rerank with a cross-encoder
    """

    def __init__(
        self,
        index_dir: Path,
        *,
        embed_model: str = "BAAI/bge-m3",
        use_reranker: bool = False,
        reranker_model: str = "BAAI/bge-reranker-large",
        device: Optional[str] = None,
    ) -> None:
        if np is None:
            raise RuntimeError("Missing numpy. Install: pip install numpy")
        if faiss is None:
            raise RuntimeError("Missing faiss. Install: pip install faiss-cpu (or faiss-gpu)")
        if SentenceTransformer is None:
            raise RuntimeError("Missing sentence-transformers. Install: pip install sentence-transformers")

        self.index_dir = Path(index_dir)
        self.embed_model_name = embed_model
        self.use_reranker = use_reranker
        self.reranker_model_name = reranker_model
        self.device = device

        self._embedder = SentenceTransformer(embed_model, device=device) if device else SentenceTransformer(embed_model)
        self._reranker = None
        if use_reranker:
            if CrossEncoder is None:
                raise RuntimeError("CrossEncoder unavailable. Install sentence-transformers.")
            self._reranker = CrossEncoder(reranker_model, device=device) if device else CrossEncoder(reranker_model)

        self._chunks_by_id: Dict[str, Dict] = {}
        self._leaf_ids: List[str] = []
        self._leaf_index = None

        self._leaf_level: int = -1
        self._load()

    def _load(self) -> None:
        meta_path = self.index_dir / "meta.json"
        chunks_path = self.index_dir / "chunks.jsonl"
        faiss_path = self.index_dir / "leaf.index.faiss"

        if not meta_path.exists() or not chunks_path.exists() or not faiss_path.exists():
            raise FileNotFoundError(
                f"index_dir missing files. Expected meta.json, chunks.jsonl, leaf.index.faiss in {self.index_dir}"
            )

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._leaf_level = int(meta["leaf_level"])

        # load chunks
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                cid = obj["chunk_id"]
                self._chunks_by_id[cid] = obj

        # leaf ids in the exact order used to build FAISS
        self._leaf_ids = [cid for cid, c in self._chunks_by_id.items() if int(c["level"]) == self._leaf_level]
        # Important: preserve deterministic ordering: sort by chunk_id (builder created them in deterministic order)
        self._leaf_ids.sort()

        self._leaf_index = faiss.read_index(str(faiss_path))

    def _embed_query(self, query: str) -> "np.ndarray":
        v = self._embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        return v.astype("float32")

    def _auto_merge(self, leaf_chunk_ids: List[str]) -> List[str]:
        """
        Replace leaf chunks with their parents (one hop), then de-dup while preserving order.
        """
        out: List[str] = []
        seen = set()
        for cid in leaf_chunk_ids:
            c = self._chunks_by_id.get(cid)
            if not c:
                continue
            parent = c.get("parent_id")
            pick = parent if parent else cid
            if pick in seen:
                continue
            seen.add(pick)
            out.append(pick)
        return out

    def _rerank(self, query: str, chunk_ids: List[str], top_n: int) -> List[Tuple[str, float]]:
        assert self._reranker is not None
        pairs = [(query, self._chunks_by_id[cid]["text"]) for cid in chunk_ids]
        scores = self._reranker.predict(pairs)
        # scores can be list/np array
        scored = list(zip(chunk_ids, [float(s) for s in scores]))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 12,
        rerank_top_n: int = 6,
        auto_merge: bool = True,
    ) -> List[RetrievedChunk]:
        if self._leaf_index is None:
            raise RuntimeError("FAISS index not loaded")

        qv = self._embed_query(query)
        # search leaf vectors
        k = max(1, int(top_k))
        scores, idxs = self._leaf_index.search(qv, k)

        leaf_ids: List[str] = []
        for j, idx in enumerate(idxs[0].tolist()):
            if idx < 0 or idx >= len(self._leaf_ids):
                continue
            leaf_ids.append(self._leaf_ids[idx])

        candidate_ids = self._auto_merge(leaf_ids) if auto_merge else leaf_ids

        # rerank if enabled
        if self.use_reranker and self._reranker is not None:
            rr = self._rerank(query, candidate_ids, top_n=max(1, int(rerank_top_n)))
            final_ids = [cid for cid, _ in rr]
            score_map = {cid: s for cid, s in rr}
        else:
            final_ids = candidate_ids[: max(1, int(rerank_top_n))]
            score_map = {cid: 0.0 for cid in final_ids}

        out: List[RetrievedChunk] = []
        for cid in final_ids:
            c = self._chunks_by_id.get(cid)
            if not c:
                continue
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=str(c["text"]),
                    score=float(score_map.get(cid, 0.0)),
                    level=int(c["level"]),
                    parent_id=c.get("parent_id"),
                    subject_uri=str(c["subject_uri"]),
                    subject_label=str(c["subject_label"]),
                )
            )
        return out
