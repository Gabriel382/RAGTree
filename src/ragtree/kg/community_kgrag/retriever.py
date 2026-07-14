# ragtree/kg/community_kgrag/retriever.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: faiss (pip install faiss-cpu or faiss-gpu)") from e

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError("Missing dependency: sentence-transformers (pip install sentence-transformers)") from e


@dataclass
class EvidenceSentence:
    sentence_id: str
    document_id: Optional[str]
    text: str
    community_id: int
    score: float


class CommunityKGRetriever:
    """
    Two-stage retrieval:
      1) Query -> top communities (FAISS over community vectors)
      2) Within those communities -> rank sentences (either via sentence FAISS or cosine with stored vectors)

    IMPORTANT:
      - SentenceTransformer defaults to CUDA if available.
      - On your machine, CUDA is often full (vLLM/other process).
      - Therefore we default devices to CPU unless explicitly overridden.

    Artifacts under root/{dataset_key}:
      meta.json
      sentences.jsonl
      communities/comm_to_sentences.jsonl
      faiss.community.index + faiss.community.ids.json
      optional: faiss.sentence.index + faiss.sentence.ids.json + embeddings.sentence_vectors.npy
    """

    def __init__(
        self,
        dataset_root: Path,
        *,
        query_embed_model: str = "BAAI/bge-m3",
        sentence_embed_model: str = "BAAI/bge-m3",
        # NEW: device controls (default CPU to avoid CUDA OOM)
        query_device: str = "cpu",
        sentence_device: str = "cpu",
    ) -> None:
        self.root = Path(dataset_root)

        # Load meta
        self.meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))

        # Embedders (force device)
        self.query_embedder = SentenceTransformer(query_embed_model, device=query_device)
        self.sent_embedder = SentenceTransformer(sentence_embed_model, device=sentence_device)

        # Community index + community IDs
        self.comm_index = faiss.read_index(str(self.root / "faiss.community.index"))
        self.comm_ids: List[int] = json.loads((self.root / "faiss.community.ids.json").read_text(encoding="utf-8"))

        # Sentence store
        self.sentence_store: Dict[str, Dict] = {}
        with (self.root / "sentences.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.sentence_store[obj["sentence_id"]] = obj

        # community -> sentence ids
        self.comm_to_sents: Dict[int, List[str]] = {}
        with (self.root / "communities" / "comm_to_sentences.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.comm_to_sents[int(obj["community_id"])] = list(obj["sentence_ids"])

        # Optional sentence index
        self.has_sentence_index = bool(self.meta.get("build_sentence_index"))
        self.sent_index = None
        self.sent_ids: List[str] = []
        self.sent_vecs: Optional[np.ndarray] = None

        if self.has_sentence_index and (self.root / "faiss.sentence.index").exists():
            self.sent_index = faiss.read_index(str(self.root / "faiss.sentence.index"))
            self.sent_ids = json.loads((self.root / "faiss.sentence.ids.json").read_text(encoding="utf-8"))

            vec_path = self.root / "embeddings.sentence_vectors.npy"
            if vec_path.exists():
                self.sent_vecs = np.load(vec_path)

    # -------------------------
    # Embedding utilities
    # -------------------------

    def _embed_query(self, text: str) -> np.ndarray:
        v = self.query_embedder.encode([text], normalize_embeddings=True, convert_to_numpy=True)
        return v.astype("float32")

    def _embed_sents(self, sents: List[str], batch_size: int = 64) -> np.ndarray:
        v = self.sent_embedder.encode(sents, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True)
        return v.astype("float32")

    # -------------------------
    # Retrieval
    # -------------------------

    def retrieve(
        self,
        query: str,
        *,
        top_communities: int = 50,
        delta_percent: Optional[float] = None,
        top_sentences: int = 12,
        lambda_percent: Optional[float] = None,
        use_sentence_faiss: bool = True,
        max_candidates: int = 5000,
        sent_batch_size: int = 64,
    ) -> List[EvidenceSentence]:
        """
        If delta_percent is set, top_communities = ceil(delta% * |communities|).
        If lambda_percent is set, keep ceil(lambda% * candidates) AFTER scoring.
        """
        qv = self._embed_query(query)

        # Stage 1: retrieve communities
        M = len(self.comm_ids)
        if M == 0:
            return []

        if delta_percent is not None:
            top_communities = max(1, int(np.ceil((delta_percent / 100.0) * M)))
        top_communities = min(max(1, top_communities), M)

        scores, idxs = self.comm_index.search(qv, top_communities)
        selected_comm_ids: List[int] = []
        for idx in idxs[0].tolist():
            if 0 <= idx < len(self.comm_ids):
                selected_comm_ids.append(int(self.comm_ids[idx]))

        # Stage 2: gather candidate sentences from selected communities
        cand_pairs: List[Tuple[int, str]] = []
        for cid in selected_comm_ids:
            for sid in self.comm_to_sents.get(cid, []):
                cand_pairs.append((cid, sid))

        # de-dup by sentence_id keeping first community assignment
        seen = set()
        unique: List[Tuple[int, str]] = []
        for cid, sid in cand_pairs:
            if sid in seen:
                continue
            seen.add(sid)
            unique.append((cid, sid))

        if not unique:
            return []

        # cap candidates
        if len(unique) > max_candidates:
            unique = unique[:max_candidates]

        # Score candidates
        ranked: List[Tuple[str, float]] = []
        sid_to_comm = {sid: cid for cid, sid in unique}

        if self.has_sentence_index and use_sentence_faiss and self.sent_index is not None:
            # Retrieve a pool globally and filter by our candidate set
            pool = min(len(self.sent_ids), max(top_sentences * 50, top_sentences))
            s_scores, s_idxs = self.sent_index.search(qv, pool)

            cand_set = set(sid for _, sid in unique)
            for score, idx in zip(s_scores[0].tolist(), s_idxs[0].tolist()):
                if idx < 0 or idx >= len(self.sent_ids):
                    continue
                sid = self.sent_ids[idx]
                if sid in cand_set:
                    ranked.append((sid, float(score)))
                if len(ranked) >= max(top_sentences * 10, top_sentences):
                    break

            # If filtering yields too few, fallback to on-the-fly embedding
            if len(ranked) < top_sentences:
                texts = [self.sentence_store[sid].get("text") or "" for _, sid in unique]
                mat = self._embed_sents(texts, batch_size=sent_batch_size)
                sim = (mat @ qv[0]).tolist()
                ranked = [(unique[i][1], float(sim[i])) for i in range(len(unique))]
        else:
            # On-the-fly embedding
            texts = [self.sentence_store[sid].get("text") or "" for _, sid in unique]
            mat = self._embed_sents(texts, batch_size=sent_batch_size)
            sim = (mat @ qv[0]).tolist()
            ranked = [(unique[i][1], float(sim[i])) for i in range(len(unique))]

        ranked.sort(key=lambda x: x[1], reverse=True)

        if lambda_percent is not None:
            keep = max(1, int(np.ceil((lambda_percent / 100.0) * len(ranked))))
            ranked = ranked[:keep]

        ranked = ranked[:max(1, top_sentences)]

        out: List[EvidenceSentence] = []
        for sid, sc in ranked:
            s = self.sentence_store.get(sid, {})
            out.append(
                EvidenceSentence(
                    sentence_id=sid,
                    document_id=s.get("document_id"),
                    text=s.get("text") or "",
                    community_id=int(sid_to_comm.get(sid, -1)),
                    score=float(sc),
                )
            )
        return out
