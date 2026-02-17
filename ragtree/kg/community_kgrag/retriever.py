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

    Artifacts expected under root/{dataset_key}:
      meta.json
      sentences.jsonl
      communities/comm_to_sentences.jsonl
      faiss.community.index + faiss.community.ids.json
      (optional) faiss.sentence.index + faiss.sentence.ids.json + embeddings.sentence_vectors.npy
    """

    def __init__(
        self,
        dataset_root: Path,
        *,
        query_embed_model: str = "BAAI/bge-m3",
        sentence_embed_model: str = "BAAI/bge-m3",
    ) -> None:
        self.root = Path(dataset_root)
        self.meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))

        self.query_embedder = SentenceTransformer(query_embed_model)
        self.sent_embedder = SentenceTransformer(sentence_embed_model)

        self.comm_index = faiss.read_index(str(self.root / "faiss.community.index"))
        self.comm_ids: List[int] = json.loads((self.root / "faiss.community.ids.json").read_text(encoding="utf-8"))

        # sentence store
        self.sentence_store: Dict[str, Dict] = {}
        with (self.root / "sentences.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.sentence_store[obj["sentence_id"]] = obj

        # comm -> sentence ids
        self.comm_to_sents: Dict[int, List[str]] = {}
        with (self.root / "communities" / "comm_to_sentences.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.comm_to_sents[int(obj["community_id"])] = list(obj["sentence_ids"])

        # optional sentence FAISS
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

    def _embed_query(self, text: str) -> np.ndarray:
        v = self.query_embedder.encode([text], normalize_embeddings=True, convert_to_numpy=True)
        return v.astype("float32")

    def _embed_sents(self, sents: List[str]) -> np.ndarray:
        v = self.sent_embedder.encode(sents, normalize_embeddings=True, convert_to_numpy=True)
        return v.astype("float32")

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
    ) -> List[EvidenceSentence]:
        """
        If delta_percent is set, top_communities is ignored and computed as ceil(delta% * |M|).
        If lambda_percent is set, candidates are reduced to ceil(lambda% * candidates) AFTER scoring.
        """

        qv = self._embed_query(query)

        # Stage 1: communities
        M = len(self.comm_ids)
        if delta_percent is not None:
            top_communities = max(1, int(np.ceil((delta_percent / 100.0) * M)))
        top_communities = min(max(1, top_communities), M)

        scores, idxs = self.comm_index.search(qv, top_communities)
        selected_comm_ids: List[int] = []
        for idx in idxs[0].tolist():
            if 0 <= idx < len(self.comm_ids):
                selected_comm_ids.append(int(self.comm_ids[idx]))

        # Stage 2: candidate sentences from communities
        cand_sids: List[Tuple[int, str]] = []  # (comm_id, sid)
        for cid in selected_comm_ids:
            for sid in self.comm_to_sents.get(cid, []):
                cand_sids.append((cid, sid))

        # de-dup by sentence_id keeping first community assignment
        seen = set()
        unique: List[Tuple[int, str]] = []
        for cid, sid in cand_sids:
            if sid in seen:
                continue
            seen.add(sid)
            unique.append((cid, sid))

        # cap candidates
        if len(unique) > max_candidates:
            unique = unique[:max_candidates]

        if not unique:
            return []

        # Score candidates
        if self.has_sentence_index and use_sentence_faiss and self.sent_index is not None:
            # Fast path: global sentence FAISS -> filter to our candidate set
            # 1) retrieve a pool larger than top_sentences to survive filtering
            pool = min(len(self.sent_ids), max(top_sentences * 50, top_sentences))
            s_scores, s_idxs = self.sent_index.search(qv, pool)

            cand_set = set(sid for _, sid in unique)
            ranked: List[Tuple[str, float]] = []
            for score, idx in zip(s_scores[0].tolist(), s_idxs[0].tolist()):
                if idx < 0 or idx >= len(self.sent_ids):
                    continue
                sid = self.sent_ids[idx]
                if sid in cand_set:
                    ranked.append((sid, float(score)))
                if len(ranked) >= max(top_sentences * 10, top_sentences):
                    break

            # If filtering yields too few, fallback to embedding candidates directly
            if len(ranked) < top_sentences:
                texts = [self.sentence_store[sid].get("text") or "" for _, sid in unique]
                mat = self._embed_sents(texts)
                sim = (mat @ qv[0]).tolist()
                ranked = [(unique[i][1], float(sim[i])) for i in range(len(unique))]
        else:
            # Compute candidate sentence embeddings on the fly
            texts = [self.sentence_store[sid].get("text") or "" for _, sid in unique]
            mat = self._embed_sents(texts)
            sim = (mat @ qv[0]).tolist()
            ranked = [(unique[i][1], float(sim[i])) for i in range(len(unique))]

        ranked.sort(key=lambda x: x[1], reverse=True)

        if lambda_percent is not None:
            keep = max(1, int(np.ceil((lambda_percent / 100.0) * len(ranked))))
            ranked = ranked[:keep]

        ranked = ranked[:max(1, top_sentences)]

        # Build output objects
        sid_to_comm = {sid: cid for cid, sid in unique}
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
