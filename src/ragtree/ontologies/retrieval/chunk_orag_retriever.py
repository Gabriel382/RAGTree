# ragtree/ontologies/retrieval/chunk_orag_retriever.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: faiss (pip install faiss-cpu or faiss-gpu)") from e

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError("Missing dependency: sentence-transformers") from e


@dataclass
class ChunkORAGChunk:
    """Returned retrieval chunk (robust)."""
    chunk_id: int
    text: str
    subject_label: str
    score: float


class ChunkORAGRetriever:
    """
    Loads:
      - meta.json
      - chunks.jsonl
      - faiss.index
      - faiss.ids.json
    Then retrieves top-k leaf chunks by vector similarity.
    """

    def __init__(
        self,
        index_dir: Path,
        *,
        embed_model: str,
        device: str = "cpu",
    ) -> None:
        self.root = Path(index_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"Index dir not found: {self.root}")

        # Hard-force CPU if requested (prevents CUDA OOM)
        self.device = device
        if self.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        self.meta: Dict[str, Any] = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))

        self._embedder = SentenceTransformer(embed_model, device=self.device)

        self.index = faiss.read_index(str(self.root / "faiss.index"))
        self.ids: List[int] = json.loads((self.root / "faiss.ids.json").read_text(encoding="utf-8"))

        # Load chunks into a dict (usually manageable; ontology chunks aren't millions)
        self._chunk_map: Dict[int, Tuple[str, str]] = {}
        with (self.root / "chunks.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                cid = int(obj["chunk_id"])
                txt = str(obj.get("text", ""))
                lab = str(obj.get("subject_label", ""))
                self._chunk_map[cid] = (txt, lab)

    def retrieve(self, query: str, *, top_k: int = 8) -> List[ChunkORAGChunk]:
        """Return top_k chunks."""
        if not isinstance(query, str) or not query.strip():
            return []

        qvec = self._embedder.encode(
            [query],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        scores, idxs = self.index.search(qvec, int(top_k))
        idxs = idxs[0].tolist()
        scores = scores[0].tolist()

        out: List[ChunkORAGChunk] = []
        for faiss_row, sc in zip(idxs, scores):
            if faiss_row < 0 or faiss_row >= len(self.ids):
                continue
            chunk_id = int(self.ids[faiss_row])
            txt, lab = self._chunk_map.get(chunk_id, ("", ""))
            out.append(
                ChunkORAGChunk(
                    chunk_id=chunk_id,
                    text=txt,
                    subject_label=lab,
                    score=float(sc),
                )
            )
        return out