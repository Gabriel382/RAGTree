# ragtree/integrations/embedders/sentence_transformers.py
"""sentence-transformers embedder adapter."""

from __future__ import annotations

from typing import Any

from ragtree.core.errors import require_extra

__all__ = ["SentenceTransformersEmbedder"]


class SentenceTransformersEmbedder:
    """Embedder over sentence-transformers (extra: ``embeddings``)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", **model_kwargs: Any) -> None:
        require_extra("sentence_transformers", "embeddings")
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, **model_kwargs)

    def embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.encode(list(texts), **kwargs)]
