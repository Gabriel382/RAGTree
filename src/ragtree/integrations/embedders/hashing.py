# ragtree/integrations/embedders/hashing.py
"""Deterministic bag-of-tokens embedder. Zero dependencies, zero network.

Not a semantic model: designed for demos, CI and contract tests where
determinism matters more than embedding quality.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

__all__ = ["HashingEmbedder"]


class HashingEmbedder:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in text.lower().split():
                digest = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                vec[digest % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            vectors.append([x / norm for x in vec])
        return vectors
