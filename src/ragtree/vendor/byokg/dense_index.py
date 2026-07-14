# ragtree/vendor/byokg/dense_index.py
import numpy as np

from .index import Index

class DenseIndex(Index):
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.items = []
        self.vectors = None

    def build(self, items):
        self.items = items
        vecs = []
        for it in items:
            vecs.append(self.embedding_model.embed(it))
        self.vectors = np.array(vecs, dtype=np.float32)

    def query(self, q, top_k=5):
        qv = np.array(self.embedding_model.embed(q), dtype=np.float32)
        # cosine sim
        denom = (np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(qv) + 1e-9)
        sims = (self.vectors @ qv) / denom
        idx = np.argsort(-sims)[:top_k]
        return [(self.items[i], float(sims[i])) for i in idx]
