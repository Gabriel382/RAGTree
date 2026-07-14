# ragtree/vendor/byokg/entity_linker.py
from .graph_store_index import GraphStoreIndex

class EntityLinker:
    """
    Links free text mentions to KG node ids using multiple strategies.
    """
    def __init__(self, graph_store, embedding_model=None):
        self.index = GraphStoreIndex(graph_store, embedding_model=embedding_model)
        self.index.build()

    def link(self, mention, top_k=5, mode="hybrid"):
        if mode == "fuzzy":
            return self.index.query_fuzzy(mention, top_k=top_k)
        if mode == "dense":
            return self.index.query_dense(mention, top_k=top_k)
        # hybrid: merge
        fuzzy = self.index.query_fuzzy(mention, top_k=top_k)
        dense = self.index.query_dense(mention, top_k=top_k)

        scores = {}
        for nid, s in fuzzy:
            scores[nid] = max(scores.get(nid, 0.0), float(s))
        for nid, s in dense:
            scores[nid] = max(scores.get(nid, 0.0), float(s))

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked
