# ragtree/vendor/byokg/graph_store_index.py
from .dense_index import DenseIndex
from .fuzzy_string import fuzzy_match

class GraphStoreIndex:
    """
    Builds multiple indexes over graph store content.
    """
    def __init__(self, graph_store, embedding_model=None):
        self.graph_store = graph_store
        self.embedding_model = embedding_model

        self.node_texts = []
        self.node_ids = []

        self.dense_index = None

    def _node_to_text(self, node):
        # node is dict-like; prefer label/name
        if isinstance(node, dict):
            for k in ["label", "name", "title", "id"]:
                if k in node and node[k]:
                    return str(node[k])
        return str(node)

    def build(self):
        node_ids = self.graph_store.nodes()
        nodes = self.graph_store.get_nodes(node_ids)

        self.node_ids = []
        self.node_texts = []

        for nid in node_ids:
            n = nodes.get(nid, {})
            self.node_ids.append(nid)
            self.node_texts.append(self._node_to_text(n))

        if self.embedding_model is not None:
            self.dense_index = DenseIndex(self.embedding_model)
            self.dense_index.build(self.node_texts)

    def query_fuzzy(self, q, top_k=5):
        scored = fuzzy_match(q, self.node_texts, top_k=top_k)
        # map back to ids
        out = []
        for txt, score in scored:
            try:
                i = self.node_texts.index(txt)
                out.append((self.node_ids[i], score))
            except ValueError:
                continue
        return out

    def query_dense(self, q, top_k=5):
        if self.dense_index is None:
            return []
        scored = self.dense_index.query(q, top_k=top_k)
        out = []
        for txt, score in scored:
            try:
                i = self.node_texts.index(txt)
                out.append((self.node_ids[i], score))
            except ValueError:
                continue
        return out
