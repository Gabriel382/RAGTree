# ragtree/vendor/byokg/kg_linker.py
from .utils import normalize_text

class KGLinker:
    """
    Links doc entities to KG nodes using string normalization.
    (In practice, youd use EntityLinker + embeddings.)
    """
    def __init__(self, graph_store):
        self.graph_store = graph_store
        self._cache = None

    def _build_cache(self):
        node_ids = self.graph_store.nodes()
        nodes = self.graph_store.get_nodes(node_ids)
        cache = {}
        for nid, n in nodes.items():
            name = None
            if isinstance(n, dict):
                name = n.get("name") or n.get("label") or n.get("title")
            if name:
                cache[normalize_text(name)] = nid
        self._cache = cache

    def link(self, mention: str):
        if self._cache is None:
            self._build_cache()
        k = normalize_text(mention)
        return self._cache.get(k)
