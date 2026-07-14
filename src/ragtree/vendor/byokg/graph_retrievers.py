# ragtree/vendor/byokg/graph_retrievers.py
from .entity_linker import EntityLinker
from .graph_traversal import GraphTraversal
from .graph_verbalizer import GraphVerbalizer
from .graph_reranker import GraphReranker

class GraphRetriever:
    """
    Multi-strategy graph retrieval:
      - link entities
      - traverse
      - rerank
      - verbalize
    """
    def __init__(self, graph_store, embedding_model=None):
        self.graph_store = graph_store
        self.linker = EntityLinker(graph_store, embedding_model=embedding_model)
        self.traversal = GraphTraversal(graph_store)
        self.verbalizer = GraphVerbalizer(graph_store)
        self.reranker = GraphReranker()

    def retrieve(self, mentions, top_k_nodes=5, max_hops=2, rel_types=None):
        # 1) link mentions -> candidate nodes
        candidates = []
        for m in mentions:
            for nid, s in self.linker.link(m, top_k=top_k_nodes, mode="hybrid"):
                candidates.append((nid, s))

        # 2) rerank nodes
        ranked_nodes = self.reranker.rerank(candidates, top_k=top_k_nodes)
        seed_nodes = [nid for nid, _ in ranked_nodes]

        # 3) traverse to collect edges
        edge_ids = self.traversal.bfs(seed_nodes, max_hops=max_hops, rel_types=rel_types)

        # 4) verbalize
        context = self.verbalizer.verbalize(node_ids=seed_nodes, edge_ids=edge_ids)

        return {
            "seed_nodes": seed_nodes,
            "edge_ids": edge_ids,
            "context": context,
        }
