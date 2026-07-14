# ragtree/vendor/byokg/graph_reranker.py
class GraphReranker:
    """
    Rerank retrieved graph paths / nodes by simple heuristics.
    Placeholder for LLM-based reranking.
    """
    def rerank(self, items, top_k=5):
        # items: list[(item, score)]
        items = sorted(items, key=lambda x: x[1], reverse=True)
        return items[:top_k]
