# ragtree/vendor/byokg/graph_verbalizer.py
class GraphVerbalizer:
    """
    Converts nodes/edges into text.
    """
    def __init__(self, graph_store):
        self.graph_store = graph_store

    def verbalize(self, node_ids=None, edge_ids=None, max_items=200):
        node_ids = node_ids or []
        edge_ids = edge_ids or []

        nodes = self.graph_store.get_nodes(node_ids) if node_ids else {}
        edges = self.graph_store.get_edges(edge_ids) if edge_ids else {}

        lines = []

        for nid, n in list(nodes.items())[:max_items]:
            lines.append(f"NODE {nid}: {n}")

        for eid, e in list(edges.items())[:max_items]:
            lines.append(f"EDGE {eid}: {e}")

        return "\n".join(lines)
