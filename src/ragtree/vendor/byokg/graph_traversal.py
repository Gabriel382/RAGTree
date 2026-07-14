# ragtree/vendor/byokg/graph_traversal.py
from collections import deque

class GraphTraversal:
    """
    Basic BFS traversal for a GraphStore.
    """
    def __init__(self, graph_store):
        self.graph_store = graph_store

    def bfs(self, start_nodes, max_hops=2, rel_types=None):
        visited = set(start_nodes)
        q = deque([(n, 0) for n in start_nodes])

        edges_collected = set()

        while q:
            node, depth = q.popleft()
            if depth >= max_hops:
                continue

            edge_ids = self.graph_store.find_edges([node], rel_types=rel_types, direction="out")
            for eid in edge_ids:
                edges_collected.add(eid)

            edges = self.graph_store.get_edges(list(edge_ids))
            for eid, e in edges.items():
                # attempt to read tail
                tail = e.get("inV") or e.get("to") or e.get("tail")
                if tail and tail not in visited:
                    visited.add(tail)
                    q.append((tail, depth + 1))

        return list(edges_collected)
