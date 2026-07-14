# ragtree/vendor/byokg/neptune.py
import json
import requests
from .graphstore import GraphStore

class NeptuneGraphStore(GraphStore):
    def __init__(self, endpoint, port=8182, use_https=True):
        self.endpoint = endpoint
        self.port = port
        self.use_https = use_https

    def _url(self):
        scheme = "https" if self.use_https else "http"
        return f"{scheme}://{self.endpoint}:{self.port}/gremlin"

    def _post(self, query):
        payload = {"gremlin": query}
        r = requests.post(self._url(), data=json.dumps(payload))
        r.raise_for_status()
        return r.json()

    def get_schema(self):
        # This is simplified; Neptune schema introspection can be more complex.
        return {}

    def nodes(self):
        q = "g.V().id()"
        res = self._post(q)
        return [x["value"] for x in res.get("result", {}).get("data", [])]

    def get_nodes(self, node_ids):
        if not node_ids:
            return {}
        ids = ",".join([f"'{nid}'" for nid in node_ids])
        q = f"g.V({ids}).valueMap(true)"
        res = self._post(q)
        out = {}
        for row in res.get("result", {}).get("data", []):
            # Neptune returns nested maps; keep raw
            nid = row.get("id") if isinstance(row, dict) else None
            if nid is None:
                continue
            out[nid] = row
        return out

    def edges(self):
        q = "g.E().id()"
        res = self._post(q)
        return [x["value"] for x in res.get("result", {}).get("data", [])]

    def get_edges(self, edge_ids):
        if not edge_ids:
            return {}
        ids = ",".join([f"'{eid}'" for eid in edge_ids])
        q = f"g.E({ids}).valueMap(true)"
        res = self._post(q)
        out = {}
        for row in res.get("result", {}).get("data", []):
            eid = row.get("id") if isinstance(row, dict) else None
            if eid is None:
                continue
            out[eid] = row
        return out

    def find_edges(self, node_ids=None, rel_types=None, direction="out"):
        # Simplified traversal query
        if node_ids:
            ids = ",".join([f"'{nid}'" for nid in node_ids])
            if direction == "out":
                q = f"g.V({ids}).outE().id()"
            elif direction == "in":
                q = f"g.V({ids}).inE().id()"
            else:
                q = f"g.V({ids}).bothE().id()"
        else:
            q = "g.E().id()"

        if rel_types:
            # Neptune Gremlin label filter: hasLabel
            labels = ",".join([f"'{l}'" for l in rel_types])
            q = q.replace(".id()", f".hasLabel({labels}).id()")

        res = self._post(q)
        return [x["value"] for x in res.get("result", {}).get("data", [])]
