# ragtree/processing/kg_rag/kg_retriever.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import re

from ragtree.kg.local_graphstore import LocalGraphStore


@dataclass
class KGFragment:
    seed_nodes: List[str]
    triples: List[Tuple[str, str, str]]
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed_nodes": self.seed_nodes,
            "triples": [[h, r, t] for (h, r, t) in self.triples],
            "meta": self.meta,
        }

    def to_text(self, max_triples: int = 200) -> str:
        lines = []
        for (h, r, t) in self.triples[:max_triples]:
            lines.append(f"{h}\t{r}\t{t}")
        return "\n".join(lines) if lines else "(empty KG fragment)"


class KGRetriever:
    """
    DocRE-oriented KG retriever:
      - Use doc entity IDs as seed nodes when available (best case).
      - Else, try a cheap mention->node match (optional extension).
      - Expand by hops in local adjacency.
    """

    def __init__(
        self,
        graph: LocalGraphStore,
        *,
        max_hops: int = 1,
        max_triples: int = 200,
        allowed_relations: Optional[Sequence[str]] = None,
    ) -> None:
        self.graph = graph
        self.max_hops = max_hops
        self.max_triples = max_triples
        self.allowed_relations = set(allowed_relations) if allowed_relations else None

    def _seed_nodes_from_doc(self, doc: Dict[str, Any]) -> List[str]:
        ents = doc.get("entities") or {}
        if isinstance(ents, dict):
            # If your KG is built from same dataset, entity IDs will align perfectly.
            return [eid for eid in ents.keys() if isinstance(eid, str)]
        return []

    def _filter_triples(self, triples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        if self.allowed_relations is None:
            return triples
        return [(h, r, t) for (h, r, t) in triples if r in self.allowed_relations]

    def retrieve(self, doc: Dict[str, Any], *, rel_types: Optional[Sequence[str]] = None) -> KGFragment:
        seed = self._seed_nodes_from_doc(doc)

        # If the strategy supplies allowed relation types (schema), we can filter KG by them.
        allowed = set(rel_types) if rel_types else None

        frontier = list(seed)
        visited = set(seed)
        triples: List[Tuple[str, str, str]] = []

        for _depth in range(self.max_hops):
            nxt = []
            for n in frontier:
                for (h, r, t) in self.graph.neighbors(n):
                    if allowed is not None and r not in allowed:
                        continue
                    triples.append((h, r, t))
                    if t not in visited:
                        visited.add(t)
                        nxt.append(t)

                # stop early if too many triples
                if len(triples) >= self.max_triples:
                    break
            frontier = nxt
            if len(triples) >= self.max_triples:
                break

        triples = triples[: self.max_triples]
        triples = self._filter_triples(triples)

        meta = {
            "max_hops": self.max_hops,
            "max_triples": self.max_triples,
            "seed_size": len(seed),
        }

        return KGFragment(seed_nodes=seed[:50], triples=triples, meta=meta)
