# ragtree/ontologies/retrieval/subontology.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import rdflib
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, OWL, XSD

# Reuse your schema helpers from Step 2
from ragtree.ontologies.linking.base import is_links_v1, upgrade_legacy_links


WD = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# -----------------------------
# Structured output types
# -----------------------------

@dataclass(frozen=True)
class SubOntologyProperty:
    uri: str
    label: Optional[str]
    domains: List[str]          # list of class URIs (full URIs)
    ranges: List[str]           # list of class URIs or datatype URIs (full URIs)
    # Convenience flags for debugging / analysis
    kept_because: str           # e.g., "domain_and_range_match", "no_constraints", ...


@dataclass(frozen=True)
class SubOntologyFragment:
    meta: Dict[str, Any]
    classes: Dict[str, Optional[str]]        # uri -> label
    datatypes: Dict[str, Optional[str]]      # uri -> label
    properties: List[SubOntologyProperty]
    num_triples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "_meta": self.meta,
            "classes": self.classes,
            "datatypes": self.datatypes,
            "properties": [asdict(p) for p in self.properties],
            "num_triples": self.num_triples,
        }

    def to_ttl(self, *, include_prefixes: bool = True) -> str:
        """
        Optional verbalization: return a compact TTL string for prompt inclusion.

        We output:
          - owl:Class declarations + labels for included classes
          - rdf:Property declarations + labels + rdfs:domain/range for included props
        """
        g = Graph()
        if include_prefixes:
            g.bind("rdfs", RDFS)
            g.bind("owl", OWL)
            g.bind("xsd", XSD)
            g.bind("wd", WD)
            g.bind("wdt", WDT)
            g.bind("skos", SKOS)

        # Classes
        for uri, label in self.classes.items():
            u = URIRef(uri)
            g.add((u, RDF.type, OWL.Class))
            if label:
                g.add((u, RDFS.label, Literal(label)))

        # Datatypes (best effort)
        for uri, label in self.datatypes.items():
            u = URIRef(uri)
            # Some ontologies model datatypes as rdfs:Datatype; we keep it light
            g.add((u, RDF.type, RDFS.Datatype))
            if label:
                g.add((u, RDFS.label, Literal(label)))

        # Properties
        for p in self.properties:
            pu = URIRef(p.uri)
            g.add((pu, RDF.type, RDF.Property))
            if p.label:
                g.add((pu, RDFS.label, Literal(p.label)))
            for d in p.domains:
                g.add((pu, RDFS.domain, URIRef(d)))
            for r in p.ranges:
                g.add((pu, RDFS.range, URIRef(r)))

        return g.serialize(format="turtle").decode("utf-8") if isinstance(g.serialize(format="turtle"), bytes) else g.serialize(format="turtle")


# -----------------------------
# Caching: parse ontology once
# -----------------------------

class OntologyGraphCache:
    """
    Simple in-process cache for parsed ontologies.

    Keyed by (resolved_path, mtime_ns).
    This keeps reproducibility sane: if TTL changes, cache invalidates.
    """
    _cache: Dict[Tuple[str, int], Graph] = {}

    @classmethod
    def load(cls, ttl_path: Path) -> Graph:
        ttl_path = ttl_path.resolve()
        mtime = ttl_path.stat().st_mtime_ns
        key = (str(ttl_path), mtime)
        if key in cls._cache:
            return cls._cache[key]

        g = Graph()
        g.parse(str(ttl_path), format="turtle")
        cls._cache[key] = g
        return g


# -----------------------------
# Retriever service
# -----------------------------

class SubOntologyRetriever:
    """
    GrOWL-RAG style sub-ontology retrieval.

    Input:
      - ontology_links (schema v1 preferred; legacy supported)
      - ontology graph (rdflib Graph) OR ttl_path to parse

    Output:
      - SubOntologyFragment (structured), plus optional TTL via fragment.to_ttl()

    Strategy (paper-aligned):
      - Keep classes / datatypes referenced by linked entities (top-k or selected)
      - Keep properties whose domain/range are satisfiable by the relevant classes/datatypes
      - Optionally include properties with no domain/range constraints
    """

    def __init__(
        self,
        *,
        ontology_key: str,
        ttl_path: Optional[Path] = None,
        graph: Optional[Graph] = None,
        include_unrestricted_properties: bool = True,
        max_properties: Optional[int] = None,
        max_classes: Optional[int] = None,
        pick: str = "candidates",   # "candidates" or "selected"
    ):
        if graph is None and ttl_path is None:
            raise ValueError("Provide either ttl_path or graph.")

        self.ontology_key = ontology_key
        self.ttl_path = ttl_path
        self._graph = graph
        self.include_unrestricted_properties = include_unrestricted_properties
        self.max_properties = max_properties
        self.max_classes = max_classes
        self.pick = pick

    @property
    def graph(self) -> Graph:
        if self._graph is not None:
            return self._graph
        assert self.ttl_path is not None
        return OntologyGraphCache.load(self.ttl_path)

    # ---------- Public API ----------

    def retrieve(
        self,
        *,
        ontology_links: Dict[str, Any],
        method: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> SubOntologyFragment:
        """
        Build a sub-ontology fragment from ontology_links.

        `method` here is the linker method that produced the links (for meta).
        """
        if not is_links_v1(ontology_links):
            # If legacy, upgrade (meta is best-effort)
            ontology_links = upgrade_legacy_links(ontology_links, method=method, ontology_key=self.ontology_key)

        relevant_class_uris, relevant_datatype_uris = self._extract_relevant_targets(ontology_links)

        # Optionally limit classes for prompt budget
        if self.max_classes is not None and len(relevant_class_uris) > self.max_classes:
            # keep deterministic ordering
            relevant_class_uris = set(sorted(relevant_class_uris)[: self.max_classes])

        g = self.graph

        class_labels = {uri: self._label_of(g, URIRef(uri)) for uri in relevant_class_uris}
        datatype_labels = {uri: self._label_of(g, URIRef(uri)) for uri in relevant_datatype_uris}

        props = self._select_properties(
            g=g,
            relevant_classes=relevant_class_uris,
            relevant_datatypes=relevant_datatype_uris,
        )

        if self.max_properties is not None and len(props) > self.max_properties:
            props = props[: self.max_properties]

        fragment_graph = self._build_fragment_graph(
            g=g,
            classes=class_labels,
            datatypes=datatype_labels,
            properties=props,
        )

        meta = {
            "schema_version": "1.0",
            "created_at": now_iso_utc(),
            "ontology_key": self.ontology_key,
            "ontology_path": str(self.ttl_path) if self.ttl_path else None,
            "linking_method": method,
            "linking_params": params or {},
            "retrieval_params": {
                "include_unrestricted_properties": self.include_unrestricted_properties,
                "max_properties": self.max_properties,
                "max_classes": self.max_classes,
                "pick": self.pick,
            },
            "stats": {
                "num_relevant_classes": len(relevant_class_uris),
                "num_relevant_datatypes": len(relevant_datatype_uris),
                "num_properties": len(props),
            },
        }

        return SubOntologyFragment(
            meta=meta,
            classes=class_labels,
            datatypes=datatype_labels,
            properties=props,
            num_triples=len(fragment_graph),
        )

    # ---------- Internals ----------

    def _extract_relevant_targets(self, links_v1: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
        """
        From schema v1 ontology_links, extract relevant class URIs and datatype URIs.

        We support:
          - pick="candidates": union of all candidates
          - pick="selected": use 'selected' field if present
        """
        by_entity = links_v1.get("by_entity") or {}
        relevant_classes: Set[str] = set()
        relevant_datatypes: Set[str] = set()

        for ent_id, payload in by_entity.items():
            if not isinstance(payload, dict):
                continue

            if self.pick == "selected":
                sel = payload.get("selected")
                if isinstance(sel, dict):
                    self._add_target_from_item(sel, relevant_classes, relevant_datatypes)
                continue

            # default: candidates
            candidates = payload.get("candidates") or []
            for item in candidates:
                if isinstance(item, dict):
                    self._add_target_from_item(item, relevant_classes, relevant_datatypes)

        return relevant_classes, relevant_datatypes

    def _add_target_from_item(self, item: Dict[str, Any], relevant_classes: Set[str], relevant_datatypes: Set[str]) -> None:
        kind = (item.get("target_kind") or "class").lower()
        uri = item.get("concept_uri")
        if not uri:
            return
        if kind == "datatype":
            relevant_datatypes.add(str(uri))
        elif kind == "property":
            # GrOWL-RAG typically links entities to classes/datatypes; properties come from retrieval.
            # We ignore property links by default.
            return
        else:
            relevant_classes.add(str(uri))

    def _label_of(self, g: Graph, res: URIRef) -> Optional[str]:
        # Try rdfs:label then skos:prefLabel; fallback to fragment
        for o in g.objects(res, RDFS.label):
            return str(o)
        for o in g.objects(res, SKOS.prefLabel):
            return str(o)
        return None

    def _objects_uris(self, g: Graph, s: URIRef, p: URIRef) -> List[str]:
        return [str(o) for o in g.objects(s, p) if isinstance(o, URIRef)]

    def _select_properties(
        self,
        *,
        g: Graph,
        relevant_classes: Set[str],
        relevant_datatypes: Set[str],
    ) -> List[SubOntologyProperty]:
        """
        Select properties that are satisfiable by relevant classes/datatypes.

        Logic (paper-aligned):
          - If property has domain(s): keep if ANY domain is in relevant_classes
          - If property has range(s): keep if ANY range is in relevant_classes OR relevant_datatypes
          - If missing domain or range: treat that side as fulfilled (open)
          - Optionally include properties with no constraints at all (no domain and no range)
        """
        selected: List[SubOntologyProperty] = []

        # Iterate over candidates: all subjects that have rdfs:domain or rdfs:range
        candidate_props: Set[URIRef] = set()
        for s in g.subjects(RDFS.domain, None):
            if isinstance(s, URIRef):
                candidate_props.add(s)
        for s in g.subjects(RDFS.range, None):
            if isinstance(s, URIRef):
                candidate_props.add(s)

        # Optionally include unrestricted properties (no domain/range)
        if self.include_unrestricted_properties:
            for s, p, o in g.triples((None, None, None)):
                if isinstance(s, URIRef):
                    # Heuristic: include wikidata direct props or anything explicitly typed as RDF.Property
                    if (str(s).startswith(str(WDT)) or (s, RDF.type, RDF.Property) in g):
                        # Only add if truly no domain/range constraints
                        has_domain = any(True for _ in g.objects(s, RDFS.domain))
                        has_range = any(True for _ in g.objects(s, RDFS.range))
                        if not has_domain and not has_range:
                            candidate_props.add(s)

        for prop in sorted(candidate_props, key=lambda u: str(u)):
            domains = [URIRef(u) for u in self._objects_uris(g, prop, RDFS.domain)]
            ranges = [URIRef(u) for u in self._objects_uris(g, prop, RDFS.range)]

            dom_ok = True
            rng_ok = True

            if domains:
                dom_ok = any(str(d) in relevant_classes for d in domains)

            if ranges:
                rng_ok = any((str(r) in relevant_classes) or (str(r) in relevant_datatypes) for r in ranges)

            if not dom_ok or not rng_ok:
                continue

            # Keep only the constraints that are actually relevant (like Bosch code)
            kept_domains = [str(d) for d in domains if str(d) in relevant_classes]
            kept_ranges = [str(r) for r in ranges if (str(r) in relevant_classes) or (str(r) in relevant_datatypes)]

            # Determine kept_because
            if domains and ranges:
                kept_because = "domain_and_range_match"
            elif domains and not ranges:
                kept_because = "domain_match_range_open"
            elif ranges and not domains:
                kept_because = "range_match_domain_open"
            else:
                kept_because = "no_constraints"

            selected.append(
                SubOntologyProperty(
                    uri=str(prop),
                    label=self._label_of(g, prop),
                    domains=kept_domains,
                    ranges=kept_ranges,
                    kept_because=kept_because,
                )
            )

        return selected

    def _build_fragment_graph(
        self,
        *,
        g: Graph,
        classes: Dict[str, Optional[str]],
        datatypes: Dict[str, Optional[str]],
        properties: List[SubOntologyProperty],
    ) -> Graph:
        """
        Build an rdflib Graph representing the fragment (for TTL output + triple counting).
        """
        out = Graph()
        out.bind("rdfs", RDFS)
        out.bind("owl", OWL)
        out.bind("xsd", XSD)
        out.bind("wd", WD)
        out.bind("wdt", WDT)
        out.bind("skos", SKOS)

        # Classes
        for uri, label in classes.items():
            u = URIRef(uri)
            out.add((u, RDF.type, OWL.Class))
            if label:
                out.add((u, RDFS.label, Literal(label)))

        # Datatypes
        for uri, label in datatypes.items():
            u = URIRef(uri)
            out.add((u, RDF.type, RDFS.Datatype))
            if label:
                out.add((u, RDFS.label, Literal(label)))

        # Properties
        for p in properties:
            pu = URIRef(p.uri)
            out.add((pu, RDF.type, RDF.Property))
            if p.label:
                out.add((pu, RDFS.label, Literal(p.label)))
            for d in p.domains:
                out.add((pu, RDFS.domain, URIRef(d)))
            for r in p.ranges:
                out.add((pu, RDFS.range, URIRef(r)))

        return out
