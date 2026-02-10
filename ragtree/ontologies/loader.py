# ragtree/ontologies/loader.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import rdflib  # pip install rdflib


@dataclass
class OntologyConcept:
    uri: str
    label: str
    description: Optional[str] = None
    aliases: List[str] = None


class OntologyIndex:
    def __init__(self, concepts: List[OntologyConcept]):
        self.concepts = concepts

    @classmethod
    def from_turtle(cls, ttl_path: Path) -> "OntologyIndex":
        g = rdflib.Graph()
        g.parse(str(ttl_path), format="turtle")

        concepts: List[OntologyConcept] = []

        for s in g.subjects(rdflib.RDF.type, rdflib.OWL.Class):
            uri = str(s)
            # labels
            labels = [str(o) for o in g.objects(s, rdflib.RDFS.label)]
            # optionally skos:prefLabel / altLabel
            try:
                SKOS = rdflib.Namespace("http://www.w3.org/2004/02/skos/core#")
                labels += [str(o) for o in g.objects(s, SKOS.prefLabel)]
                aliases = [str(o) for o in g.objects(s, SKOS.altLabel)]
            except Exception:
                aliases = []

            label = labels[0] if labels else uri.split("#")[-1]
            description = None
            for o in g.objects(s, rdflib.RDFS.comment):
                description = str(o)
                break

            concepts.append(
                OntologyConcept(
                    uri=uri,
                    label=label,
                    description=description,
                    aliases=aliases,
                )
            )

        return cls(concepts)
