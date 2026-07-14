"""RdflibOntologyStore + OntologyGuidedRetriever over the tiny fixture TTL."""

from pathlib import Path

import pytest

pytest.importorskip("rdflib")

from ragtree.core.protocols import OntologyStore
from ragtree.core.schemas import RAGTask
from ragtree.integrations.ontologies import RdflibOntologyStore
from ragtree.retrieval import OntologyGuidedRetriever

pytestmark = [pytest.mark.integration, pytest.mark.rdf]

TTL = Path(__file__).parents[1] / "fixtures" / "ontology" / "tiny_ontology.ttl"


@pytest.fixture(scope="module")
def store() -> RdflibOntologyStore:
    return RdflibOntologyStore(str(TTL))


def test_satisfies_protocol(store):
    assert isinstance(store, OntologyStore)


def test_loads_all_fixture_classes(store):
    labels = {c["label"] for c in store.search_concepts("anything", top_k=10)}
    assert {"Pump", "Mechanical seal", "Failure", "Maintenance", "Alarm", "Pressure spike"} <= labels


def test_search_ranks_relevant_concept_first(store):
    top = store.search_concepts("pump", top_k=2)
    assert top[0]["label"] == "Pump"
    seal = store.search_concepts("mechanical seal leakage", top_k=2)
    assert "Mechanical seal" in {c["label"] for c in seal}


def test_alias_matching(store):
    hits = store.search_concepts("pressure surge event", top_k=3)
    assert "Pressure spike" in {c["label"] for c in hits}


def test_ontology_guided_retriever_returns_concept_evidence(store):
    retriever = OntologyGuidedRetriever(store, top_k=3)
    spans = retriever.retrieve(RAGTask(task_type="qa", query="why did the pump seal fail"))
    assert spans
    assert all(span.metadata.get("source") == "ontology" for span in spans)
    assert any("Pump" in span.text or "seal" in span.text.lower() for span in spans)
