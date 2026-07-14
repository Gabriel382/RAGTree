"""Neo4j adapter contract test. Needs a running server: set NEO4J_URI."""

import os

import pytest

pytest.importorskip("neo4j")

from ragtree.integrations.graphstores import Neo4jGraphStore
from tests.contract.bases import GraphStoreContractTests

pytestmark = [
    pytest.mark.integration,
    pytest.mark.neo4j,
    pytest.mark.skipif(
        not os.getenv("NEO4J_URI"), reason="NEO4J_URI not set; start neo4j via docker compose"
    ),
]


class TestNeo4jGraphStore(GraphStoreContractTests):
    list_nodes_query = "MATCH (n:Entity) RETURN n.id AS id, n.label AS label"
    list_edges_query = "MATCH ()-[r:REL]->() RETURN r.type AS type"

    @pytest.fixture
    def graph_store(self):
        store = Neo4jGraphStore()
        store.query("MATCH (n:Entity) WHERE n.id IN ['n1', 'n2'] DETACH DELETE n")
        yield store
        store.query("MATCH (n:Entity) WHERE n.id IN ['n1', 'n2'] DETACH DELETE n")
        store.close()
