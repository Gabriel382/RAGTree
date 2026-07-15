"""FastAPI surface tests (extra: api). In-process via TestClient."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from ragtree import __version__
from ragtree.apps.api.app import create_app

pytestmark = [pytest.mark.integration, pytest.mark.api]

DOCS = [
    {"id": "maint-001", "text": "Pump P-102 failed because the mechanical seal wore out."},
    {"id": "maint-002", "text": "Routine maintenance was performed in June."},
]


@pytest.fixture()
def client():
    return TestClient(create_app())


def test_health_and_version(client):
    assert client.get("/health").json() == {"status": "ok"}
    payload = client.get("/version").json()
    assert payload == {"name": "ragtree", "version": __version__}


def test_retrieve_returns_ranked_evidence(client):
    response = client.post(
        "/retrieve", json={"query": "why did the pump fail", "documents": DOCS, "top_k": 2}
    )
    assert response.status_code == 200
    evidence = response.json()["evidence"]
    assert evidence and evidence[0]["document_id"] == "maint-001"


def test_retrieve_rejects_empty_documents(client):
    assert client.post("/retrieve", json={"query": "x", "documents": []}).status_code == 422


def test_run_qa_and_fetch_by_id(client):
    body = {
        "task": {"name": "question_answering", "question": "Why did the pump fail?"},
        "documents": DOCS,
        "llm": {"provider": "mock", "reply": "Seal wear [maint-001/maint-001-c0]."},
        "top_k": 2,
    }
    response = client.post("/runs", json=body)
    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["output"].startswith("Seal wear")
    assert payload["result"]["evidence"]

    run_id = payload["run_id"]
    assert client.get(f"/runs/{run_id}").json() == payload
    assert client.get("/runs/does-not-exist").status_code == 404


def test_run_relation_extraction_with_reference(client):
    gold = {"CAUSES": [["E1", "E2"]]}
    body = {
        "task": {"name": "relation_extraction", "relation_types": ["CAUSES"]},
        "documents": [{"id": "d1", "text": "A causes B."}],
        "llm": {"provider": "mock", "reply": '{"CAUSES": [["E1", "E2"]]}'},
        "reference": gold,
    }
    payload = client.post("/runs", json=body).json()
    assert payload["result"]["output"] == gold
    assert payload["result"]["metrics"]["f1"] == 1.0


def test_run_with_bad_task_returns_422(client):
    body = {"task": {"name": "no-such-task"}, "documents": DOCS}
    assert client.post("/runs", json=body).status_code == 422


def test_evaluate_endpoint(client):
    gold = {"CAUSES": [["E1", "E2"], ["E3", "E4"]]}
    pred = {"CAUSES": [["E1", "E2"]]}
    payload = client.post(
        "/evaluate", json={"predictions": pred, "reference": gold}
    ).json()
    assert payload["metrics"]["precision"] == 1.0
    assert payload["metrics"]["recall"] == 0.5
    assert payload["counts"] == {"tp": 1, "fp": 0, "fn": 1, "num_docs": 1}
