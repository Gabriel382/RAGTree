# ragtree/apps/api/app.py
"""RAGTree FastAPI surface (design doc, section 8.2).

A stateless-by-default demo service: requests carry their own documents and
LLM spec, runs are kept in an in-memory registry for retrieval by id. It is
a surface over the core, not the core itself — production users compose
``RAGTreePipeline`` inside their own services.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ragtree import __version__
from ragtree.apps.runner import build_provider, build_task, build_retriever
from ragtree.core.errors import MissingDependencyError, RagtreeError
from ragtree.core.pipeline import RAGTreePipeline
from ragtree.evaluation.relation_evaluator import RelationEvaluator

__all__ = ["create_app"]


class DocumentIn(BaseModel):
    id: str
    text: str
    title: str | None = None


class LLMSpec(BaseModel):
    provider: str = "mock"
    model: str | None = None
    reply: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class RetrieveRequest(BaseModel):
    query: str
    documents: list[DocumentIn]
    top_k: int = 5


class TaskSpec(BaseModel):
    name: str
    question: str | None = None
    claim: str | None = None
    focus: str | None = None
    relation_types: list[str] | None = None
    require_evidence: bool = True


class RunRequest(BaseModel):
    task: TaskSpec
    documents: list[DocumentIn] = Field(default_factory=list)
    llm: LLMSpec = Field(default_factory=LLMSpec)
    top_k: int = 3
    reference: Any | None = None


class EvaluateRequest(BaseModel):
    predictions: dict[str, list[list[str]]]
    reference: dict[str, list[list[str]]]
    ignore_labels: list[str] = Field(default_factory=list)


def _docs_to_dicts(documents: list[DocumentIn]) -> list[dict[str, Any]]:
    return [doc.model_dump() for doc in documents]


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAGTree",
        version=__version__,
        description="Bring-your-own-stack Semantic RAG service surface.",
    )
    runs: dict[str, dict[str, Any]] = {}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/version")
    def version() -> dict[str, str]:
        return {"name": "ragtree", "version": __version__}

    @app.post("/retrieve")
    def retrieve(request: RetrieveRequest) -> dict[str, Any]:
        if not request.documents:
            raise HTTPException(status_code=422, detail="documents must not be empty")
        retriever = build_retriever(
            _docs_to_dicts(request.documents), {"top_k": request.top_k}
        )
        from ragtree.core.schemas import RAGTask

        spans = retriever.retrieve(RAGTask(task_type="retrieval", query=request.query))
        return {"evidence": [span.model_dump(mode="json") for span in spans]}

    @app.post("/runs")
    def create_run(request: RunRequest) -> dict[str, Any]:
        documents = _docs_to_dicts(request.documents)
        try:
            provider = build_provider(request.llm.model_dump())
            task = build_task(
                request.task.model_dump(exclude_none=True),
                document=documents[0] if documents else None,
            )
            retriever = (
                build_retriever(documents, {"top_k": request.top_k}) if documents else None
            )
            evaluator = (
                RelationEvaluator() if request.task.name == "relation_extraction" else None
            )
            pipeline = RAGTreePipeline(
                retriever=retriever, generator=provider, evaluator=evaluator
            )
            result = pipeline.run(task, reference=request.reference)
        except MissingDependencyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RagtreeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        run_id = uuid.uuid4().hex[:12]
        payload = {"run_id": run_id, "result": result.model_dump(mode="json")}
        runs[run_id] = payload
        return payload

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        if run_id not in runs:
            raise HTTPException(status_code=404, detail=f"Unknown run_id {run_id!r}")
        return runs[run_id]

    @app.post("/evaluate")
    def evaluate(request: EvaluateRequest) -> dict[str, Any]:
        from ragtree.core.schemas import RAGResult

        evaluator = RelationEvaluator(ignore_labels=set(request.ignore_labels))
        report = evaluator.evaluate(
            RAGResult(task_type="relation_extraction", output=request.predictions),
            reference=request.reference,
        )
        return report.model_dump(mode="json")

    return app
