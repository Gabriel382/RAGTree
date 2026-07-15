# ragtree/apps/runner.py
"""Config- and spec-driven pipeline assembly shared by CLI, API and UI.

Turns declarative specs (YAML config files or JSON request bodies) into
providers, retrievers, tasks and full runs, always through the core
protocols — the BYOS wiring layer for the application surfaces.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import yaml

from ragtree import __version__
from ragtree.core.errors import ConfigurationError
from ragtree.core.pipeline import RAGTreePipeline
from ragtree.core.schemas import Chunk, RunManifest
from ragtree.evaluation.relation_evaluator import RelationEvaluator
from ragtree.integrations.exporters import JsonlExporter
from ragtree.integrations.llms import MockLLMProvider
from ragtree.integrations.vectorstores import InMemoryVectorStore
from ragtree.retrieval import DenseRetriever
from ragtree.tasks import (
    BaseTask,
    ClaimVerificationTask,
    QuestionAnsweringTask,
    RelationExtractionTask,
    SummarizationTask,
)

__all__ = [
    "build_provider",
    "build_task",
    "build_retriever",
    "load_documents_jsonl",
    "documents_to_chunks",
    "run_from_config",
]

_TASK_ALIASES = {
    "qa": "question_answering",
    "question_answering": "question_answering",
    "relation_extraction": "relation_extraction",
    "re": "relation_extraction",
    "summarization": "summarization",
    "claim_verification": "claim_verification",
}


def build_provider(spec: dict[str, Any] | None):
    """Build an LLMProvider from a spec like ``{"provider": "mock", ...}``.

    Providers: mock (default), litellm, ollama, openrouter, vllm. Optional
    stacks raise MissingDependencyError with the extra to install.
    """
    spec = dict(spec or {})
    name = str(spec.get("provider") or "mock").lower()
    model = spec.get("model")
    params = dict(spec.get("params") or {})

    if name == "mock":
        return MockLLMProvider(reply=spec.get("reply"))
    if name == "litellm":
        if not model:
            raise ConfigurationError("llm.model is required for the litellm provider")
        from ragtree.integrations.llms import LiteLLMProvider

        return LiteLLMProvider(model=model, **params)
    if name == "ollama":
        from ragtree.integrations.llms import OllamaProvider

        return OllamaProvider(chat_model=model)
    if name == "openrouter":
        from ragtree.integrations.llms import OpenRouterProvider

        return OpenRouterProvider(chat_model=model)
    if name == "vllm":
        from ragtree.integrations.llms import VLLMProvider

        return VLLMProvider(chat_model=model)
    raise ConfigurationError(
        f"Unknown llm.provider {name!r}. Choose from: mock, litellm, ollama, openrouter, vllm."
    )


def build_task(spec: dict[str, Any], document: dict[str, Any] | None = None) -> BaseTask:
    """Build a task object from a spec like ``{"name": "question_answering", ...}``."""
    spec = dict(spec or {})
    raw_name = str(spec.get("name") or spec.get("task_type") or "").lower()
    name = _TASK_ALIASES.get(raw_name)
    if name is None:
        raise ConfigurationError(
            f"Unknown task {raw_name!r}. Choose from: question_answering, "
            "relation_extraction, summarization, claim_verification."
        )

    if name == "question_answering":
        question = spec.get("question") or spec.get("query")
        if not question:
            raise ConfigurationError("task.question is required for question_answering")
        return QuestionAnsweringTask(
            str(question), require_evidence=bool(spec.get("require_evidence", True))
        )
    if name == "summarization":
        return SummarizationTask(
            focus=spec.get("focus"), max_sentences=int(spec.get("max_sentences", 5))
        )
    if name == "claim_verification":
        claim = spec.get("claim") or spec.get("query")
        if not claim:
            raise ConfigurationError("task.claim is required for claim_verification")
        return ClaimVerificationTask(str(claim))

    # relation_extraction
    relation_types = spec.get("relation_types")
    if not relation_types and isinstance(document, dict):
        relation_types = list((document.get("relations") or {}).keys())
    if not relation_types:
        raise ConfigurationError(
            "task.relation_types is required for relation_extraction when documents "
            "carry no gold relations"
        )
    return RelationExtractionTask(list(relation_types), document=document)


def load_documents_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load documents from JSONL with ``id``/``document_id`` and ``text`` fields."""
    doc_path = Path(path)
    if not doc_path.is_file():
        raise ConfigurationError(f"Documents file not found: {doc_path}")
    documents: list[dict[str, Any]] = []
    for line in doc_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        doc = json.loads(line)
        doc_id = doc.get("document_id") or doc.get("id")
        if not doc_id or not doc.get("text"):
            raise ConfigurationError(
                f"Each document needs an id/document_id and text: got {sorted(doc.keys())}"
            )
        documents.append(doc)
    return documents


def documents_to_chunks(documents: list[dict[str, Any]]) -> list[Chunk]:
    chunks = []
    for doc in documents:
        doc_id = str(doc.get("document_id") or doc.get("id"))
        chunks.append(Chunk(id=f"{doc_id}-c0", document_id=doc_id, text=str(doc["text"])))
    return chunks


def build_retriever(documents: list[dict[str, Any]], spec: dict[str, Any] | None):
    spec = dict(spec or {})
    provider = str(spec.get("provider") or "in_memory").lower()
    if provider != "in_memory":
        raise ConfigurationError(
            f"Unknown retriever.provider {provider!r}; this runner ships 'in_memory'. "
            "Bring your own stack by constructing RAGTreePipeline in Python."
        )
    store = InMemoryVectorStore()
    store.add_chunks(documents_to_chunks(documents))
    return DenseRetriever(store, top_k=int(spec.get("top_k", 3)))


def _micro_prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def run_from_config(config_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Execute a declarative run config; returns a summary dict.

    See ``examples/configs/semantic_rag_demo.yaml`` and
    ``examples/configs/relation_extraction_benchmark.yaml``.
    """
    cfg_path = Path(config_path)
    if not cfg_path.is_file():
        raise ConfigurationError(f"Config file not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    docs_spec = cfg.get("documents") or {}
    docs_path = Path(str(docs_spec.get("path", "")))
    if not docs_path.is_absolute():
        docs_path = (cfg_path.parent / docs_path).resolve()
    documents = load_documents_jsonl(docs_path)

    out_dir = Path(output_dir or (cfg.get("outputs") or {}).get("directory") or "outputs/run")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    results_path.write_text("", encoding="utf-8")

    provider = build_provider(cfg.get("llm"))
    task_spec = dict(cfg.get("task") or {})
    task_name = _TASK_ALIASES.get(str(task_spec.get("name", "")).lower())
    run_id = str((cfg.get("project") or {}).get("run_id") or uuid.uuid4().hex[:12])
    manifest = RunManifest(
        run_id=run_id,
        config=cfg,
        versions={"ragtree": __version__},
        artifacts={"results": str(results_path)},
    )
    exporter = JsonlExporter()

    metrics: dict[str, Any] = {}
    n_results = 0

    if task_name == "relation_extraction":
        evaluator = RelationEvaluator(
            ignore_labels=set(task_spec.get("ignore_labels") or ())
        )
        totals = {"tp": 0, "fp": 0, "fn": 0}
        evaluated = 0
        for doc in documents:
            task = build_task(task_spec, document=doc)
            pipeline = RAGTreePipeline(generator=provider, evaluator=evaluator)
            reference = doc.get("relations") if isinstance(doc.get("relations"), dict) else None
            result = pipeline.run(task, reference=reference)
            result.metadata["document_id"] = task.document_id
            exporter.export(result, str(results_path))
            n_results += 1
            evaluation = result.artifacts.get("evaluation") or {}
            counts = evaluation.get("counts") or {}
            if reference is not None and counts:
                for key in totals:
                    totals[key] += int(counts.get(key, 0))
                evaluated += 1
        if evaluated:
            metrics = {**_micro_prf(**totals), **totals, "num_docs": evaluated}
    else:
        retriever = build_retriever(documents, cfg.get("retriever"))
        task = build_task(task_spec)
        pipeline = RAGTreePipeline(retriever=retriever, generator=provider)
        result = pipeline.run(task)
        exporter.export(result, str(results_path))
        n_results = 1
        metrics = {"n_evidence": len(result.evidence)}

    from datetime import datetime, timezone

    manifest.finished_at = datetime.now(timezone.utc)
    (out_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "run_id": run_id,
        "task": task_name,
        "n_documents": len(documents),
        "n_results": n_results,
        "output_dir": str(out_dir),
        "metrics": metrics,
    }
