# ragtree/core/schemas.py
"""Stable data contracts shared by every RAGTree layer.

These schemas are the lingua franca between tasks, retrievers, generators,
evaluators and exporters. They stay small and dependency-light on purpose:
the BYOS promise rests on users being able to implement integrations against
these types in a few lines.

Design reference: BYOS architecture document, section 6.1.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Document",
    "Chunk",
    "EvidenceSpan",
    "RAGTask",
    "RAGResult",
    "RelationPrediction",
    "RunManifest",
    "EvaluationResult",
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class _Schema(BaseModel):
    """Base model: tolerant of extra fields so adapters can round-trip metadata."""

    model_config = ConfigDict(extra="allow")


class Document(_Schema):
    """Original input unit."""

    id: str
    text: str
    title: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(_Schema):
    """Indexed content unit derived from a document."""

    id: str
    document_id: str
    text: str
    index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceSpan(_Schema):
    """Grounding information attached to an output."""

    document_id: str
    text: str
    chunk_id: str | None = None
    score: float | None = None
    span: tuple[int, int] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGTask(_Schema):
    """Abstract task request handled by a pipeline.

    ``output_schema`` corresponds to the ``schema`` field in the design
    document; it is renamed here to avoid clashing with pydantic's own
    ``BaseModel.schema`` attribute.
    """

    task_type: str
    query: str | None = None
    output_schema: dict[str, Any] | None = None
    constraints: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGResult(_Schema):
    """Generic output object produced by a pipeline run."""

    task_type: str
    output: Any = None
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RelationPrediction(_Schema):
    """Relation-extraction output for one document.

    ``relations`` uses the historical ragtree format so that current
    benchmark outputs remain readable (compatibility contract, design doc
    section 11.1): ``{RELATION_TYPE: [[head_id, tail_id], ...]}``.
    """

    document_id: str
    relations: dict[str, list[list[str]]] = Field(default_factory=dict)
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    method: str | None = None
    model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunManifest(_Schema):
    """Reproducibility record for a pipeline run."""

    run_id: str
    config: dict[str, Any] = Field(default_factory=dict)
    versions: dict[str, str] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: datetime | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)


class EvaluationResult(_Schema):
    """Metric report produced by an evaluator."""

    metrics: dict[str, float] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    method: str | None = None
    dataset: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
