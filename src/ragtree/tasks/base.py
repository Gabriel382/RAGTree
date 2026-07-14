# ragtree/tasks/base.py
"""Task layer: user-level RAG tasks over the stable core contracts.

A task knows three things: how to describe itself as a ``RAGTask``, how to
turn retrieved evidence into chat messages, and how to parse raw model output
into a structured result. Providers, stores and retrievers stay pluggable.
"""

from __future__ import annotations

from typing import Any

from ragtree.core.schemas import EvidenceSpan, RAGTask

__all__ = ["BaseTask"]


class BaseTask:
    """Base class for user-level tasks (design doc, sections 2.1 and 4.1)."""

    task_type: str = "generic"
    default_system_prompt: str = (
        "You are a careful assistant. Use only the provided evidence when "
        "evidence is given, and say so explicitly when the evidence is "
        "insufficient."
    )

    def __init__(
        self,
        query: str | None = None,
        output_schema: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.query = query
        self.output_schema = output_schema
        self.constraints = dict(constraints or {})
        self.metadata = dict(metadata or {})
        self.system_prompt = system_prompt or self.default_system_prompt

    # ------------------------------------------------------------------
    # Core-schema view
    # ------------------------------------------------------------------
    def to_ragtask(self) -> RAGTask:
        return RAGTask(
            task_type=self.task_type,
            query=self.query,
            output_schema=self.output_schema,
            constraints=self.constraints,
            metadata=self.metadata,
        )

    # ------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------
    @staticmethod
    def format_evidence(evidence: list[EvidenceSpan]) -> str:
        if not evidence:
            return "(no evidence retrieved)"
        lines = []
        for span in evidence:
            ref = span.document_id if span.chunk_id is None else f"{span.document_id}/{span.chunk_id}"
            lines.append(f"[{ref}] {span.text}")
        return "\n".join(lines)

    def instructions(self) -> str:
        """Task-specific user instructions; subclasses override."""
        return self.query or ""

    def build_messages(self, evidence: list[EvidenceSpan]) -> list[dict[str, str]]:
        user_content = (
            f"Evidence:\n{self.format_evidence(evidence)}\n\n{self.instructions()}"
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Output handling
    # ------------------------------------------------------------------
    def parse_output(self, text: str) -> Any:
        return text.strip()
