# ragtree/core/pipeline.py
"""Pipeline lifecycle: retrieve -> generate -> parse -> evaluate -> export.

The pipeline only speaks protocols and schemas (design doc, section 17.1).
Any component can be swapped: that is the BYOS promise.
"""

from __future__ import annotations

from typing import Any

from .errors import RagtreeError
from .protocols import Evaluator, Exporter, LLMProvider, Retriever
from .schemas import Document, EvidenceSpan, RAGResult, RAGTask

__all__ = ["RAGTreePipeline"]


def _default_messages(task: RAGTask, evidence: list[EvidenceSpan]) -> list[dict[str, str]]:
    evidence_text = "\n".join(
        f"[{span.document_id}{'/' + span.chunk_id if span.chunk_id else ''}] {span.text}"
        for span in evidence
    ) or "(no evidence retrieved)"
    return [
        {"role": "system", "content": "Use only the provided evidence."},
        {"role": "user", "content": f"Evidence:\n{evidence_text}\n\n{task.query or ''}"},
    ]


class RAGTreePipeline:
    """Composable pipeline over the core protocols.

    ``run`` accepts either a task object from ``ragtree.tasks`` (anything
    with ``to_ragtask``/``build_messages``/``parse_output``) or a plain
    ``RAGTask`` schema, in which case a default prompt builder is used.
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        generator: LLMProvider | None = None,
        evaluator: Evaluator | None = None,
        exporter: Exporter | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.evaluator = evaluator
        self.exporter = exporter

    def run(
        self,
        task: Any,
        documents: list[Document] | None = None,
        reference: Any | None = None,
        output_path: str | None = None,
        **generate_kwargs: Any,
    ) -> RAGResult:
        if self.generator is None:
            raise RagtreeError("RAGTreePipeline.run requires a generator (LLMProvider).")

        rag_task: RAGTask = task.to_ragtask() if hasattr(task, "to_ragtask") else task
        if not isinstance(rag_task, RAGTask):
            raise RagtreeError(
                "task must be a ragtree.tasks task or a ragtree.core.schemas.RAGTask, "
                f"got {type(task).__name__}"
            )

        evidence: list[EvidenceSpan] = []
        if self.retriever is not None:
            evidence = list(self.retriever.retrieve(rag_task, documents))

        if hasattr(task, "build_messages"):
            messages = task.build_messages(evidence)
        else:
            messages = _default_messages(rag_task, evidence)

        raw_output = self.generator.complete(messages, **generate_kwargs)
        output = task.parse_output(raw_output) if hasattr(task, "parse_output") else raw_output

        result = RAGResult(
            task_type=rag_task.task_type,
            output=output,
            evidence=evidence,
            metadata={"raw_output": raw_output},
        )

        if self.evaluator is not None:
            report = self.evaluator.evaluate(result, reference)
            result.metrics.update(report.metrics)
            result.artifacts["evaluation"] = report.model_dump()

        if self.exporter is not None and output_path is not None:
            self.exporter.export(result, output_path)
            result.artifacts["exported_to"] = output_path

        return result
