# ragtree/tasks/question_answering.py
"""Question answering over retrieved evidence."""

from __future__ import annotations

from .base import BaseTask

__all__ = ["QuestionAnsweringTask"]


class QuestionAnsweringTask(BaseTask):
    task_type = "question_answering"
    default_system_prompt = (
        "You answer questions using ONLY the provided evidence. Cite the "
        "evidence you used with its bracketed reference, e.g. [doc1/c2]. "
        "If the evidence is insufficient, answer exactly: INSUFFICIENT EVIDENCE."
    )

    def __init__(self, question: str, require_evidence: bool = True, **kwargs) -> None:
        super().__init__(query=question, **kwargs)
        self.question = question
        self.require_evidence = require_evidence
        self.constraints.setdefault("require_evidence", require_evidence)

    def instructions(self) -> str:
        return f"Question: {self.question}\nAnswer concisely with citations."
