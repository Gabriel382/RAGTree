# ragtree/tasks/summarization.py
"""Faithful summarization over one document or a retrieved set."""

from __future__ import annotations

from .base import BaseTask

__all__ = ["SummarizationTask"]


class SummarizationTask(BaseTask):
    task_type = "summarization"
    default_system_prompt = (
        "You write faithful summaries. Use only facts present in the "
        "evidence; never introduce outside information. Cite evidence "
        "references in brackets for each key claim."
    )

    def __init__(self, focus: str | None = None, max_sentences: int = 5, **kwargs) -> None:
        super().__init__(query=focus, **kwargs)
        self.focus = focus
        self.max_sentences = max_sentences

    def instructions(self) -> str:
        base = f"Summarize the evidence in at most {self.max_sentences} sentences."
        if self.focus:
            base += f" Focus on: {self.focus}."
        return base
