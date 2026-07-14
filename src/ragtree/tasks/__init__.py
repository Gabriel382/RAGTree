# ragtree/tasks/__init__.py
"""User-level RAG tasks (design doc, section 2.1)."""

from .base import BaseTask
from .claim_verification import ClaimVerificationTask
from .question_answering import QuestionAnsweringTask
from .relation_extraction import RelationExtractionTask, results_from_strategy
from .summarization import SummarizationTask

__all__ = [
    "BaseTask",
    "QuestionAnsweringTask",
    "RelationExtractionTask",
    "SummarizationTask",
    "ClaimVerificationTask",
    "results_from_strategy",
]
