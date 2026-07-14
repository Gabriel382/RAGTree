"""Reference implementations used by the contract suite.

Sprint 2 note: these classes graduated into ``ragtree.integrations`` as the
real no-dependency adapters; the aliases below keep the contract suite
exercising the shipped package code rather than test-local copies.
"""

from __future__ import annotations

from typing import Any

from ragtree.core.schemas import EvaluationResult, RAGResult
from ragtree.integrations.embedders import HashingEmbedder
from ragtree.integrations.exporters import JsonExporter
from ragtree.integrations.graphstores import LocalGraphStore
from ragtree.integrations.llms import MockLLMProvider
from ragtree.integrations.vectorstores import InMemoryVectorStore
from ragtree.retrieval import DenseRetriever

__all__ = [
    "FakeLLMProvider",
    "HashingEmbedder",
    "InMemoryVectorStore",
    "SimpleRetriever",
    "InMemoryGraphStore",
    "ExactMatchEvaluator",
    "JsonExporter",
]

# Package adapters exposed under their historical fake names.
FakeLLMProvider = MockLLMProvider
SimpleRetriever = DenseRetriever
InMemoryGraphStore = LocalGraphStore


class ExactMatchEvaluator:
    """Trivial evaluator kept test-local: exact string match on output."""

    def evaluate(self, result: RAGResult, reference: Any | None = None) -> EvaluationResult:
        match = float(reference is not None and result.output == reference)
        return EvaluationResult(
            metrics={"exact_match": match}, counts={"n": 1}, method="exact_match"
        )
