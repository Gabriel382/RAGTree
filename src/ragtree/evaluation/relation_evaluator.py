# ragtree/evaluation/relation_evaluator.py
"""Evaluator-protocol wrapper around the historical relation metrics.

Reuses ``ragtree.evaluation.relations.metrics`` (the code behind every
benchmark table) so old and new evaluation paths cannot drift apart.
"""

from __future__ import annotations

from typing import Any

from ragtree.core.schemas import EvaluationResult, RAGResult
from ragtree.evaluation.relations.metrics import RelationEvalAggregator

__all__ = ["RelationEvaluator"]


class RelationEvaluator:
    """Micro precision/recall/F1 over ``{TYPE: [[head, tail], ...]}`` dicts."""

    def __init__(self, ignore_labels: set[str] | None = None) -> None:
        self.ignore_labels = set(ignore_labels or ())

    def evaluate(self, result: RAGResult, reference: Any | None = None) -> EvaluationResult:
        pred = result.output if isinstance(result.output, dict) else {}
        gold = reference if isinstance(reference, dict) else {}

        aggregator = RelationEvalAggregator(ignore_labels=set(self.ignore_labels))
        aggregator.update(gold_relations=gold, pred_relations=pred)
        computed = aggregator.compute_metrics()

        return EvaluationResult(
            metrics={
                "precision": computed["micro"]["precision"],
                "recall": computed["micro"]["recall"],
                "f1": computed["micro"]["f1"],
            },
            counts={
                "tp": computed["counts"]["tp"],
                "fp": computed["counts"]["fp"],
                "fn": computed["counts"]["fn"],
                "num_docs": computed["counts"]["num_docs_seen"],
            },
            method="relation_micro",
            metadata={"per_relation": computed["per_relation"]},
        )
