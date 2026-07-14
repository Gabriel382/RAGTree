# ragtree/evaluation/relations/metrics.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


Triple = Tuple[str, str, str]  # (relation_type, head_id, tail_id)


def relations_to_triples(
    rel_dict: Dict[str, Any],
    *,
    ignore_labels: Optional[Set[str]] = None,
) -> Set[Triple]:
    """
    Convert a dict of relation_type -> [[head, tail], ...]
    into a set of (relation_type, head, tail) triples.

    - rel_dict is expected to be doc["relations"] or doc["pred_relations"].
    - ignore_labels: relation types to skip (e.g. {"null"}).
    """
    triples: Set[Triple] = set()
    if not isinstance(rel_dict, dict):
        return triples

    ignore_labels = ignore_labels or set()

    for rel_type, pairs in rel_dict.items():
        if rel_type in ignore_labels:
            continue
        if not isinstance(pairs, list):
            continue
        for pair in pairs:
            if (
                isinstance(pair, (list, tuple))
                and len(pair) == 2
                and isinstance(pair[0], str)
                and isinstance(pair[1], str)
            ):
                triples.add((rel_type, pair[0], pair[1]))
    return triples


@dataclass
class LabelCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0


@dataclass
class RelationEvalAggregator:
    """
    Accumulates TP/FP/FN counts across documents and computes micro / per-label metrics.
    """

    ignore_labels: Set[str] = field(default_factory=set)

    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0

    per_label: Dict[str, LabelCounts] = field(default_factory=dict)

    num_docs_seen: int = 0

    def _get_label_counts(self, label: str) -> LabelCounts:
        if label not in self.per_label:
            self.per_label[label] = LabelCounts()
        return self.per_label[label]

    def update(
        self,
        *,
        gold_relations: Dict[str, Any],
        pred_relations: Dict[str, Any],
    ) -> None:
        """
        Update counts given gold and predicted relation dicts for a single document.
        """
        self.num_docs_seen += 1

        gold_triples = relations_to_triples(gold_relations, ignore_labels=self.ignore_labels)
        pred_triples = relations_to_triples(pred_relations, ignore_labels=self.ignore_labels)

        # Micro-level sets
        inter = gold_triples & pred_triples
        gold_only = gold_triples - pred_triples
        pred_only = pred_triples - gold_triples

        tp = len(inter)
        fn = len(gold_only)
        fp = len(pred_only)

        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn

        # Per-label counts
        for rel_type, h, t in inter:
            lc = self._get_label_counts(rel_type)
            lc.tp += 1
        for rel_type, h, t in pred_only:
            lc = self._get_label_counts(rel_type)
            lc.fp += 1
        for rel_type, h, t in gold_only:
            lc = self._get_label_counts(rel_type)
            lc.fn += 1

    @staticmethod
    def _safe_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
        """
        Compute precision/recall/F1 with safeguards for zero denominators.
        """
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Return a dict with:
        - 'micro': precision/recall/f1 over all relations
        - 'per_relation': metrics per relation type
        - 'counts': TP/FP/FN and num_docs_seen
        """
        micro = self._safe_prf(self.total_tp, self.total_fp, self.total_fn)

        per_rel_metrics: Dict[str, Any] = {}
        for rel_type, counts in sorted(self.per_label.items()):
            m = self._safe_prf(counts.tp, counts.fp, counts.fn)
            m.update(
                {
                    "tp": counts.tp,
                    "fp": counts.fp,
                    "fn": counts.fn,
                    "support_gold": counts.tp + counts.fn,
                    "support_pred": counts.tp + counts.fp,
                }
            )
            per_rel_metrics[rel_type] = m

        return {
            "micro": micro,
            "per_relation": per_rel_metrics,
            "counts": {
                "tp": self.total_tp,
                "fp": self.total_fp,
                "fn": self.total_fn,
                "num_docs_seen": self.num_docs_seen,
            },
        }
