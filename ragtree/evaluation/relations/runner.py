# ragtree/evaluation/relations/runner.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set

from .io import extract_doc_id, load_gold_relations_by_id
from .metrics import RelationEvalAggregator


def evaluate_relations(
    *,
    gold_path: Path,
    pred_path: Path,
    doc_type_filter: str = "all",
    ignore_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate relation extraction predictions against gold annotations.

    Assumptions:
      - pred_path: JSONL where each line is a document-level object that has at least:
          - 'document_id' or 'id'
          - 'pred_relations' (dict) OR we treat missing as {}
          - optionally 'relations' (gold in same file)
          - optionally 'type' (e.g. 'train', 'dev', 'test', ...)
      - gold_path: JSONL with 'relations' for each document_id, used as fallback
        if 'relations' is not present in the prediction file.

    Logic:
      - Iterate over pred_path line by line.
      - Filter by doc_type_filter if not "all".
      - For each doc:
          * gold_relations = doc["relations"] if present
            else gold_by_id[doc_id] from gold_path
          * pred_relations = doc.get("pred_relations", {})
      - Update micro and per-label counts using RelationEvalAggregator.
    """
    ignore_set: Set[str] = set(ignore_labels or [])

    # Load gold relations once, indexed by doc_id.
    gold_by_id = load_gold_relations_by_id(gold_path)

    aggregator = RelationEvalAggregator(ignore_labels=ignore_set)

    num_docs_eval = 0
    num_docs_missing_gold = 0

    with pred_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            doc_id = extract_doc_id(doc)
            if not doc_id:
                continue

            # Optional doc-type filtering
            if doc_type_filter and doc_type_filter != "all":
                doc_type = doc.get("type")
                if doc_type != doc_type_filter:
                    continue

            # Predictions
            pred_rel = doc.get("pred_relations")
            if not isinstance(pred_rel, dict):
                pred_rel = {}

            # Gold relations: prefer inline, fallback to preprocessed mapping
            gold_rel = doc.get("relations")
            if not isinstance(gold_rel, dict):
                gold_rel = gold_by_id.get(doc_id)

            if gold_rel is None:
                num_docs_missing_gold += 1
                continue

            aggregator.update(
                gold_relations=gold_rel,
                pred_relations=pred_rel,
            )
            num_docs_eval += 1

    metrics = aggregator.compute_metrics()
    # Enrich with additional counts
    metrics.setdefault("counts", {})
    metrics["counts"]["num_docs_eval"] = num_docs_eval
    metrics["counts"]["num_docs_missing_gold"] = num_docs_missing_gold

    return metrics
