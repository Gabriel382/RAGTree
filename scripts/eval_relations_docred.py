# scripts/eval_relations_docred.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from ragtree.core.config import load_config
from ragtree.evaluation.relations.runner import evaluate_relations


def _parse_ignore_labels(arg: Optional[str]) -> List[str]:
    if not arg:
        return []
    items = [x.strip() for x in arg.split(",")]
    return [x for x in items if x]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DocRED-Causal predictions using gold relations from a chosen file."
    )

    parser.add_argument("--config", type=Path, default=None)

    parser.add_argument("--dataset-key", type=str, default="docred_causal")
    parser.add_argument("--method", required=True)
    parser.add_argument("--backend", required=True)

    # NEW: explicit gold file path override
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=None,
        help="Path to JSONL containing GOLD 'relations' for DocRED. Overrides datasets.preprocessed lookup.",
    )

    parser.add_argument("--doc-type", type=str, default="all")
    parser.add_argument("--ignore-labels", type=str, default="null")

    args = parser.parse_args()
    cfg = load_config(args.config)

    processed_root = Path(cfg["paths"]["data_processed"])

    # gold path
    if args.gold_path is not None:
        gold_path = args.gold_path
    else:
        gold_path = Path(cfg["datasets"]["preprocessed"][args.dataset_key])

    if not gold_path.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_path}")

    # pred path
    pred_path = processed_root / f"{args.dataset_key}.{args.method}.{args.backend}.jsonl"
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    results_root = Path(cfg["paths"].get("results", "results"))
    results_dir = results_root / "relations" / args.dataset_key
    results_dir.mkdir(parents=True, exist_ok=True)

    doc_type_label = args.doc_type if args.doc_type else "all"
    metrics_path = results_dir / f"{args.method}.{args.backend}.{doc_type_label}.json"

    ignore_labels = _parse_ignore_labels(args.ignore_labels)

    print("[eval] dataset-key:", args.dataset_key)
    print("[eval] method:", args.method)
    print("[eval] backend:", args.backend)
    print("[eval] doc-type:", doc_type_label)
    print("[eval] gold:", gold_path)
    print("[eval] preds:", pred_path)
    print("[eval] ignore-labels:", ignore_labels)
    print("[eval] metrics-out:", metrics_path)

    metrics = evaluate_relations(
        gold_path=gold_path,
        pred_path=pred_path,
        doc_type_filter=doc_type_label,
        ignore_labels=ignore_labels,
    )

    micro = metrics.get("micro", {})
    print("\n=== Micro-level metrics ===")
    print(f"Precision: {micro.get('precision', 0.0):.4f}")
    print(f"Recall:    {micro.get('recall', 0.0):.4f}")
    print(f"F1:        {micro.get('f1', 0.0):.4f}")

    counts = metrics.get("counts", {})
    print("\n=== Counts ===")
    print("TP:", counts.get("tp"))
    print("FP:", counts.get("fp"))
    print("FN:", counts.get("fn"))
    print("num_docs_seen:", counts.get("num_docs_seen"))
    print("num_docs_eval:", counts.get("num_docs_eval"))
    print("num_docs_missing_gold:", counts.get("num_docs_missing_gold"))

    with metrics_path.open("w", encoding="utf-8") as fout:
        json.dump(metrics, fout, indent=2, ensure_ascii=False)

    print("\n[eval] Done.")


if __name__ == "__main__":
    main()
