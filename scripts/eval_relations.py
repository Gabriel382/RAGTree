# scripts/eval_relations.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ragtree.core.config import load_config
from ragtree.evaluation.relations.runner import evaluate_relations

def resolve_preprocessed_path(cfg: Dict[str, Any], dataset_key: str) -> Path:
    """
    Resolve a dataset input path from config.

    Priority:
      1) cfg["datasets"]["preprocessed"][dataset_key]
      2) cfg["paths"]["data_preprocessed"] / f"{dataset_key}.jsonl"  (fallback)

    This allows passing derived artifact names without adding them to default.yaml.
    """
    ds_pre = cfg.get("datasets", {}).get("preprocessed", {}) or {}
    if dataset_key in ds_pre:
        return Path(ds_pre[dataset_key])

    # Fallback: treat dataset_key as filename in data/preprocessed
    pre_root = Path(cfg["paths"]["data_preprocessed"])
    candidate = pre_root / f"{dataset_key}.jsonl"
    if candidate.exists():
        return candidate

    available = ", ".join(sorted(ds_pre.keys()))
    raise KeyError(
        f"Unknown dataset key '{dataset_key}'. "
        f"Available preprocessed datasets: {available}. "
        f"Also tried file: {candidate} (not found)."
    )


def _parse_ignore_labels(arg: Optional[str]) -> List[str]:
    """
    Parse a comma-separated list of labels to ignore (e.g. "null,NA").
    """
    if not arg:
        return []
    items = [x.strip() for x in arg.split(",")]
    return [x for x in items if x]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate relation extraction predictions (gold vs pred_relations) "
            "and compute micro / per-relation F1."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to default.yaml. If omitted, load_config() should resolve it.",
    )

    parser.add_argument(
        "--dataset-key",
        required=True,
        help="Key under datasets.preprocessed in the config (e.g. 'eventstoryline').",
    )

    parser.add_argument(
        "--method",
        required=True,
        help=(
            "Method name used in the predictions filename. "
            "For example, 'baseline', 'icl', 'cot', 'ontorag', etc. "
            "Predictions are expected at: "
            "paths.data_processed / f'{dataset}.{method}.{backend}.jsonl'"
        ),
    )

    parser.add_argument(
        "--backend",
        required=True,
        help="Backend name used in the predictions filename (e.g. 'vllm', 'ollama').",
    )

    parser.add_argument(
        "--doc-type",
        type=str,
        default="all",
        help="Optional doc['type'] filter (e.g. 'dev', 'test', 'train'). Default: 'all'.",
    )

    parser.add_argument(
        "--ignore-labels",
        type=str,
        default="null",
        help=(
            "Comma-separated list of relation labels to ignore when scoring "
            "(e.g. 'null,NA'). Default: 'null'. Use empty string to disable."
        ),
    )

    args = parser.parse_args()

    # 1) Load config
    cfg = load_config(args.config)

    # 2) Resolve gold path (supports base dataset keys or derived artifact filenames)
    gold_path = resolve_preprocessed_path(cfg, args.dataset_key)

    # 3) Resolve predictions path from paths.data_processed
    try:
        processed_root = Path(cfg["paths"]["data_processed"])
    except KeyError as e:
        raise KeyError(f"Config missing 'paths.data_processed' section: {e}")

    pred_filename = f"{args.dataset_key}.{args.method}.{args.backend}.jsonl"
    pred_path = processed_root / pred_filename

    if not pred_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {pred_path}. "
            f"Expected pattern: <dataset>.<method>.<backend>.jsonl"
        )

    # 4) Results folder for relations
    results_root = Path(cfg["paths"].get("results", "results"))
    results_dir = results_root / "relations" / args.dataset_key
    results_dir.mkdir(parents=True, exist_ok=True)

    # Metrics output file
    doc_type_label = args.doc_type if args.doc_type else "all"
    metrics_filename = f"{args.method}.{args.backend}.{doc_type_label}.json"
    metrics_path = results_dir / metrics_filename

    ignore_labels = _parse_ignore_labels(args.ignore_labels)

    print("[eval] dataset-key:", args.dataset_key)
    print("[eval] method:", args.method)
    print("[eval] backend:", args.backend)
    print("[eval] doc-type:", doc_type_label)
    print("[eval] gold:", gold_path)
    print("[eval] preds:", pred_path)
    print("[eval] ignore-labels:", ignore_labels)
    print("[eval] metrics-out:", metrics_path)

    # 5) Run evaluation
    metrics = evaluate_relations(
        gold_path=gold_path,
        pred_path=pred_path,
        doc_type_filter=doc_type_label,
        ignore_labels=ignore_labels,
    )

    # 6) Print micro metrics to stdout
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

    # 7) Save full metrics JSON
    with metrics_path.open("w", encoding="utf-8") as fout:
        json.dump(metrics, fout, indent=2, ensure_ascii=False)

    print("\n[eval] Done.")


if __name__ == "__main__":
    main()
