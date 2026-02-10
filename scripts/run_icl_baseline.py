# scripts/run_icl_baseline.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tqdm import tqdm  # only used during ICL example collection

from ragtree.processing.orchestrators.relations_runner import (
    PreparedContext,
    RunnerLLMSections,
    run_relation_experiment,
)
from ragtree.processing.rag.strategies.baseline_icl import ICLRelationStrategy


def _parse_cli_relation_types(arg: Optional[str]) -> Optional[List[str]]:
    if not arg:
        return None
    items = [x.strip() for x in arg.split(",")]
    items = [x for x in items if x]
    return items or None


def _parse_predict_types(arg: str) -> Sequence[str] | str:
    """
    '--icl-predict-types "dev,test"' -> ["dev", "test"]
    '--icl-predict-types "all"'      -> "all"
    """
    if not arg or arg == "all":
        return "all"
    items = [x.strip() for x in arg.split(",")]
    items = [x for x in items if x]
    return items or "all"


def prepare_icl_context(
    input_path: Path,
    cfg: Dict[str, Any],  # cfg is unused for now, but kept for future extensions
    *,
    icl_train_type: str,
    icl_train_num: int,
) -> PreparedContext:
    """
    First pass over the dataset to collect few-shot examples.

    We read the same input JSONL and keep docs where:

      - doc['type'] == icl_train_type
      - doc['relations'] exists and is non-empty

    We stop once we have collected `icl_train_num` examples.
    """
    few_shots: List[Dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as fin:
        for line in tqdm(
            fin,
            desc=f"Collecting ICL examples (type={icl_train_type})",
            unit="doc",
        ):
            line = line.strip()
            if not line:
                continue

            doc = json.loads(line)
            if doc.get("type") != icl_train_type:
                continue

            rels = doc.get("relations")
            if not isinstance(rels, dict) or not rels:
                continue

            few_shots.append(doc)
            if len(few_shots) >= icl_train_num:
                break

    print(f"[icl] Collected {len(few_shots)} few-shot examples of type '{icl_train_type}'.")

    # All examples are passed as a constant argument to predict_relations()
    return PreparedContext(
        strategy_kwargs={},
        predict_kwargs={"few_shots": few_shots},
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "In-context-learning (ICL) baseline for relation extraction. "
            "Uses a few-shot subset of the dataset as examples in the prompt."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to default.yaml (or similar). If omitted, load_config() should resolve it.",
    )

    parser.add_argument(
        "--dataset-key",
        required=True,
        help="Key under datasets.preprocessed in the config (e.g. 'maven_ere', 'docred_causal').",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help=(
            "LLM backend to use (e.g. 'vllm', 'ollama', 'openrouter'). "
            "If omitted, use llm.baseline.default_backend."
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model override for the chosen backend.",
    )

    parser.add_argument(
        "--relation-types",
        type=str,
        default=None,
        help=(
            "Optional comma-separated list of relation types to enforce "
            "(e.g. 'CAUSE,PRECONDITION' or 'P17,P27'). "
            "If omitted, relation types are inferred from doc['relations'] or "
            "fall back to a single default label."
        ),
    )

    parser.add_argument(
        "--output-format",
        choices=["full", "pred-only"],
        default="full",
        help=(
            "Control JSONL output structure:\n"
            "  - 'full': keep the full original document and add a 'pred_relations' field.\n"
            "  - 'pred-only': write only {document_id, pred_relations} per line."
        ),
    )

    # ICL-specific parameters
    parser.add_argument(
        "--icl-train-type",
        type=str,
        default="train",
        help=(
            "doc['type'] value used to select few-shot examples for ICL "
            "(e.g. 'train'). Default: 'train'."
        ),
    )

    parser.add_argument(
        "--icl-train-num",
        type=int,
        default=8,
        help="Number of few-shot examples to use from the training type. Default: 8.",
    )

    parser.add_argument(
        "--icl-predict-types",
        type=str,
        default="all",
        help=(
            "Comma-separated list of doc['type'] values to run prediction on "
            "(e.g. 'dev,test'). Use 'all' to process all types. Default: 'all'."
        ),
    )

    args = parser.parse_args()

    cli_rel_types = _parse_cli_relation_types(args.relation_types)
    predict_filter = _parse_predict_types(args.icl_predict_types)

    # For now, reuse the 'baseline' prompt section and system prompt.
    sections = RunnerLLMSections(
        llm_section="icl",          # logical method label used in output filename
        prompt_section="baseline",  # can change later if you create prompts.icl
        system_prompt_key="causal_relations",
    )

    def _prep(input_path: Path, cfg: Dict[str, Any]) -> PreparedContext:
        return prepare_icl_context(
            input_path,
            cfg,
            icl_train_type=args.icl_train_type,
            icl_train_num=args.icl_train_num,
        )

    run_relation_experiment(
        strategy_cls=ICLRelationStrategy,
        config_path=args.config,
        dataset_key=args.dataset_key,
        backend=args.backend,
        model=args.model,
        cli_relation_types=cli_rel_types,
        output_format=args.output_format,
        doc_type_filter=predict_filter,
        sections=sections,
        prepare_context_fn=_prep,
    )


if __name__ == "__main__":
    main()
