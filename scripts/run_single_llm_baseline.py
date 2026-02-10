# scripts/run_single_llm_baseline.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Union

from ragtree.processing.rag.strategies.baseline_relations import (
    BaselineRelationStrategy,
)
from ragtree.processing.orchestrators.relations_runner import (
    RunnerLLMSections,
    run_relation_experiment,
)


def _parse_cli_relation_types(arg: Optional[str]) -> Optional[List[str]]:
    if not arg:
        return None
    items = [x.strip() for x in arg.split(",")]
    items = [x for x in items if x]
    return items or None


def _parse_doc_types(arg: str) -> Union[Sequence[str], str]:
    """
    '--doc-type "dev,test"' -> ["dev", "test"]
    '--doc-type "test"'     -> ["test"]
    '--doc-type "all"'      -> "all"
    """
    if not arg or arg == "all":
        return "all"
    items = [x.strip() for x in arg.split(",")]
    items = [x for x in items if x]
    return items or "all"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Single-LLM baseline for relation extraction, without RAG or ontology. "
            "Reads a preprocessed JSONL and writes predictions under 'pred_relations'."
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

    parser.add_argument(
        "--doc-type",
        type=str,
        default="all",
        help=(
            "Document type filter over doc['type'].\n"
            "  - 'all' (default): process all documents\n"
            "  - single value: e.g. 'test'\n"
            "  - comma-separated list: e.g. 'dev,test'"
        ),
    )

    args = parser.parse_args()

    cli_rel_types = _parse_cli_relation_types(args.relation_types)
    doc_type_filter = _parse_doc_types(args.doc_type)

    sections = RunnerLLMSections(
        llm_section="baseline",
        prompt_section="baseline",
        system_prompt_key="causal_relations",
    )

    run_relation_experiment(
        strategy_cls=BaselineRelationStrategy,
        config_path=args.config,
        dataset_key=args.dataset_key,
        backend=args.backend,
        model=args.model,
        cli_relation_types=cli_rel_types,
        output_format=args.output_format,
        doc_type_filter=doc_type_filter,
        sections=sections,
        prepare_context_fn=None,
    )


if __name__ == "__main__":
    main()
