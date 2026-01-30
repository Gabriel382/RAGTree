# scripts/run_growlrag.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ragtree.processing.orchestrators.relations_runner import (
    PreparedContext,
    RunnerLLMSections,
    run_relation_experiment,
)
from ragtree.processing.rag.strategies.growl_relations import GrowlRelationStrategy


def _parse_cli_relation_types(arg: Optional[str]) -> Optional[List[str]]:
    """
    Parse --relation-types "REL1,REL2" into ["REL1", "REL2"].

    If arg is None / empty, return None (meaning: infer from doc['relations']).
    """
    if not arg:
        return None
    items = [x.strip() for x in arg.split(",")]
    items = [x for x in items if x]
    return items or None


def prepare_growl_context(
    input_path: Path,
    cfg: Dict[str, Any],
) -> PreparedContext:
    """
    For now, GrOWL does not need extra global context: all ontology info
    comes from `doc['ontology_links']`, which must already be present in
    the input JSONL (via run_ontology_linking.py run in-place or to a
    dedicated processed file).

    This is kept as a hook so we can later extend it (e.g., load rdflib
    graphs, extra KG info, etc.) without changing the runner.
    """
    return PreparedContext(
        strategy_kwargs={},   # no extra kwargs for the strategy
        predict_kwargs={},    # no extra kwargs for predict_relations
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "GrOWL / Ontology-guided RAG baseline for relation extraction.\n"
            "Uses precomputed ontology_links in the input JSONL to enrich "
            "the LLM prompt with ontology context."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to config YAML (e.g. config/default.yaml). "
            "If omitted, load_config() inside the orchestrator should resolve "
            "the default."
        ),
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
            "If omitted, use llm.growl.default_backend."
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
            "If omitted, relation types are inferred from doc['relations'] "
            "or fall back to a single default label."
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
            "Filter documents by doc['type'] (e.g. 'train', 'dev', 'test'). "
            "Use 'all' to process every type. Default: 'all'."
        ),
    )

    args = parser.parse_args()

    cli_rel_types = _parse_cli_relation_types(args.relation_types)

    # We reuse the same prompt_section and system_prompt_key as baseline;
    # only the llm_section changes, so you can configure llm.growl separately
    # in your YAML (backend/model/temperature, etc.).
    sections = RunnerLLMSections(
        llm_section="growl",
        prompt_section="baseline",
        system_prompt_key="causal_relations",
    )

    def _prep(input_path: Path, cfg: Dict[str, Any]) -> PreparedContext:
        return prepare_growl_context(input_path, cfg)

    run_relation_experiment(
        strategy_cls=GrowlRelationStrategy,
        config_path=args.config,
        dataset_key=args.dataset_key,
        backend=args.backend,
        model=args.model,
        cli_relation_types=cli_rel_types,
        output_format=args.output_format,
        doc_type=args.doc_type,
        sections=sections,
        prepare_context_fn=_prep,
    )


if __name__ == "__main__":
    main()
