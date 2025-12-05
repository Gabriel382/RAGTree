# scripts/run_cot_baseline.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any

from ragtree.processing.orchestrators.relations_runner import (
    RunnerLLMSections,
    run_relation_experiment,
)
from ragtree.processing.rag.strategies.chain_of_thought import (
    ChainOfThoughtRelationStrategy,
)
from ragtree.processing.orchestrators.relations_runner import PreparedContext

def _parse_cli_relation_types(arg: Optional[str]) -> Optional[List[str]]:
    if not arg:
        return None
    items = [x.strip() for x in arg.split(",")]
    return [x for x in items if x] or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chain-of-Thought baseline for relation extraction."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to default.yaml. If omitted, load_config() resolves it.",
    )

    parser.add_argument(
        "--dataset-key",
        required=True,
        help="Key under datasets.preprocessed (e.g. 'docred_causal').",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Override backend (vllm/ollama/openrouter).",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model override.",
    )

    parser.add_argument(
        "--relation-types",
        type=str,
        default=None,
        help="Comma-separated list of relation types to enforce.",
    )

    parser.add_argument(
        "--output-format",
        choices=["full", "pred-only"],
        default="full",
    )

    parser.add_argument(
        "--doc-type",
        type=str,
        default="all",
        help="Filter doc['type'] (train/dev/test/all).",
    )

    # 🔥 NEW FLAG
    parser.add_argument(
        "--print-cot",
        action="store_true",
        help="Print reasoning for each document. Default: off.",
    )

    args = parser.parse_args()

    cli_rel_types = _parse_cli_relation_types(args.relation_types)

    # CoT uses the SAME config section as baseline
    sections = RunnerLLMSections(
        llm_section="baseline",
        prompt_section="baseline",
        system_prompt_key="causal_relations",
    )

    def _prep(input_path: Path, cfg: Dict[str, Any]) -> PreparedContext:
        return PreparedContext(
            strategy_kwargs={},
            predict_kwargs={"print_cot": args.print_cot},
        )


    run_relation_experiment(
        strategy_cls=ChainOfThoughtRelationStrategy,
        config_path=args.config,
        dataset_key=args.dataset_key,
        backend=args.backend,
        model=args.model,
        cli_relation_types=cli_rel_types,
        output_format=args.output_format,
        doc_type_filter=args.doc_type,
        sections=sections,
        prepare_context_fn=lambda ip, cfg: _prep(ip, cfg),
    )


if __name__ == "__main__":
    main()
