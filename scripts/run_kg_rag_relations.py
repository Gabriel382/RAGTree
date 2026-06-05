# scripts/run_kg_rag_relations.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from ragtree.core.config import load_config
from ragtree.processing.orchestrators.relations_runner import (
    RunnerLLMSections,
    PreparedContext,
    run_relation_experiment,
)
from ragtree.processing.kg_rag.kg_loader import load_local_graphstore
from ragtree.processing.kg_rag.kg_retriever import KGRetriever
from ragtree.processing.rag.strategies.kg_rag_relations import KGRagRelationStrategy


def prepare_kg_rag_context(_input_path: Path, cfg: Dict[str, Any], *, kg_path: Path, max_hops: int, max_triples: int) -> PreparedContext:
    gs = load_local_graphstore(kg_path)

    retriever = KGRetriever(
        gs,
        max_hops=max_hops,
        max_triples=max_triples,
        allowed_relations=None,  # optionally restrict to schema
    )

    strategy_kwargs = {
        "retriever": retriever,
        "max_sentences_in_prompt": (cfg.get("kg_rag", {}) or {}).get("max_sentences_in_prompt"),
    }
    return PreparedContext(strategy_kwargs=strategy_kwargs, predict_kwargs={})


def main() -> None:
    p = argparse.ArgumentParser(description="Run KG-RAG relation extraction (local KG).")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--dataset-key", required=True)
    p.add_argument("--backend", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--output-format", choices=["full", "pred-only"], default="full")
    p.add_argument("--doc-type-filter", default="all", help="Filter doc['type'] or 'all'.")
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)

    # KG args
    p.add_argument("--kg-path", type=Path, required=True, help="Path to local KG JSON produced by build_kg_from_preprocessed.py")
    p.add_argument("--kg-max-hops", type=int, default=1)
    p.add_argument("--kg-max-triples", type=int, default=200)

    args = p.parse_args()

    sections = RunnerLLMSections(
        llm_section="kg_rag",
        prompt_section="kg_rag",
        system_prompt_key="kg_rag_docre",
    )

    def _prep(input_path: Path, cfg: Dict[str, Any]) -> PreparedContext:
        return prepare_kg_rag_context(
            input_path,
            cfg,
            kg_path=args.kg_path,
            max_hops=args.kg_max_hops,
            max_triples=args.kg_max_triples,
        )

    run_relation_experiment(
        strategy_cls=KGRagRelationStrategy,
        config_path=args.config,
        dataset_key=args.dataset_key,
        backend=args.backend,
        model=args.model,
        cli_relation_types=None,
        output_format=args.output_format,
        doc_type_filter=args.doc_type_filter,
        sections=sections,
        prepare_context_fn=_prep,
        skip=args.skip,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
