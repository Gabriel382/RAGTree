# scripts/run_growlrag_relations.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from ragtree.core.config import load_config
from ragtree.processing.orchestrators.relations_runner import (
    RunnerLLMSections,
    PreparedContext,
    run_relation_extraction,
)
from ragtree.processing.rag.strategies.growlrag_relations import GrowlRagRelationStrategy
from ragtree.ontologies.retrieval.subontology import SubOntologyRetriever


def prepare_growlrag_context(input_path: Path, cfg: Dict[str, Any]) -> PreparedContext:
    """
    Prepare strategy-level dependencies once:
      - SubOntologyRetriever with TTL path (cached internally)
    """
    growl_cfg = cfg.get("growlrag", {}) or {}

    ontology_key = growl_cfg.get("ontology_key", "docredontology")
    linking_method = growl_cfg.get("linking_method", "llm_embedding")

    # TTL path comes from cfg["ontology"][ontology_key]
    onto_paths = cfg.get("ontology", {}) or {}
    if ontology_key not in onto_paths:
        available = ", ".join(sorted(onto_paths.keys()))
        raise KeyError(f"Unknown ontology key '{ontology_key}'. Available: {available}")

    ttl_path = Path(onto_paths[ontology_key])

    retr_cfg = growl_cfg.get("retrieval", {}) or {}
    retriever = SubOntologyRetriever(
        ontology_key=ontology_key,
        ttl_path=ttl_path,
        include_unrestricted_properties=bool(retr_cfg.get("include_unrestricted_properties", True)),
        max_properties=retr_cfg.get("max_properties"),
        max_classes=retr_cfg.get("max_classes"),
        pick=retr_cfg.get("pick", "candidates"),
    )

    strategy_kwargs = {
        "retriever": retriever,
        "ontology_key": ontology_key,
        "linking_method": linking_method,
        "include_ttl": bool(growl_cfg.get("include_ttl", True)),
        "include_structured_fragment": bool(growl_cfg.get("include_structured_fragment", True)),
        "max_sentences_in_prompt": growl_cfg.get("max_sentences_in_prompt"),
    }

    predict_kwargs = {}  # nothing special yet
    return PreparedContext(strategy_kwargs=strategy_kwargs, predict_kwargs=predict_kwargs)


def main() -> None:
    p = argparse.ArgumentParser(description="Run GrOWL-RAG (ontology-guided RAG) relation extraction.")
    p.add_argument("--config", type=Path, default=None, help="Path to default.yaml (optional).")
    p.add_argument("--dataset-key", required=True, help="Key in cfg['datasets']['preprocessed'] (ideally a *_olink_*.jsonl).")
    p.add_argument("--backend", type=str, default=None, help="Override LLM backend.")
    p.add_argument("--model", type=str, default=None, help="Override LLM model.")
    p.add_argument("--relation-types", nargs="*", default=None, help="Optional fixed relation schema.")
    p.add_argument("--output-format", choices=["full", "pred-only"], default="full")
    p.add_argument("--doc-type-filter", default="all", help="Filter on doc['type'] (or 'all').")
    args = p.parse_args()

    # Tell the runner which config sections to use
    sections = RunnerLLMSections(
        llm_section="growlrag",
        prompt_section="growlrag",
        system_prompt_key="growlrag_docre",
    )

    run_relation_extraction(
        strategy_cls=GrowlRagRelationStrategy,
        config_path=args.config,
        dataset_key=args.dataset_key,
        backend=args.backend,
        model=args.model,
        cli_relation_types=args.relation_types,
        output_format=args.output_format,
        doc_type_filter=args.doc_type_filter,
        sections=sections,
        prepare_context_fn=prepare_growlrag_context,
    )


if __name__ == "__main__":
    main()
