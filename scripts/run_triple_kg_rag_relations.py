# scripts/run_triple_kg_rag_relations.py
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
from ragtree.processing.kg_rag.triple_kg_retriever import TripleKGRetriever, TripleKGRetrieverParams
from ragtree.processing.rag.strategies.triple_kg_rag_relations import TripleKGRagRelationStrategy


def _autopick_kg_file(cfg: Dict[str, Any], dataset_key: str) -> Path:
    kg_root = Path((cfg.get("paths", {}) or {}).get("kg", "data/kg"))
    if not kg_root.exists():
        raise FileNotFoundError(f"KG root folder not found: {kg_root}")

    cands = sorted(kg_root.glob(f"{dataset_key}__*__kg.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(
            f"No KG artifact found for dataset_key='{dataset_key}' in {kg_root}. "
            f"Expected something like: {dataset_key}__types=...__kg.json"
        )
    return cands[0]


def prepare_triple_kg_rag_context(
    _input_path: Path,
    cfg: Dict[str, Any],
    *,
    kg_path: Path,
    max_hops: int,
    max_triples: int,
    include_in_edges: bool,
    max_sentences_in_prompt: Optional[int],
    max_triples_in_text: Optional[int],
) -> PreparedContext:
    gs = load_local_graphstore(kg_path)

    retriever = TripleKGRetriever(
        gs,
        params=TripleKGRetrieverParams(
            max_hops=max_hops,
            max_triples=max_triples,
            include_in_edges=include_in_edges,
            scoring="token_overlap",
        ),
    )

    strategy_kwargs = {
        "retriever": retriever,
        "max_sentences_in_prompt": max_sentences_in_prompt,
        "max_triples_in_text": max_triples_in_text,
    }
    return PreparedContext(strategy_kwargs=strategy_kwargs, predict_kwargs={})


def main() -> None:
    p = argparse.ArgumentParser(description="Run Triple-KG-RAG relation extraction (reuse LocalGraphStore KG).")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--dataset-key", required=True)
    p.add_argument("--backend", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--output-format", choices=["full", "pred-only"], default="full")
    p.add_argument("--doc-type-filter", default="all", help="Filter doc['type'] or 'all'.")
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)

    # KG args (kg-path optional -> auto-pick from data/kg)
    p.add_argument("--kg-path", type=Path, default=None, help="Path to local KG JSON produced by build_kg_from_preprocessed.py")
    p.add_argument("--kg-max-hops", type=int, default=1)
    p.add_argument("--kg-max-triples", type=int, default=80)
    p.add_argument("--kg-include-in-edges", action="store_true", default=True)
    p.add_argument("--no-kg-include-in-edges", action="store_false", dest="kg_include_in_edges")

    # Prompt sizing
    p.add_argument("--max-sentences-in-prompt", type=int, default=None)
    p.add_argument("--max-triples-in-text", type=int, default=80)

    args = p.parse_args()

    cfg = load_config(args.config)

    kg_file = args.kg_path if args.kg_path else _autopick_kg_file(cfg, args.dataset_key)
    if not kg_file.exists():
        raise FileNotFoundError(f"KG file not found: {kg_file}")

    sections = RunnerLLMSections(
        llm_section="triple_kg_rag",
        prompt_section="triple_kg_rag",
        system_prompt_key="triple_kg_rag_docre",
    )

    def _prep(input_path: Path, cfg2: Dict[str, Any]) -> PreparedContext:
        # allow YAML defaults (optional)
        tri_cfg = cfg2.get("triple_kg_rag", {}) or {}
        retr_cfg = tri_cfg.get("retrieval", {}) or {}

        max_sent = args.max_sentences_in_prompt
        if max_sent is None:
            max_sent = tri_cfg.get("max_sentences_in_prompt")

        max_tr_in_text = args.max_triples_in_text
        if max_tr_in_text is None:
            max_tr_in_text = tri_cfg.get("max_triples_in_text", 80)

        return prepare_triple_kg_rag_context(
            input_path,
            cfg2,
            kg_path=kg_file,
            max_hops=int(args.kg_max_hops or retr_cfg.get("max_hops", 1)),
            max_triples=int(args.kg_max_triples or retr_cfg.get("max_triples", 80)),
            include_in_edges=bool(args.kg_include_in_edges),
            max_sentences_in_prompt=max_sent,
            max_triples_in_text=max_tr_in_text,
        )

    run_relation_experiment(
        strategy_cls=TripleKGRagRelationStrategy,
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
