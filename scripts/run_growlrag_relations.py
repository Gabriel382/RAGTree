# scripts/run_growlrag_relations.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tqdm import tqdm

from ragtree.processing.orchestrators.relations_runner import (
    PreparedContext,
    RunnerLLMSections,
    run_relation_experiment,
)
from ragtree.processing.rag.strategies.growlrag_relations import GrowlRagRelationStrategy
from ragtree.ontologies.retrieval.subontology import SubOntologyRetriever


def _parse_doc_types(arg: str) -> Sequence[str] | str:
    """
    '--doc-type-filter "dev,test"' -> ["dev", "test"]
    '--doc-type-filter "all"'      -> "all"
    """
    if not arg or arg == "all":
        return "all"
    items = [x.strip() for x in arg.split(",")]
    items = [x for x in items if x]
    return items or "all"


def _parse_cli_relation_types(arg: Optional[str]) -> Optional[List[str]]:
    """
    Optional comma-separated list of relation types.
    """
    if not arg:
        return None
    items = [x.strip() for x in arg.split(",")]
    items = [x for x in items if x]
    return items or None


def _collect_few_shots(
    input_path: Path,
    *,
    shot_type: str,
    shot_num: int,
    shot_skip: int = 0,
    shot_limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Collect few-shot examples from the same dataset JSONL.

    Rules:
      - Only docs where doc['type'] == shot_type
      - Only docs with non-empty doc['relations'] dict
      - Apply shot_skip / shot_limit AFTER type filter (same semantics as runner)
      - Stop when collected shot_num (if shot_num > 0)
    """
    if shot_num <= 0:
        return []

    few_shots: List[Dict[str, Any]] = []
    seen_after_type = 0
    considered = 0  # after skip/limit

    with input_path.open("r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc=f"[growlrag] Collecting few-shots (type={shot_type})", unit="doc"):
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)

            if doc.get("type") != shot_type:
                continue

            # after type filter
            seen_after_type += 1

            # skip after type filter
            if shot_skip and seen_after_type <= shot_skip:
                continue

            # limit after type filter+skip
            if shot_limit is not None and considered >= shot_limit:
                break

            considered += 1

            rels = doc.get("relations")
            if not isinstance(rels, dict) or not rels:
                continue

            few_shots.append(doc)
            if len(few_shots) >= shot_num:
                break

    print(
        f"[growlrag] few-shots: requested={shot_num} collected={len(few_shots)} "
        f"type={shot_type} shot_skip={shot_skip} shot_limit={shot_limit}"
    )
    return few_shots


def prepare_growlrag_context(
    input_path: Path,
    cfg: Dict[str, Any],
    *,
    shot_type: str,
    shot_num: int,
    shot_skip: int,
    shot_limit: Optional[int],
) -> PreparedContext:
    """
    Prepare strategy-level dependencies once:
      - SubOntologyRetriever with TTL path (cached internally)
      - Optional few_shots list (prompt examples)
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

    few_shots = _collect_few_shots(
        input_path,
        shot_type=shot_type,
        shot_num=shot_num,
        shot_skip=shot_skip,
        shot_limit=shot_limit,
    )

    predict_kwargs = {"few_shots": few_shots}  # strategy uses this; empty list => zero-shot
    return PreparedContext(strategy_kwargs=strategy_kwargs, predict_kwargs=predict_kwargs)


def main() -> None:
    p = argparse.ArgumentParser(description="Run GrOWL-RAG (ontology-guided RAG) relation extraction.")

    p.add_argument("--config", type=Path, default=None, help="Path to default.yaml (optional).")
    p.add_argument(
        "--dataset-key",
        required=True,
        help="Key in cfg['datasets']['preprocessed'] (ideally a *_olink_*.jsonl).",
    )
    p.add_argument("--backend", type=str, default=None, help="Override LLM backend.")
    p.add_argument("--model", type=str, default=None, help="Override LLM model.")

    p.add_argument(
        "--relation-types",
        type=str,
        default=None,
        help="Optional comma-separated relation schema override (e.g. 'P57 : director,P577 : publication date').",
    )

    p.add_argument("--output-format", choices=["full", "pred-only"], default="full")

    p.add_argument(
        "--doc-type-filter",
        type=str,
        default="all",
        help="Filter on doc['type'] (e.g. 'test' or 'dev,test') or 'all'. Default: 'all'.",
    )

    # Universal slicing for prediction (handled in relations_runner)
    p.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip the first N documents AFTER doc-type filtering. Default: 0.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most K documents AFTER doc-type filtering. Default: None (all).",
    )

    # Few-shot options for GrOWL-RAG (optional, paper-style)
    p.add_argument(
        "--growlrag-shot-type",
        type=str,
        default="train",
        help="doc['type'] used to collect few-shot examples. Default: 'train'.",
    )
    p.add_argument(
        "--growlrag-shot-num",
        type=int,
        default=0,
        help="Number of few-shot examples to include. Default: 0 (zero-shot).",
    )
    p.add_argument(
        "--growlrag-shot-skip",
        type=int,
        default=0,
        help="Skip first N eligible few-shot docs AFTER type filtering. Default: 0.",
    )
    p.add_argument(
        "--growlrag-shot-limit",
        type=int,
        default=None,
        help="Consider at most K eligible few-shot docs AFTER type filtering+skip. Default: None.",
    )

    args = p.parse_args()

    cli_rel_types = _parse_cli_relation_types(args.relation_types)
    doc_type_filter = _parse_doc_types(args.doc_type_filter)

    # Tell the runner which config sections to use
    sections = RunnerLLMSections(
        llm_section="growlrag",
        prompt_section="growlrag",
        system_prompt_key="growlrag_docre",
    )

    def _prep(input_path: Path, cfg: Dict[str, Any]) -> PreparedContext:
        return prepare_growlrag_context(
            input_path,
            cfg,
            shot_type=args.growlrag_shot_type,
            shot_num=args.growlrag_shot_num,
            shot_skip=args.growlrag_shot_skip,
            shot_limit=args.growlrag_shot_limit,
        )

    run_relation_experiment(
        strategy_cls=GrowlRagRelationStrategy,
        config_path=args.config,
        dataset_key=args.dataset_key,
        backend=args.backend,
        model=args.model,
        cli_relation_types=cli_rel_types,
        output_format=args.output_format,
        doc_type_filter=doc_type_filter,
        skip=args.skip,
        limit=args.limit,
        sections=sections,
        prepare_context_fn=_prep,
    )


if __name__ == "__main__":
    main()
