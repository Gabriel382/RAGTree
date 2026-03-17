#!/usr/bin/env python3
# scripts/run_community_kgrag_relations.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tqdm import tqdm

from ragtree.core.config import load_config
from ragtree.processing.orchestrators.relations_runner import (
    PreparedContext,
    RunnerLLMSections,
    run_relation_experiment,
)
from ragtree.kg.community_kgrag.retriever import CommunityKGRetriever
from ragtree.processing.rag.strategies.community_kgrag_relations import (
    CommunityKGRAGRelationStrategy,
    CommunityKGRAGParams,
)


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


def _collect_few_shots(
    input_path: Path,
    *,
    shot_type: str,
    shot_num: int,
    shot_skip: int = 0,
    shot_limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Collect few-shot examples from the SAME dataset JSONL.

    Rules:
      - Only docs where doc['type'] == shot_type
      - Only docs with non-empty doc['relations'] dict
      - Apply shot_skip / shot_limit AFTER type filter
      - Stop when collected shot_num
    """
    if shot_num <= 0:
        return []

    few_shots: List[Dict[str, Any]] = []
    seen_after_type = 0
    considered = 0

    with input_path.open("r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc=f"[community_kgrag] Collecting few-shots (type={shot_type})", unit="doc"):
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)

            if doc.get("type") != shot_type:
                continue

            seen_after_type += 1

            if shot_skip and seen_after_type <= shot_skip:
                continue

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
        f"[community_kgrag] few-shots: requested={shot_num} collected={len(few_shots)} "
        f"type={shot_type} shot_skip={shot_skip} shot_limit={shot_limit}"
    )
    return few_shots


def prepare_community_kgrag_context(
    input_path: Path,
    cfg: Dict[str, Any],
    *,
    communitykg_root: Path,
    dataset_key: str,
    # retrieval knobs
    top_communities: int,
    delta_percent: Optional[float],
    top_sentences: int,
    lambda_percent: Optional[float],
    use_sentence_faiss: bool,
    max_ctx_chars: int,
    query_embed_model: str,
    sentence_embed_model: str,
    # few-shot knobs
    shot_type: str,
    shot_num: int,
    shot_skip: int,
    shot_limit: Optional[int],
) -> PreparedContext:
    """
    Prepare dependencies ONCE (retriever + optional few_shots).
    Returned PreparedContext:
      - strategy_kwargs: passed to CommunityKGRAGRelationStrategy constructor
      - predict_kwargs: passed to strategy.predict_relations(...)
    """
    dataset_root = communitykg_root / dataset_key

    retriever = CommunityKGRetriever(
        dataset_root,
        query_embed_model=query_embed_model,
        sentence_embed_model=sentence_embed_model,
    )

    params = CommunityKGRAGParams(
        top_communities=top_communities,
        delta_percent=delta_percent,
        top_sentences=top_sentences,
        lambda_percent=lambda_percent,
        use_sentence_faiss=use_sentence_faiss,
        max_ctx_chars=max_ctx_chars,
    )

    strategy_kwargs = {
        "retriever": retriever,
        "params": params,
    }

    few_shots = _collect_few_shots(
        input_path,
        shot_type=shot_type,
        shot_num=shot_num,
        shot_skip=shot_skip,
        shot_limit=shot_limit,
    )

    predict_kwargs = {"few_shots": few_shots}
    return PreparedContext(strategy_kwargs=strategy_kwargs, predict_kwargs=predict_kwargs)


def main() -> None:
    p = argparse.ArgumentParser(description="Run CommunityKG-RAG relation extraction (with optional few-shot).")

    p.add_argument("--config", type=Path, default=None, help="Path to default.yaml (optional).")
    p.add_argument("--dataset-key", required=True, help="Key in cfg['datasets']['preprocessed'].")

    p.add_argument("--backend", type=str, default=None, help="Override LLM backend.")
    p.add_argument("--model", type=str, default=None, help="Override LLM model.")

    p.add_argument("--output-format", choices=["full", "pred-only"], default="full")

    p.add_argument(
        "--doc-type-filter",
        type=str,
        default="all",
        help="Filter on doc['type'] (e.g. 'test' or 'dev,test') or 'all'.",
    )

    p.add_argument("--skip", type=int, default=0, help="Skip N docs AFTER doc-type filtering.")
    p.add_argument("--limit", type=int, default=None, help="Process at most K docs AFTER doc-type filtering.")

    # artifacts root
    p.add_argument("--communitykg-root", type=Path, required=True, help="Root dir containing {dataset_key}/meta.json etc.")

    # retrieval knobs
    p.add_argument("--top-communities", type=int, default=50)
    p.add_argument("--delta-percent", type=float, default=None)
    p.add_argument("--top-sentences", type=int, default=12)
    p.add_argument("--lambda-percent", type=float, default=None)
    p.add_argument("--use-sentence-faiss", type=str, default="true", choices=["true", "false"])
    p.add_argument("--max-ctx-chars", type=int, default=8000)

    p.add_argument("--query-embed-model", type=str, default="BAAI/bge-m3")
    p.add_argument("--sentence-embed-model", type=str, default="BAAI/bge-m3")

    # few-shot knobs
    p.add_argument("--shot-type", type=str, default="train")
    p.add_argument("--shot-num", type=int, default=0)
    p.add_argument("--shot-skip", type=int, default=0)
    p.add_argument("--shot-limit", type=int, default=None)

    args = p.parse_args()
    cfg = load_config(args.config)

    doc_type_filter = _parse_doc_types(args.doc_type_filter)

    # Tell the runner which LLM/prompt sections to use
    sections = RunnerLLMSections(
        llm_section="community_kgrag",
        prompt_section="community_kgrag",
        system_prompt_key="community_kgrag_docre",
    )

    def _prep(input_path: Path, cfg2: Dict[str, Any]) -> PreparedContext:
        return prepare_community_kgrag_context(
            input_path,
            cfg2,
            communitykg_root=args.communitykg_root,
            dataset_key=args.dataset_key,
            top_communities=args.top_communities,
            delta_percent=args.delta_percent,
            top_sentences=args.top_sentences,
            lambda_percent=args.lambda_percent,
            use_sentence_faiss=(args.use_sentence_faiss == "true"),
            max_ctx_chars=args.max_ctx_chars,
            query_embed_model=args.query_embed_model,
            sentence_embed_model=args.sentence_embed_model,
            shot_type=args.shot_type,
            shot_num=args.shot_num,
            shot_skip=args.shot_skip,
            shot_limit=args.shot_limit,
        )

    run_relation_experiment(
        strategy_cls=CommunityKGRAGRelationStrategy,
        config_path=args.config,
        dataset_key=args.dataset_key,
        backend=args.backend,
        model=args.model,
        cli_relation_types=None,
        output_format=args.output_format,
        doc_type_filter=doc_type_filter,
        skip=args.skip,
        limit=args.limit,
        sections=sections,
        prepare_context_fn=_prep,
    )


if __name__ == "__main__":
    main()
