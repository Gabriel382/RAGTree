#!/usr/bin/env python3
# scripts/run_chunk_orag_relations.py
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
from ragtree.processing.rag.strategies.chunk_orag_relations import ChunkORAGRelationStrategy, ChunkORAGParams
from ragtree.ontologies.retrieval.chunk_orag_retriever import ChunkORAGRetriever


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
    Collect few-shot examples from the same dataset JSONL.

    Rules:
      - Only docs where doc['type'] == shot_type
      - Only docs with non-empty doc['relations'] dict
      - Apply shot_skip / shot_limit AFTER type filter
      - Stop when collected shot_num (if shot_num > 0)
    """
    if shot_num <= 0:
        return []

    few_shots: List[Dict[str, Any]] = []
    seen_after_type = 0
    considered = 0

    with input_path.open("r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc=f"[chunk-orag] Collecting few-shots (type={shot_type})", unit="doc"):
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
        f"[chunk-orag] few-shots: requested={shot_num} collected={len(few_shots)} "
        f"type={shot_type} shot_skip={shot_skip} shot_limit={shot_limit}"
    )
    return few_shots


def prepare_chunk_orag_context(
    input_path: Path,
    cfg: Dict[str, Any],
    *,
    index_dir: Path,
    embed_model: str,
    device: str,
    use_reranker: bool,
    reranker_model: str,
    top_k: int,
    #rerank_top_n: int,
    auto_merge: bool,
    max_ctx_chars: int,
    shot_type: str,
    shot_num: int,
    shot_skip: int,
    shot_limit: Optional[int],
) -> PreparedContext:
    """
    Prepare strategy-level dependencies once:
      - ChunkORAGRetriever
      - ChunkORAGParams
      - Optional few_shots list
    """
    retriever = ChunkORAGRetriever(
        index_dir,
        embed_model=embed_model,
        #use_reranker=use_reranker,
        #reranker_model=reranker_model,
        device=device,  # CPU-safe
    )

    params = ChunkORAGParams(
        top_k=top_k,
        #rerank_top_n=rerank_top_n,
        #auto_merge=auto_merge,
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
    p = argparse.ArgumentParser(description="Run Chunk-O-RAG relation extraction.")

    p.add_argument("--config", type=Path, default=None)

    p.add_argument("--dataset-key", required=True)
    p.add_argument("--backend", type=str, default=None)
    p.add_argument("--model", type=str, default=None)

    p.add_argument(
        "--doc-type-filter",
        type=str,
        default="all",
        help="Filter on doc['type'] (e.g. 'test' or 'dev,test') or 'all'. Default: 'all'.",
    )

    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)

    # index / retrieval
    p.add_argument("--ontology-key", required=True)
    p.add_argument("--index-dir", type=Path, required=True)

    p.add_argument("--embed-model", type=str, default="BAAI/bge-m3")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--rerank-top-n", type=int, default=6)
    p.add_argument("--auto-merge", type=str, default="true", choices=["true", "false"])

    p.add_argument("--use-reranker", type=str, default="false", choices=["true", "false"])
    p.add_argument("--reranker-model", type=str, default="BAAI/bge-reranker-large")

    p.add_argument("--max-ctx-chars", type=int, default=6000)

    p.add_argument("--output-format", choices=["full", "pred-only"], default="full")

    # Few-shot options
    p.add_argument("--shot-type", type=str, default="train")
    p.add_argument("--shot-num", type=int, default=0)
    p.add_argument("--shot-skip", type=int, default=0)
    p.add_argument("--shot-limit", type=int, default=None)

    args = p.parse_args()

    doc_type_filter = _parse_doc_types(args.doc_type_filter)

    sections = RunnerLLMSections(
        llm_section="chunk_orag",
        prompt_section="chunk_orag",
        system_prompt_key="chunk_orag_docre",
    )

    def _prep(input_path: Path, cfg: Dict[str, Any]) -> PreparedContext:
        return prepare_chunk_orag_context(
            input_path,
            cfg,
            index_dir=args.index_dir,
            embed_model=args.embed_model,
            device=args.device,
            use_reranker=(args.use_reranker == "true"),
            reranker_model=args.reranker_model,
            top_k=args.top_k,
            #rerank_top_n=args.rerank_top_n,
            auto_merge=(args.auto_merge == "true"),
            max_ctx_chars=args.max_ctx_chars,
            shot_type=args.shot_type,
            shot_num=args.shot_num,
            shot_skip=args.shot_skip,
            shot_limit=args.shot_limit,
        )

    run_relation_experiment(
        strategy_cls=ChunkORAGRelationStrategy,
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