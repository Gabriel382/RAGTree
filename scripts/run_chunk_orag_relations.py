#!/usr/bin/env python3
# scripts/run_chunk_orag_relations.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tqdm import tqdm

from ragtree.core.config import load_config
from ragtree.processing.orchestrators.relations_runner import _build_llm_config_from_yaml, RunnerLLMSections
from ragtree.ontologies.retrieval.chunk_orag_retriever import ChunkORAGRetriever
from ragtree.processing.rag.strategies.chunk_orag_relations import ChunkORAGRelationStrategy, ChunkORAGParams


def _parse_doc_types(arg: str) -> Sequence[str] | str:
    if not arg or arg == "all":
        return "all"
    items = [x.strip() for x in arg.split(",") if x.strip()]
    return items or "all"


def _resolve_input_path(cfg: Dict[str, Any], dataset_key: Optional[str], input_path: Optional[Path]) -> Path:
    if input_path is not None:
        return input_path
    if not dataset_key:
        raise ValueError("Provide --dataset-key or --input-path.")
    ds_pre = cfg.get("datasets", {}).get("preprocessed", {})
    if dataset_key not in ds_pre:
        raise KeyError(f"Unknown dataset-key '{dataset_key}'. Available: {sorted(ds_pre.keys())}")
    return Path(ds_pre[dataset_key])


def main() -> None:
    ap = argparse.ArgumentParser("Run Chunk-O-RAG relation extraction (ontology chunk retrieval + LLM).")

    ap.add_argument("--config", type=Path, default=None)

    # data
    ap.add_argument("--dataset-key", default=None)
    ap.add_argument("--input-path", type=Path, default=None)
    ap.add_argument("--doc-types", default="all", help="all or comma-separated: train,dev,test")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)

    # ontology / index
    ap.add_argument("--ontology-key", required=True)
    ap.add_argument("--index-dir", type=Path, required=True, help="Path produced by build_chunk_orag_index.py")

    # LLM
    ap.add_argument("--backend", type=str, default=None)
    ap.add_argument("--model", type=str, default=None)

    # retrieval knobs
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--rerank-top-n", type=int, default=6)
    ap.add_argument("--auto-merge", type=str, default="true", choices=["true", "false"])
    ap.add_argument("--use-reranker", type=str, default="false", choices=["true", "false"])
    ap.add_argument("--reranker-model", type=str, default="BAAI/bge-reranker-large")
    ap.add_argument("--embed-model", type=str, default="BAAI/bge-m3")
    ap.add_argument("--max-ctx-chars", type=int, default=8000)

    # output
    ap.add_argument("--output-format", choices=["full", "pred-only"], default="full")
    ap.add_argument("--method", default="chunk_orag")

    args = ap.parse_args()
    cfg = load_config(args.config)

    input_path = _resolve_input_path(cfg, args.dataset_key, args.input_path)

    processed_root = Path(cfg["paths"]["data_processed"])
    processed_root.mkdir(parents=True, exist_ok=True)

    ds_label = args.dataset_key or input_path.stem
    backend_label = args.backend or cfg.get("llm", {}).get("chunk_orag", {}).get("default_backend", "ollama")
    output_path = processed_root / f"{ds_label}.{args.method}.{backend_label}.jsonl"

    # LLM config
    sections = RunnerLLMSections(
        llm_section="chunk_orag",
        prompt_section="chunk_orag",
        system_prompt_key="chunk_orag_docre",
    )
    llm_config = _build_llm_config_from_yaml(cfg, sections, args.backend, args.model)

    # Retriever
    retriever = ChunkORAGRetriever(
        args.index_dir,
        embed_model=args.embed_model,
        use_reranker=(args.use_reranker == "true"),
        reranker_model=args.reranker_model,
    )

    params = ChunkORAGParams(
        top_k=args.top_k,
        rerank_top_n=args.rerank_top_n,
        auto_merge=(args.auto_merge == "true"),
        max_ctx_chars=args.max_ctx_chars,
    )

    strategy = ChunkORAGRelationStrategy(llm_config, retriever=retriever, params=params)

    doc_type_filter = _parse_doc_types(args.doc_types)
    allowed_types = set(doc_type_filter) if doc_type_filter != "all" else None

    seen_after_filter = 0
    wrote = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"ChunkORAG on {ds_label}", unit="doc"):
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)

            if allowed_types is not None:
                if doc.get("type") not in allowed_types:
                    continue

            if seen_after_filter < args.skip:
                seen_after_filter += 1
                continue
            if args.limit is not None and (seen_after_filter - args.skip) >= args.limit:
                break
            seen_after_filter += 1

            pred = strategy.predict_relations(doc)

            if args.output_format == "pred-only":
                out_obj = {"document_id": doc.get("document_id"), "pred_relations": pred}
            else:
                doc["pred_relations"] = pred
                out_obj = doc

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            wrote += 1

    print(f"[chunk-orag] wrote: {output_path} ({wrote} docs)")


if __name__ == "__main__":
    main()
