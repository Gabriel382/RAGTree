# scripts/run_ograg_relations.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from ragtree.core.config import load_config
from ragtree.processing.orchestrators.relations_runner import _build_llm_config_from_yaml, RunnerLLMSections
from ragtree.processing.rag.strategies.ograg_relations import OGRagParams, OGRagRelationStrategy


def _parse_doc_types(arg: str) -> Sequence[str] | str:
    if not arg or arg == "all":
        return "all"
    items = [x.strip() for x in arg.split(",")]
    items = [x for x in items if x]
    return items or "all"


def _resolve_input_path(cfg: Dict[str, Any], dataset_key: Optional[str], input_path: Optional[Path]) -> Path:
    if input_path is not None:
        return input_path

    if not dataset_key:
        raise ValueError("You must provide either --dataset-key or --input-path.")

    ds_pre = cfg.get("datasets", {}).get("preprocessed", {})
    if dataset_key not in ds_pre:
        available = ", ".join(sorted(ds_pre.keys()))
        raise KeyError(f"Unknown dataset key '{dataset_key}'. Available: {available}")
    return Path(ds_pre[dataset_key])


def _load_jsonl_doc_map(path: Path, field: str) -> Dict[str, Any]:
    """
    Load a JSONL and return {document_id: obj[field]} for lines that have it.
    """
    out: Dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj.get("document_id") or obj.get("id")
            if not doc_id:
                continue
            if field in obj:
                out[doc_id] = obj[field]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Run OG-RAG (ontology-grounded RAG) relation extraction.")

    p.add_argument("--config", type=Path, default=None, help="Path to default.yaml (optional).")

    # Input dataset
    p.add_argument("--dataset-key", default=None, help="Key in cfg['datasets']['preprocessed'].")
    p.add_argument("--input-path", type=Path, default=None, help="Direct path to input JSONL (overrides --dataset-key).")

    # Ontology
    p.add_argument("--ontology-key", required=True, help="Key in cfg['ontology'] (e.g., docredontology, framenet, owltime...).")
    p.add_argument("--ontology-method", default="llm_embedding", help="Linking method label to store in _meta / debugging.")
    p.add_argument("--ontology-links-path", type=Path, default=None, help="JSONL with ontology_links per doc_id. If omitted, assumes input docs already contain ontology_links.")

    # LLM
    p.add_argument("--backend", type=str, default=None, help="Override LLM backend.")
    p.add_argument("--model", type=str, default=None, help="Override LLM model.")

    # Output
    p.add_argument("--output-format", choices=["full", "pred-only"], default="full")
    p.add_argument("--method", default="ograg", help="Label used in output filename (default: ograg).")

    # Filters + slicing
    p.add_argument("--doc-types", type=str, default="all", help="Doc types: 'all' or comma-separated (train,dev,test).")
    p.add_argument("--skip", type=int, default=0, help="Skip first N AFTER doc-type filtering.")
    p.add_argument("--limit", type=int, default=None, help="Take at most K AFTER doc-type filtering (None=all).")

    # Optional fixed relation schema
    p.add_argument("--relation-types", type=str, default=None, help="Optional comma-separated relation types to enforce.")

    # OGRAG retrieval knobs
    p.add_argument("--top-k-entities", type=int, default=None)
    p.add_argument("--max-ontology-lines", type=int, default=None)

    args = p.parse_args()
    cfg = load_config(args.config)

    input_path = _resolve_input_path(cfg, args.dataset_key, args.input_path)

    processed_root = Path(cfg["paths"]["data_processed"])
    processed_root.mkdir(parents=True, exist_ok=True)

    # If user provided dataset-key, include it in output filename; else use stem of input file
    ds_label = args.dataset_key or input_path.stem
    backend_label = args.backend or cfg["llm"]["ograg"].get("default_backend", "ollama")
    output_path = processed_root / f"{ds_label}.{args.method}.{backend_label}.jsonl"

    # Ontology TTL path from YAML
    onto_paths = cfg.get("ontology", {}) or {}
    if args.ontology_key not in onto_paths:
        available = ", ".join(sorted(onto_paths.keys()))
        raise KeyError(f"Unknown ontology key '{args.ontology_key}'. Available: {available}")
    ttl_path = Path(onto_paths[args.ontology_key])

    # Build llm_config using existing helper (no relations_runner patch)
    sections = RunnerLLMSections(
        llm_section="ograg",
        prompt_section="ograg",
        system_prompt_key="ograg_docre",
    )
    llm_config = _build_llm_config_from_yaml(
        cfg=cfg,
        sections=sections,
        backend_override=args.backend,
        model_override=args.model,
    )

    # Load ontology_links if provided
    docid_to_links: Dict[str, Any] = {}
    if args.ontology_links_path is not None:
        docid_to_links = _load_jsonl_doc_map(args.ontology_links_path, "ontology_links")

    # Relation schema override
    cli_rel_types: Optional[List[str]] = None
    if args.relation_types:
        items = [x.strip() for x in args.relation_types.split(",")]
        cli_rel_types = [x for x in items if x] or None

    # Parse doc types filter
    doc_type_filter = _parse_doc_types(args.doc_types)

    # OGRAG params (with YAML defaults optionally)
    ograg_cfg = cfg.get("ograg", {}) or {}
    retr_cfg = ograg_cfg.get("retrieval", {}) or {}

    params = OGRagParams(
        ontology_key=args.ontology_key,
        ontology_ttl_path=ttl_path,
        linking_method=args.ontology_method,
        top_k_entities=int(args.top_k_entities or retr_cfg.get("top_k_entities", 6)),
        max_ontology_lines=int(args.max_ontology_lines or retr_cfg.get("max_ontology_lines", 200)),
        include_labels=bool(retr_cfg.get("include_labels", True)),
        include_comments=bool(retr_cfg.get("include_comments", True)),
        include_types=bool(retr_cfg.get("include_types", True)),
    )

    strategy = OGRagRelationStrategy(llm_config, params=params)

    # Run loop with filter->skip->limit semantics
    seen_after_filter = 0
    wrote = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"OGRAG on {ds_label}", unit="doc"):
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)

            # merge ontology_links from external file if needed
            if docid_to_links:
                doc_id = doc.get("document_id")
                if doc_id and doc_id in docid_to_links:
                    doc["ontology_links"] = docid_to_links[doc_id]

            # doc type filter
            if doc_type_filter != "all":
                if doc.get("type") not in set(doc_type_filter):  # type: ignore[arg-type]
                    continue

            # skip/limit after filtering
            if seen_after_filter < args.skip:
                seen_after_filter += 1
                continue
            if args.limit is not None and (seen_after_filter - args.skip) >= args.limit:
                break
            seen_after_filter += 1

            rel_types = cli_rel_types  # may be None => infer per doc
            pred = strategy.predict_relations(doc, relation_types=rel_types)

            if args.output_format == "pred-only":
                out_obj = {"document_id": doc.get("document_id"), "pred_relations": pred}
            else:
                doc["pred_relations"] = pred
                out_obj = doc

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            wrote += 1

    print(f"[ograg] wrote: {output_path} ({wrote} docs)")


if __name__ == "__main__":
    main()
