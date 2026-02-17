# scripts/run_langgraph_agentic_simple_relations.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tqdm import tqdm

from ragtree.core.config import load_config
from ragtree.processing.orchestrators.relations_runner import (
    RunnerLLMSections,
    _build_llm_config_from_yaml,
    _determine_relation_types_for_doc,
    _augment_docred_relation_types,
    _load_docred_rel_info,
    _doc_type_matches,
)
from ragtree.processing.rag.strategies.langgraph_agentic_simple_relations import (
    LangGraphAgenticSimpleRelationStrategy,
    LangGraphAgenticSimpleParams,
)


def _parse_csv_or_all(arg: str) -> Sequence[str] | str:
    if not arg or arg == "all":
        return "all"
    items = [x.strip() for x in arg.split(",") if x.strip()]
    return items or "all"


def _resolve_input_path(cfg: Dict[str, Any], dataset_key: str) -> Path:
    ds_pre = cfg.get("datasets", {}).get("preprocessed", {}) or {}
    if dataset_key in ds_pre:
        return Path(ds_pre[dataset_key])

    p = Path(dataset_key)
    if p.exists() and p.is_file():
        return p

    pre_root = Path(cfg["paths"]["data_preprocessed"])
    cand1 = pre_root / f"{dataset_key}.jsonl"
    if cand1.exists():
        return cand1
    cand2 = pre_root / dataset_key
    if cand2.exists():
        return cand2

    raise KeyError(f"Unknown dataset key '{dataset_key}'. Not in YAML and no file found under {pre_root}.")


def _resolve_processed_root(cfg: Dict[str, Any]) -> Path:
    out = Path(cfg["paths"]["data_processed"])
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run LangGraph single-agent simple Agentic RAG (web + wikidata only) for DocRE."
    )

    p.add_argument("--config", type=Path, default=None, help="Path to default.yaml (optional).")
    p.add_argument("--dataset-key", required=True, help="Dataset key (YAML key or file/filename).")
    p.add_argument("--backend", type=str, default=None, help="Override LLM backend.")
    p.add_argument("--model", type=str, default=None, help="Override LLM model.")
    p.add_argument("--output-format", choices=["full", "pred-only"], default="full")

    # Prediction slice
    p.add_argument("--doc-types", default="all", help="Doc types to predict on: all or train,dev,test.")
    p.add_argument("--skip", type=int, default=0, help="Skip N docs AFTER doc-type filtering.")
    p.add_argument("--limit", type=int, default=None, help="Process at most K docs AFTER doc-type filtering.")

    # Few-shot pool
    p.add_argument("--shot-num", type=int, default=0, help="Few-shot demonstrations to include (0 disables).")
    p.add_argument("--shot-dataset-key", default=None, help="Dataset key to sample few-shots from (default: dataset-key).")
    p.add_argument("--shot-doc-types", default="train", help="Doc types for few-shot pool: all or train,dev,test.")
    p.add_argument("--shot-skip", type=int, default=0)
    p.add_argument("--shot-limit", type=int, default=None)

    # Planner + tools
    p.add_argument("--planner-mode", choices=["rule", "llm"], default="rule", help="Planner type. 'rule' is fastest.")
    p.add_argument("--planner-depth", type=int, default=1, help="How many planning iterations (1 is minimal).")
    p.add_argument("--planner-verbosity", choices=["minimal", "normal", "verbose"], default="minimal")

    # Tools are ON by default -> provide disabling flags
    p.add_argument("--no-web", action="store_false", dest="enable_web", help="Disable Web tool (Wikipedia).")
    p.add_argument("--no-wikidata", action="store_false", dest="enable_wikidata", help="Disable Wikidata tool.")
    p.set_defaults(enable_web=True, enable_wikidata=True)

    p.add_argument("--web-timeout-sec", type=float, default=3.0)
    p.add_argument("--web-max-chars", type=int, default=1500)
    p.add_argument("--wikidata-timeout-sec", type=float, default=3.0)
    p.add_argument("--wikidata-max-chars", type=int, default=1500)
    p.add_argument("--web-top-k", type=int, default=2)
    p.add_argument("--wikidata-top-k", type=int, default=2)

    # Prompt sizing
    p.add_argument("--max-sentences-in-prompt", type=int, default=None)

    # LLM budget
    p.add_argument("--max-llm-calls", type=int, default=1, help="Total LLM calls budget per doc (default 1).")

    args = p.parse_args()

    cfg = load_config(args.config)
    input_path = _resolve_input_path(cfg, args.dataset_key)
    processed_root = _resolve_processed_root(cfg)

    doc_types = _parse_csv_or_all(args.doc_types)
    shot_types = _parse_csv_or_all(args.shot_doc_types)

    # Prefer a dedicated YAML section if you add it; fallback to agentic_hybrid
    try:
        sections = RunnerLLMSections(
            llm_section="agentic_simple",
            prompt_section="agentic_simple",
            system_prompt_key="agentic_simple_docre",
        )
        llm_config = _build_llm_config_from_yaml(
            cfg=cfg,
            sections=sections,
            backend_override=args.backend,
            model_override=args.model,
        )
    except Exception:
        sections = RunnerLLMSections(
            llm_section="agentic_hybrid",
            prompt_section="agentic_hybrid",
            system_prompt_key="agentic_hybrid_docre",
        )
        llm_config = _build_llm_config_from_yaml(
            cfg=cfg,
            sections=sections,
            backend_override=args.backend,
            model_override=args.model,
        )

    output_path = processed_root / f"{args.dataset_key}.langgraph_agentic_simple.{llm_config.backend}.jsonl"

    # DocRED rel_info (optional)
    docred_rel_info: Optional[Dict[str, str]] = None
    if "docred" in str(args.dataset_key).lower():
        try:
            docred_rel_info = _load_docred_rel_info(cfg)
            print(f"[langgraph_agentic_simple] Loaded DocRED rel_info with {len(docred_rel_info)} entries.")
        except Exception as e:
            print(f"[langgraph_agentic_simple] Warning: could not load DocRED rel_info.json: {e}")

    print(f"[langgraph_agentic_simple] input={input_path}")
    print(f"[langgraph_agentic_simple] output={output_path}")
    print(f"[langgraph_agentic_simple] predict doc-types={doc_types} skip={args.skip} limit={args.limit}")
    print(f"[langgraph_agentic_simple] backend={llm_config.backend} model={llm_config.model}")
    print(f"[langgraph_agentic_simple] planner_mode={args.planner_mode} max_llm_calls={args.max_llm_calls} web={args.enable_web} wikidata={args.enable_wikidata}")

    # Few-shots (reusing the same helper logic style: sample from gold docs with relations)
    few_shots: List[Dict[str, Any]] = []
    if args.shot_num > 0:
        shot_dataset_key = args.shot_dataset_key or args.dataset_key
        shot_input_path = _resolve_input_path(cfg, shot_dataset_key)

        kept_after_filter = 0
        with shot_input_path.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)

                if not _doc_type_matches(d, shot_types):
                    continue

                rels = d.get("relations")
                if not isinstance(rels, dict) or not rels:
                    continue

                if kept_after_filter < args.shot_skip:
                    kept_after_filter += 1
                    continue
                if args.shot_limit is not None and (kept_after_filter - args.shot_skip) >= args.shot_limit:
                    break

                few_shots.append(d)
                kept_after_filter += 1
                if len(few_shots) >= args.shot_num:
                    break

        print(f"[langgraph_agentic_simple] few-shots collected: {len(few_shots)}")

    params = LangGraphAgenticSimpleParams(
        max_sentences_in_prompt=args.max_sentences_in_prompt,
        planner_mode=str(args.planner_mode),
        planner_depth=int(args.planner_depth),
        planner_verbosity=str(args.planner_verbosity),
        enable_web=bool(args.enable_web),
        enable_wikidata=bool(args.enable_wikidata),
        web_timeout_sec=float(args.web_timeout_sec),
        web_max_chars=int(args.web_max_chars),
        wikidata_timeout_sec=float(args.wikidata_timeout_sec),
        wikidata_max_chars=int(args.wikidata_max_chars),
        web_top_k=int(args.web_top_k),
        wikidata_top_k=int(args.wikidata_top_k),
        max_llm_calls=int(args.max_llm_calls),
    )

    strategy = LangGraphAgenticSimpleRelationStrategy(llm_config=llm_config, params=params)

    num_seen_after_type = 0
    num_pred = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"LangGraphAgenticSimple on {args.dataset_key}", unit="doc"):
            line = line.strip()
            if not line:
                continue

            doc = json.loads(line)

            if not _doc_type_matches(doc, doc_types):
                continue

            if num_seen_after_type < args.skip:
                num_seen_after_type += 1
                continue
            if args.limit is not None and (num_seen_after_type - args.skip) >= args.limit:
                break
            num_seen_after_type += 1

            relation_types_for_doc = _determine_relation_types_for_doc(doc, cli_relation_types=None)
            if docred_rel_info is not None and "docred" in str(args.dataset_key).lower():
                relation_types_for_doc = _augment_docred_relation_types(relation_types_for_doc, docred_rel_info)

            pred_relations = strategy.predict_relations(
                doc,
                relation_types=relation_types_for_doc,
                few_shots=few_shots,
            )

            if args.output_format == "pred-only":
                out_obj: Dict[str, Any] = {
                    "document_id": doc.get("document_id") or doc.get("id"),
                    "pred_relations": pred_relations,
                }
            else:
                doc["pred_relations"] = pred_relations
                out_obj = doc

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            num_pred += 1

    print(f"[langgraph_agentic_simple] done. predicted={num_pred}")
    print(f"[langgraph_agentic_simple] wrote: {output_path}")


if __name__ == "__main__":
    main()
