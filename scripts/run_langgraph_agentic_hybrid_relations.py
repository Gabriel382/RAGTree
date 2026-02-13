# scripts/run_langgraph_agentic_hybrid_relations.py
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
from ragtree.ontologies.retrieval.subontology import SubOntologyRetriever
from ragtree.processing.rag.strategies.langgraph_agentic_hybrid_relations import (
    LangGraphAgenticHybridRelationStrategy,
    LangGraphAgenticHybridParams,
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


def _load_jsonl_doc_map(path: Path, field: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc_id = doc.get("document_id") or doc.get("id")
            if not doc_id:
                continue
            if field in doc:
                out[str(doc_id)] = doc[field]
    return out


def _load_kg_doc_triples(path: Path) -> Dict[str, List[Any]]:
    """
    Supports:
      - JSON (single object: list/dict)
      - JSONL (one object per line)
    Per-doc shapes supported:
      - {"document_id":..., "triples":[...]} or "kg_triples"/"edges"
      - [{"document_id":..., "triples":[...]}...]
      - {"docs":[{...}, ...]}
      - {"docid": {"triples":[...]}, ...}
      - {"docid": [...triples...], ...}
    """
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return {}

    out: Dict[str, List[Any]] = {}

    def ingest_obj(obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        doc_id = obj.get("document_id") or obj.get("id")
        if not doc_id:
            return
        triples = obj.get("triples") or obj.get("kg_triples") or obj.get("edges") or []
        if isinstance(triples, list):
            out[str(doc_id)] = triples

    # Try JSON first
    try:
        root = json.loads(txt)

        if isinstance(root, list):
            for item in root:
                ingest_obj(item)
            if out:
                return out

        if isinstance(root, dict) and isinstance(root.get("docs"), list):
            for item in root["docs"]:
                ingest_obj(item)
            if out:
                return out

        if isinstance(root, dict):
            if "document_id" in root or "id" in root:
                ingest_obj(root)
                if out:
                    return out

            for k, v in root.items():
                if isinstance(v, dict):
                    obj = {"document_id": k, **v}
                    ingest_obj(obj)
                elif isinstance(v, list):
                    out[str(k)] = v
            if out:
                return out

    except json.JSONDecodeError:
        pass

    # Fallback JSONL
    bad = 0
    first_bad = None
    for i, line in enumerate(txt.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            if first_bad is None:
                first_bad = (i, line[:200])
            continue
        ingest_obj(obj)

    if not out:
        if first_bad:
            ln, preview = first_bad
            raise ValueError(
                f"KG file '{path}' could not be parsed as JSON or JSONL. "
                f"First invalid JSONL line #{ln}: {preview}"
            )
        raise ValueError(f"KG file '{path}' could not be parsed (no doc triples found).")

    if bad > 0:
        print(f"[langgraph_agentic_hybrid] Warning: skipped {bad} invalid JSONL lines in {path}")

    return out


def _collect_few_shots(
    path: Path,
    *,
    shot_doc_types: Sequence[str] | str,
    shot_num: int,
    shot_skip: int,
    shot_limit: Optional[int],
) -> List[Dict[str, Any]]:
    if shot_num <= 0:
        return []

    few: List[Dict[str, Any]] = []
    kept_after_filter = 0

    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)

            if not _doc_type_matches(doc, shot_doc_types):
                continue

            rels = doc.get("relations")
            if not isinstance(rels, dict) or not rels:
                continue

            if kept_after_filter < shot_skip:
                kept_after_filter += 1
                continue

            if shot_limit is not None and (kept_after_filter - shot_skip) >= shot_limit:
                break

            few.append(doc)
            kept_after_filter += 1

            if len(few) >= shot_num:
                break

    return few


def main() -> None:
    p = argparse.ArgumentParser(description="Run LangGraph single-agent hybrid RAG (ontology + KG + optional web/wikidata) for DocRE.")

    p.add_argument("--config", type=Path, default=None, help="Path to default.yaml (optional).")
    p.add_argument("--dataset-key", required=True, help="Dataset key (YAML key or file/filename).")
    p.add_argument("--backend", type=str, default=None, help="Override LLM backend.")
    p.add_argument("--model", type=str, default=None, help="Override LLM model.")
    p.add_argument("--output-format", choices=["full", "pred-only"], default="full")

    # Prediction slice
    p.add_argument("--doc-types", default="all", help="Doc types to predict on: all or train,dev,test.")
    p.add_argument("--skip", type=int, default=0, help="Skip N docs AFTER doc-type filtering.")
    p.add_argument("--limit", type=int, default=None, help="Process at most K docs AFTER doc-type filtering.")

    # Ontology artifact (path-based, as you prefer)
    p.add_argument("--ontology-links-path", type=str, required=True, help="Path to ontology-linking JSONL artifact.")
    p.add_argument("--ontology-key", default="docredontology", help="Ontology key used to load TTL via config['ontology'][key].")
    p.add_argument("--ontology-method", default="llm_embedding", help="Linking method label inside ontology_links (used by retriever).")

    # KG artifact (path-based)
    p.add_argument("--kg-path", type=str, required=True, help="Path to KG artifact JSON/JSONL (per-doc triples).")
    p.add_argument("--kg-max-triples", type=int, default=40)

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
    p.add_argument("--enable-web", action="store_true", help="Enable Internet tool (Wikipedia). OFF by default.")
    p.add_argument("--enable-wikidata", action="store_true", help="Enable Wikidata tool. OFF by default.")
    p.add_argument("--web-timeout-sec", type=float, default=3.0)
    p.add_argument("--web-max-chars", type=int, default=1500)

    # Prompt sizing
    p.add_argument("--max-sentences-in-prompt", type=int, default=None)
    p.add_argument("--include-ontology-structured", action="store_true", default=True)
    p.add_argument("--no-ontology-structured", action="store_false", dest="include_ontology_structured")
    p.add_argument("--include-ontology-ttl", action="store_true", default=False)

    # LLM budget
    p.add_argument("--max-llm-calls", type=int, default=1, help="Total LLM calls budget per doc (default 1).")

    args = p.parse_args()

    cfg = load_config(args.config)
    input_path = _resolve_input_path(cfg, args.dataset_key)
    processed_root = _resolve_processed_root(cfg)

    doc_types = _parse_csv_or_all(args.doc_types)
    shot_types = _parse_csv_or_all(args.shot_doc_types)

    # IMPORTANT: reuse your existing llm config section "agentic_hybrid"
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

    output_path = processed_root / f"{args.dataset_key}.langgraph_agentic_hybrid.{llm_config.backend}.jsonl"

    # DocRED rel_info (optional)
    docred_rel_info: Optional[Dict[str, str]] = None
    if "docred" in str(args.dataset_key).lower():
        try:
            docred_rel_info = _load_docred_rel_info(cfg)
            print(f"[langgraph_agentic_hybrid] Loaded DocRED rel_info with {len(docred_rel_info)} entries.")
        except Exception as e:
            print(f"[langgraph_agentic_hybrid] Warning: could not load DocRED rel_info.json: {e}")

    onto_file = Path(args.ontology_links_path)
    if not onto_file.exists():
        raise FileNotFoundError(f"Ontology-links file not found: {onto_file}")

    kg_file = Path(args.kg_path)
    if not kg_file.exists():
        raise FileNotFoundError(f"KG file not found: {kg_file}")

    print(f"[langgraph_agentic_hybrid] input={input_path}")
    print(f"[langgraph_agentic_hybrid] output={output_path}")
    print(f"[langgraph_agentic_hybrid] ontology_artifact={onto_file}")
    print(f"[langgraph_agentic_hybrid] kg_artifact={kg_file}")
    print(f"[langgraph_agentic_hybrid] predict doc-types={doc_types} skip={args.skip} limit={args.limit}")
    print(f"[langgraph_agentic_hybrid] backend={llm_config.backend} model={llm_config.model}")
    print(f"[langgraph_agentic_hybrid] planner_mode={args.planner_mode} max_llm_calls={args.max_llm_calls} web={args.enable_web} wikidata={args.enable_wikidata}")

    # Load doc maps
    docid_to_links = _load_jsonl_doc_map(onto_file, "ontology_links")
    docid_to_triples = _load_kg_doc_triples(kg_file)

    # Few-shot pool
    shot_dataset_key = args.shot_dataset_key or args.dataset_key
    shot_input_path = _resolve_input_path(cfg, shot_dataset_key)
    few_shots = _collect_few_shots(
        shot_input_path,
        shot_doc_types=shot_types,
        shot_num=args.shot_num,
        shot_skip=args.shot_skip,
        shot_limit=args.shot_limit,
    )
    if args.shot_num > 0:
        print(f"[langgraph_agentic_hybrid] few-shots collected: {len(few_shots)}")

    # Build ontology retriever (TTL path from cfg["ontology"][ontology_key])
    ttl_path = Path(cfg["ontology"][args.ontology_key])
    retriever = SubOntologyRetriever(
        ontology_key=args.ontology_key,
        ttl_path=ttl_path,
        include_unrestricted_properties=True,
        max_properties=120,
        max_classes=200,
        pick="candidates",
    )

    params = LangGraphAgenticHybridParams(
        include_ontology_structured=bool(args.include_ontology_structured),
        include_ontology_ttl=bool(args.include_ontology_ttl),
        kg_max_triples=int(args.kg_max_triples),
        max_sentences_in_prompt=args.max_sentences_in_prompt,
        planner_mode=str(args.planner_mode),
        planner_depth=int(args.planner_depth),
        planner_verbosity=str(args.planner_verbosity),
        enable_web=bool(args.enable_web),
        enable_wikidata=bool(args.enable_wikidata),
        max_llm_calls=int(args.max_llm_calls),
        web_timeout_sec=float(args.web_timeout_sec),
        web_max_chars=int(args.web_max_chars),
    )

    strategy = LangGraphAgenticHybridRelationStrategy(
        llm_config=llm_config,
        retriever=retriever,
        ontology_key=args.ontology_key,
        linking_method=args.ontology_method,
        params=params,
    )

    num_seen_after_type = 0
    num_pred = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"LangGraphAgenticHybrid on {args.dataset_key}", unit="doc"):
            line = line.strip()
            if not line:
                continue

            doc = json.loads(line)

            # type filter
            if not _doc_type_matches(doc, doc_types):
                continue

            # skip/limit after filter
            if num_seen_after_type < args.skip:
                num_seen_after_type += 1
                continue
            if args.limit is not None and (num_seen_after_type - args.skip) >= args.limit:
                break
            num_seen_after_type += 1

            doc_id = str(doc.get("document_id") or doc.get("id") or "")

            # inject ontology_links if missing
            if "ontology_links" not in doc and doc_id in docid_to_links:
                doc["ontology_links"] = docid_to_links[doc_id]

            # inject KG triples context
            triples = docid_to_triples.get(doc_id, [])
            doc["_kg_context"] = {"triples": triples}

            # schema
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

    print(f"[langgraph_agentic_hybrid] done. predicted={num_pred}")
    print(f"[langgraph_agentic_hybrid] wrote: {output_path}")


if __name__ == "__main__":
    main()
