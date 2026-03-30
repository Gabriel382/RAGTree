from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from ragtree.core.config import load_config
from ragtree.processing.rag.strategies.marag_relations import (
    MARagRelationStrategy,
    MARAGParams,
)
from ragtree.ontologies.retrieval.subontology import SubOntologyRetriever


# ----------------------------
# Lightweight local config object
# ----------------------------
@dataclass
class MARAGLLMConfig:
    """
    Minimal LLM config object for MA-RAG.
    """
    backend: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    system_prompt: str = ""


# ----------------------------
# Helpers
# ----------------------------
def _parse_types(arg: str) -> Sequence[str] | str:
    """Parse comma-separated doc types or 'all'."""
    if not arg or arg == "all":
        return "all"
    items = [x.strip() for x in arg.split(",")]
    items = [x for x in items if x]
    return items or "all"


def _should_keep_doc(doc: Dict[str, Any], doc_types: Sequence[str] | str) -> bool:
    """Check whether a document matches the requested doc types."""
    if doc_types == "all":
        return True
    return doc.get("type") in set(doc_types)


def _iter_jsonl(path: Path):
    """Yield JSON objects from a JSONL file."""
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_ontology_links_map(path: Path) -> Dict[str, Any]:
    """
    Load ontology links by document ID from JSONL.
    """
    out: Dict[str, Any] = {}
    for obj in _iter_jsonl(path):
        doc_id = obj.get("document_id") or obj.get("id")
        if not doc_id:
            continue
        links = obj.get("ontology_links")
        if isinstance(links, dict):
            out[str(doc_id)] = links
    return out


def _load_kg_doc_triples(path: Path) -> Dict[str, List[List[str]]]:
    """
    Load document-level triples from JSON or JSONL.

    Supported per-doc shapes:
      - {"document_id": ..., "triples": [...]}
      - {"document_id": ..., "kg_triples": [...]}
      - {"document_id": ..., "edges": [...]}
      - {"docs": {...}}
      - {"doc_id": {"triples": [...]}, ...}
    """
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return {}

    out: Dict[str, List[List[str]]] = {}

    def normalize_triples(triples: Any) -> List[List[str]]:
        kept: List[List[str]] = []
        if not isinstance(triples, list):
            return kept

        for t in triples:
            if isinstance(t, (list, tuple)) and len(t) >= 3:
                kept.append([str(t[0]), str(t[1]), str(t[2])])
            elif isinstance(t, dict):
                h = t.get("h") or t.get("head")
                r = t.get("r") or t.get("rel") or t.get("relation")
                tail = t.get("t") or t.get("tail")
                if h is not None and r is not None and tail is not None:
                    kept.append([str(h), str(r), str(tail)])
        return kept

    def ingest_obj(obj: Any) -> None:
        if not isinstance(obj, dict):
            return

        doc_id = obj.get("document_id") or obj.get("id")
        if not doc_id:
            return

        triples = (
            obj.get("triples")
            or obj.get("kg_triples")
            or obj.get("edges")
            or obj.get("_kg_context", {}).get("triples")
            or []
        )
        norm = normalize_triples(triples)
        if norm:
            out[str(doc_id)] = norm

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

        if isinstance(root, dict) and isinstance(root.get("docs"), dict):
            for doc_id, payload in root["docs"].items():
                if isinstance(payload, dict):
                    triples = payload.get("triples") or payload.get("kg_triples") or payload.get("edges") or []
                    norm = normalize_triples(triples)
                    if norm:
                        out[str(doc_id)] = norm
            if out:
                return out

        if isinstance(root, dict) and ("document_id" in root or "id" in root):
            ingest_obj(root)
            if out:
                return out

        if isinstance(root, dict):
            for k, v in root.items():
                if isinstance(v, dict):
                    triples = v.get("triples") or v.get("kg_triples") or v.get("edges") or []
                    norm = normalize_triples(triples)
                    if norm:
                        out[str(k)] = norm
                elif isinstance(v, list):
                    norm = normalize_triples(v)
                    if norm:
                        out[str(k)] = norm
            if out:
                return out

        if isinstance(root, dict) and isinstance(root.get("graph"), dict):
            raise ValueError(
                f"KG file '{path}' appears to be a corpus-level graph export "
                f"and not a document_id -> triples artifact."
            )

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

    if out:
        return out

    if first_bad:
        ln, preview = first_bad
        raise ValueError(
            f"KG file '{path}' could not be parsed as JSON or JSONL doc-triples. "
            f"First invalid JSONL line #{ln}: {preview}"
        )

    raise ValueError(
        f"KG file '{path}' could not be parsed into document_id -> triples."
    )


def _build_llm_config(
    cfg: Dict[str, Any],
    *,
    llm_section: str,
    backend_override: Optional[str],
    model_override: Optional[str],
) -> Tuple[MARAGLLMConfig, str]:
    """
    Build a minimal LLM config from YAML.
    """
    llm_cfg = cfg["llm"][llm_section]
    default_backend = llm_cfg.get("default_backend", "ollama")
    backend_name = backend_override or default_backend

    backends = llm_cfg.get("backends", {})
    if backend_name not in backends:
        raise KeyError(f"Unknown backend '{backend_name}' in llm.{llm_section}.backends")

    b = backends[backend_name]
    model = model_override or b.get("model")
    if not model:
        raise KeyError(f"Missing model for backend '{backend_name}' in llm.{llm_section}.backends")

    system_prompt_key = llm_cfg.get("system_prompt_key", "marag_docre")
    prompt_bank = cfg.get("prompts", {}).get(llm_section, {}) or {}
    system_prompt = prompt_bank.get(system_prompt_key)
    if not system_prompt:
        raise KeyError(f"Missing prompts.{llm_section}.{system_prompt_key} in config")

    llm_config = MARAGLLMConfig(
        backend=backend_name,
        model=model,
        temperature=float(b.get("temperature", 0.0)),
        max_tokens=int(b.get("max_tokens", 1024)),
        base_url=b.get("base_url"),
        api_key=b.get("api_key"),
        system_prompt=system_prompt,
    )
    return llm_config, backend_name


def _collect_few_shots(
    input_path: Path,
    *,
    shot_type: str,
    shot_num: int,
    shot_doc_types: Sequence[str] | str,
    shot_skip: int,
    shot_limit: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Collect few-shot examples from the input dataset.
    """
    if shot_num <= 0:
        return []

    shots: List[Dict[str, Any]] = []
    skipped = 0
    taken = 0

    for doc in _iter_jsonl(input_path):
        if not _should_keep_doc(doc, shot_doc_types):
            continue
        if doc.get("type") != shot_type:
            continue

        rels = doc.get("relations")
        if not isinstance(rels, dict) or not rels:
            continue

        if skipped < shot_skip:
            skipped += 1
            continue
        if shot_limit is not None and taken >= shot_limit:
            break

        shots.append(doc)
        taken += 1
        if len(shots) >= shot_num:
            break

    return shots


def main() -> None:
    """
    CLI entry point for recall-oriented MA-RAG.
    """
    p = argparse.ArgumentParser(description="Run recall-oriented MA-RAG relation extraction.")

    p.add_argument("--config", type=Path, default=None, help="Path to default.yaml (optional).")
    p.add_argument("--dataset-key", required=True, help="Key in cfg['datasets']['preprocessed'].")

    p.add_argument("--backend", type=str, default=None, help="Override LLM backend.")
    p.add_argument("--model", type=str, default=None, help="Override LLM model.")

    p.add_argument("--doc-types", type=str, default="all", help="all or comma-separated (dev,test,train).")
    p.add_argument("--skip", type=int, default=0, help="Skip first N docs AFTER doc-type filtering.")
    p.add_argument("--limit", type=int, default=None, help="Process at most K docs AFTER doc-type filtering.")

    # Artifact paths
    p.add_argument("--ontology-links-path", type=Path, default=None, help="JSONL with ontology_links (optional).")
    p.add_argument("--kg-path", type=Path, default=None, help="KG file path (.json or .jsonl).")

    # Ontology retriever config
    p.add_argument("--ontology-key", type=str, default=None, help="Key under cfg['ontology'] for TTL path.")
    p.add_argument("--linking-method", type=str, default="llm_embedding", help="Ontology linking method label.")

    # MA-RAG behavior
    p.add_argument("--max-llm-calls", type=int, default=3, help="Total allowed LLM calls per doc.")
    p.add_argument("--enable-web", action="store_true", help="Allow web context tool.")
    p.add_argument("--enable-wikidata", action="store_true", help="Allow Wikidata context tool.")
    p.add_argument("--kg-max-triples", type=int, default=40, help="Top-K KG triples in context.")
    p.add_argument("--max-sentences-in-prompt", type=int, default=None)
    p.add_argument("--no-gap-critic", action="store_true", help="Disable the recall-oriented gap critic.")
    p.add_argument("--no-verifier", action="store_true", help="Disable the light verifier.")
    p.add_argument("--output-format", choices=["full", "pred-only"], default="full")

    # Few-shot
    p.add_argument("--shot-num", type=int, default=0, help="0 disables few-shot.")
    p.add_argument("--shot-type", type=str, default="train", help="Which doc['type'] provides few-shots.")
    p.add_argument("--shot-doc-types", type=str, default="all", help="Doc-type filter for few-shot pool.")
    p.add_argument("--shot-skip", type=int, default=0)
    p.add_argument("--shot-limit", type=int, default=None)

    args = p.parse_args()

    cfg = load_config(args.config)

    # Resolve input JSONL path
    ds_pre = cfg["datasets"]["preprocessed"]
    if args.dataset_key not in ds_pre:
        available = ", ".join(sorted(ds_pre.keys()))
        raise KeyError(f"Unknown dataset key '{args.dataset_key}'. Available: {available}")

    input_path = Path(ds_pre[args.dataset_key])

    # Output path
    processed_root = Path(cfg["paths"]["data_processed"])
    processed_root.mkdir(parents=True, exist_ok=True)

    method_label = "marag"
    llm_config, backend_name = _build_llm_config(
        cfg,
        llm_section=method_label,
        backend_override=args.backend,
        model_override=args.model,
    )

    out_path = processed_root / f"{args.dataset_key}.{method_label}.{backend_name}.jsonl"

    doc_types = _parse_types(args.doc_types)
    shot_doc_types = _parse_types(args.shot_doc_types)

    # Load ontology links
    ontology_links_map: Dict[str, Any] = {}
    if args.ontology_links_path is not None:
        ontology_links_map = _load_ontology_links_map(args.ontology_links_path)

    # Load KG triples
    kg_triples_map: Dict[str, List[List[str]]] = {}
    if args.kg_path is not None:
        kg_triples_map = _load_kg_doc_triples(args.kg_path)

    # Ontology retriever
    retriever = None
    if args.ontology_key is not None:
        onto_paths = cfg.get("ontology", {}) or {}
        if args.ontology_key not in onto_paths:
            available = ", ".join(sorted(onto_paths.keys()))
            raise KeyError(f"Unknown ontology key '{args.ontology_key}'. Available: {available}")

        ttl_path = Path(onto_paths[args.ontology_key])
        retriever = SubOntologyRetriever(
            ontology_key=args.ontology_key,
            ttl_path=ttl_path,
            include_unrestricted_properties=True,
            max_properties=None,
            max_classes=None,
            pick="candidates",
        )

    params = MARAGParams(
        max_llm_calls=args.max_llm_calls,
        enable_ontology=True,
        enable_kg=True,
        enable_web=bool(args.enable_web),
        enable_wikidata=bool(args.enable_wikidata),
        enable_gap_critic=(not args.no_gap_critic),
        enable_verifier=(not args.no_verifier),
        kg_max_triples=int(args.kg_max_triples),
        max_sentences_in_prompt=args.max_sentences_in_prompt,
        keep_debug=True,
        verbose=False,
    )

    # Few-shot pool
    few_shots = _collect_few_shots(
        input_path,
        shot_type=args.shot_type,
        shot_num=int(args.shot_num),
        shot_doc_types=shot_doc_types,
        shot_skip=int(args.shot_skip),
        shot_limit=args.shot_limit,
    )

    # Pre-filter docs exactly like the good hybrid behavior
    all_filtered_docs = [doc for doc in _iter_jsonl(input_path) if _should_keep_doc(doc, doc_types)]
    filtered_docs = all_filtered_docs[args.skip:]
    if args.limit is not None:
        filtered_docs = filtered_docs[: args.limit]

    print(f"[marag] input={input_path}")
    print(f"[marag] output={out_path}")
    print(f"[marag] ontology_artifact={args.ontology_links_path}")
    print(f"[marag] kg_artifact={args.kg_path}")
    print(f"[marag] predict doc-types={doc_types} skip={args.skip} limit={args.limit}")
    print(f"[marag] backend={llm_config.backend} model={llm_config.model}")
    print(f"[marag] docs matching filter before skip/limit: {len(all_filtered_docs)}")
    print(f"[marag] docs to process after skip/limit: {len(filtered_docs)}")
    if args.shot_num > 0:
        print(f"[marag] few-shots collected: {len(few_shots)}")

    strat = MARagRelationStrategy(
        llm_config,
        params=params,
        subontology_retriever=retriever,
        linking_method=args.linking_method,
        ontology_links_by_docid=ontology_links_map,
        kg_triples_by_docid=kg_triples_map,
    )

    written = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for doc in tqdm(filtered_docs, desc=f"MA-RAG on {args.dataset_key}", unit="doc"):
            rel_types = (
                list(doc.get("relations", {}).keys())
                if isinstance(doc.get("relations"), dict) and doc.get("relations")
                else None
            )

            pred = strat.predict_relations(doc, relation_types=rel_types, few_shots=few_shots)

            if args.output_format == "pred-only":
                out_obj = {
                    "document_id": doc.get("document_id") or doc.get("id"),
                    "pred_relations": pred,
                }
            else:
                doc["pred_relations"] = pred
                out_obj = doc

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            written += 1

    print(f"[marag] done. predicted={written}")
    print(f"[marag] wrote: {out_path}")


if __name__ == "__main__":
    main()