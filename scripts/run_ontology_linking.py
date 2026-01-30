from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from tqdm import tqdm

from ragtree.core.config import load_config
from ragtree.ontologies.loader import OntologyIndex
from ragtree.ontologies.mapping import OntologyEntityLinker


def derive_output_path(input_path: Path, *, output_folder: Path, method: str, ontology_key: str) -> Path:
    stem = input_path.stem
    out_name = f"{stem}_olink_{method}_onto_{ontology_key}.jsonl"
    return output_folder / out_name


def resolve_paths_from_config(config_path: Optional[Path], dataset_key: str, ontology_key: str) -> Tuple[Dict[str, Any], Path, Path]:
    cfg = load_config(str(config_path) if config_path else None)

    ds_pre = cfg.get("datasets", {}).get("preprocessed", {})
    if dataset_key not in ds_pre:
        available = ", ".join(sorted(ds_pre.keys()))
        raise KeyError(f"Unknown dataset key '{dataset_key}'. Available: {available}")
    input_path = Path(ds_pre[dataset_key])

    ont_cfg = cfg.get("ontology", {})
    if ontology_key not in ont_cfg:
        available = ", ".join(sorted(ont_cfg.keys()))
        raise KeyError(f"Unknown ontology key '{ontology_key}'. Available: {available}")
    ontology_path = Path(ont_cfg[ontology_key])

    return cfg, input_path, ontology_path


def run_ontology_linking(
    input_jsonl: Path,
    output_jsonl: Path,
    ontology_ttl: Path,
    *,
    backend: str = "ollama",
    top_k: int = 3,
    min_score: float = 0.3,
    method: str = "llm_embedding",
    ontology_key: str = "unknown",
    legacy_format: bool = False,
) -> None:
    print(f"[run_ontology_linking] Loading ontology from {ontology_ttl} ...")
    ont_index = OntologyIndex.from_turtle(ontology_ttl)

    print(f"[run_ontology_linking] Building linker method={method} backend={backend} ...")
    linker = OntologyEntityLinker(
        ontology_index=ont_index,
        backend=backend,
        method=method,
        ontology_key=ontology_key,
    )

    in_place = input_jsonl.resolve() == output_jsonl.resolve()
    tmp_path: Optional[Path] = None
    if in_place:
        tmp_path = output_jsonl.with_suffix(output_jsonl.suffix + ".ontmp")
        target = tmp_path
        print(f"[run_ontology_linking] In-place mode: writing to temp {tmp_path} then replacing.")
    else:
        target = output_jsonl

    print(f"[run_ontology_linking] Reading {input_jsonl} and writing {target} ...")
    num_docs = 0

    with input_jsonl.open("r", encoding="utf-8") as fin, target.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin):
            line = line.strip()
            if not line:
                continue

            doc = json.loads(line)

            if legacy_format:
                links = linker.link_document_legacy(doc, top_k=top_k, min_score=min_score)
            else:
                links = linker.link_document(doc, top_k=top_k, min_score=min_score)

            doc["ontology_links"] = links
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            num_docs += 1

    if in_place and tmp_path is not None:
        tmp_path.replace(output_jsonl)
        print(f"[run_ontology_linking] Replaced {output_jsonl} with updated file.")

    print(f"[run_ontology_linking] Done. Processed {num_docs} documents.")


def main() -> None:
    p = argparse.ArgumentParser(description="Ontology linking: create a preprocessed variant JSONL that includes doc['ontology_links'].")

    p.add_argument("--config", type=Path, default=None, help="Path to configs/default.yaml (optional).")
    p.add_argument("--dataset-key", required=True, help="Key under datasets.preprocessed in default.yaml.")
    p.add_argument("--ontology-key", required=True, help="Key under ontology in default.yaml.")
    p.add_argument("--method", type=str, default=None, help="Ontology linking method key (from cfg['ontology_linking']). If omitted, uses default_method.")
    p.add_argument("--output", type=Path, default=None, help="Optional output JSONL file. If omitted, derives a new filename in data/preprocessed/.")
    p.add_argument("--in-place", action="store_true", help="If set, overwrite the input preprocessed JSONL in-place (NOT recommended).")
    p.add_argument("--backend", type=str, default="ollama", choices=["ollama", "openrouter", "vllm"], help="Embedding backend.")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--min-score", type=float, default=0.3)
    p.add_argument("--legacy-format", action="store_true", help="If set, writes ontology_links in legacy format (ent_id -> list[{concept_uri,label,score}]).")

    args = p.parse_args()

    cfg, input_path, ontology_path = resolve_paths_from_config(args.config, args.dataset_key, args.ontology_key)

    ol_cfg = cfg.get("ontology_linking", {}) or {}
    method = args.method or ol_cfg.get("default_method") or "llm_embedding"

    output_folder = Path(cfg.get("paths", {}).get("data_preprocessed", "data/preprocessed"))
    if args.in_place:
        output_path = input_path
    elif args.output is not None:
        output_path = args.output
    else:
        output_path = derive_output_path(input_path, output_folder=output_folder, method=method, ontology_key=args.ontology_key)

    print(f"[config] dataset-key={args.dataset_key} -> input={input_path}")
    print(f"[config] ontology-key={args.ontology_key} -> ontology={ontology_path}")
    print(f"[config] method={method}")
    print(f"[config] output={output_path}")

    run_ontology_linking(
        input_jsonl=input_path,
        output_jsonl=output_path,
        ontology_ttl=ontology_path,
        backend=args.backend,
        top_k=args.top_k,
        min_score=args.min_score,
        method=method,
        ontology_key=args.ontology_key,
        legacy_format=args.legacy_format,
    )

if __name__ == "__main__":
    main()
