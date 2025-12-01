# scripts/run_single_llm_baseline.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence

from tqdm import tqdm

from ragtree.core.config import load_config
from ragtree.processing.rag.strategies.baseline_relations import (
    BaselineRelationStrategy,
    LLMBackendConfig,
    DEFAULT_FALLBACK_RELATION_TYPE,
)


def _resolve_paths_and_config(
    config_path: Optional[Path],
    dataset_key: str,
) -> tuple[Path, Path, Dict[str, Any]]:
    """
    Resolve:
      - input JSONL (datasets.preprocessed[dataset_key])
      - output JSONL (paths.data_processed / f"{dataset_key}.baseline.<backend>.jsonl")
      - full config dict
    """
    cfg = load_config(config_path)

    try:
        ds_pre = cfg["datasets"]["preprocessed"]
    except KeyError as e:
        raise KeyError(f"Config missing 'datasets.preprocessed' section: {e}")

    if dataset_key not in ds_pre:
        available = ", ".join(sorted(ds_pre.keys()))
        raise KeyError(
            f"Unknown dataset key '{dataset_key}'. "
            f"Available preprocessed datasets: {available}"
        )

    input_path = Path(ds_pre[dataset_key])

    try:
        processed_root = Path(cfg["paths"]["data_processed"])
    except KeyError as e:
        raise KeyError(f"Config missing 'paths.data_processed' section: {e}")

    processed_root.mkdir(parents=True, exist_ok=True)

    # The backend is needed to form the default output name,
    # but we don't know it yet; output filename will be finalized in main().
    return input_path, processed_root, cfg  # type: ignore[return-value]


def _build_llm_config_from_yaml(
    cfg: Dict[str, Any],
    backend: Optional[str],
    model: Optional[str],
) -> LLMBackendConfig:
    """
    Construct LLMBackendConfig from default.yaml and optional CLI overrides.
    Expects structure:

    llm:
      baseline:
        default_backend: "ollama"
        backends:
          ollama:
            model: "qwen2.5:3b"
            temperature: 0.0
            max_tokens: 512
          openrouter:
            model: "gpt-4.1-mini"
            temperature: 0.0
            max_tokens: 512
        system_prompt_key: "causal_relations"

    prompts:
      baseline:
        causal_relations: "..."
    """
    try:
        llm_cfg = cfg["llm"]["baseline"]
    except KeyError as e:
        raise KeyError(f"Config missing 'llm.baseline' section: {e}")

    default_backend: str = llm_cfg.get("default_backend", "ollama")
    backend_name = backend or default_backend

    try:
        backend_cfg = llm_cfg["backends"][backend_name]
    except KeyError as e:
        raise KeyError(
            f"Config missing backends entry for backend '{backend_name}' in 'llm.baseline': {e}"
        )

    model_name = model or backend_cfg.get("model")
    if not model_name:
        raise ValueError(
            f"No model defined for backend '{backend_name}' and no --model override given."
        )

    temperature = float(backend_cfg.get("temperature", 0.0))
    max_tokens = int(backend_cfg.get("max_tokens", 512))

    system_prompt_key = llm_cfg.get("system_prompt_key", "causal_relations")

    try:
        system_prompt = cfg["prompts"]["baseline"][system_prompt_key]
    except KeyError as e:
        raise KeyError(
            f"Config missing prompts.baseline['{system_prompt_key}']: {e}"
        )

    return LLMBackendConfig(
        backend=backend_name,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )


def _determine_relation_types_for_doc(
    doc: Dict[str, Any],
    cli_relation_types: Optional[Sequence[str]],
) -> List[str]:
    """
    Decide which relation types to use for a single doc.

    Priority:
      1. If CLI list provided (e.g. --relation-types CAUSE,PRECONDITION) -> use those.
      2. Else if doc["relations"] exists and is a dict -> use its keys (even if lists are empty).
      3. Else -> [DEFAULT_FALLBACK_RELATION_TYPE].
    """
    if cli_relation_types:
        return list(cli_relation_types)

    rels = doc.get("relations")
    if isinstance(rels, dict) and rels:
        return list(rels.keys())

    return [DEFAULT_FALLBACK_RELATION_TYPE]


# --------- NEW: DocRED-specific helper to load rel_info.json ---------

def _load_docred_rel_info(cfg: Dict[str, Any]) -> Dict[str, str]:
    """
    Load DocRED rel_info.json from datasets.raw.DocRED configured in default.yaml.

    Expected config fragment:
      datasets:
        raw:
          DocRED: data/raw/DocRED/DocRED

    Returns: dict like {"P159": "headquarters location", ...}
    """
    try:
        raw_cfg = cfg["datasets"]["raw"]
    except KeyError as e:
        raise KeyError(f"Config missing 'datasets.raw' section: {e}")

    docred_path: Optional[Path] = None
    for key, val in raw_cfg.items():
        if key.lower() == "docred":
            docred_path = Path(val)
            break

    if docred_path is None:
        raise KeyError(
            "Could not find a 'DocRED' entry under datasets.raw in the config."
        )

    rel_info_path = docred_path / "rel_info.json"
    if not rel_info_path.exists():
        raise FileNotFoundError(
            f"DocRED rel_info.json not found at: {rel_info_path}"
        )

    with rel_info_path.open("r", encoding="utf-8") as f:
        rel_info = json.load(f)

    # rel_info is expected to be { "P159": "headquarters location", ... }
    return rel_info


def _augment_docred_relation_types(
    relation_types_for_doc: List[str],
    docred_rel_info: Dict[str, str],
) -> List[str]:
    """
    Produce a full list of DocRED-style augmented relations:
    "PID : description" for every entry in docred_rel_info,
    and ensure all relations from relation_types_for_doc are included,
    without duplicates.
    """
    augmented = set()

    # Add every official relation type from DocRED mapping
    for pid, desc in docred_rel_info.items():
        augmented.add(f"{pid} : {desc}")

    # Ensure all relations from the document are included too
    for rt in relation_types_for_doc:
        if " : " in rt:
            augmented.add(rt)
        else:
            desc = docred_rel_info.get(rt)
            if desc:
                augmented.add(f"{rt} : {desc}")
            else:
                augmented.add(rt)

    return sorted(augmented)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Single-LLM baseline for relation extraction, without RAG or ontology. "
            "Reads a preprocessed JSONL and writes predictions under 'pred_relations'."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to default.yaml (or similar). If omitted, load_config() should resolve it.",
    )

    parser.add_argument(
        "--dataset-key",
        required=True,
        help="Key under datasets.preprocessed in the config (e.g. 'maven_ere', 'docred_causal').",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help=(
            "LLM backend to use (e.g. 'vllm', 'ollama', 'openrouter'). "
            "If omitted, use llm.baseline.default_backend."
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model override for the chosen backend.",
    )

    parser.add_argument(
        "--relation-types",
        type=str,
        default=None,
        help=(
            "Optional comma-separated list of relation types to enforce "
            "(e.g. 'CAUSE,PRECONDITION' or 'P17,P27'). "
            "If omitted, relation types are inferred from doc['relations'] or "
            f"fall back to '{DEFAULT_FALLBACK_RELATION_TYPE}'."
        ),
    )

    parser.add_argument(
        "--output-format",
        choices=["full", "pred-only"],
        default="full",
        help=(
            "Control JSONL output structure:\n"
            "  - 'full': keep the full original document and add a 'pred_relations' field.\n"
            "  - 'pred-only': write only {document_id, pred_relations} per line."
        ),
    )

    args = parser.parse_args()

    input_path, processed_root, cfg = _resolve_paths_and_config(
        config_path=args.config,
        dataset_key=args.dataset_key,
    )

    llm_config = _build_llm_config_from_yaml(
        cfg=cfg,
        backend=args.backend,
        model=args.model,
    )

    # Decide a default output path that includes backend name for clarity
    output_filename = f"{args.dataset_key}.baseline.{llm_config.backend}.jsonl"
    output_path = processed_root / output_filename

    print(f"[baseline] dataset-key={args.dataset_key}")
    print(f"[baseline] backend={llm_config.backend}, model={llm_config.model}")
    print(f"[baseline] input={input_path}")
    print(f"[baseline] output={output_path}")
    print(f"[baseline] output-format={args.output_format}")

    # Parse CLI relation types list if provided
    cli_relation_types: Optional[List[str]] = None
    if args.relation_types:
        cli_relation_types = [
            rt.strip()
            for rt in args.relation_types.split(",")
            if rt.strip()
        ]
        if not cli_relation_types:
            cli_relation_types = None

    # --------- NEW: load DocRED rel_info if this is a DocRED-based dataset ---------
    docred_rel_info: Optional[Dict[str, str]] = None
    if "docred_causal" in args.dataset_key.lower():
        try:
            docred_rel_info = _load_docred_rel_info(cfg)
            print(f"[baseline] Loaded DocRED rel_info with {len(docred_rel_info)} entries.")
        except Exception as e:
            print(f"[baseline] Warning: could not load DocRED rel_info.json: {e}")
            docred_rel_info = None
    
    strategy = BaselineRelationStrategy(llm_config=llm_config)

    num_docs = 0
    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin):
            line = line.strip()
            if not line:
                continue

            doc = json.loads(line)
            rel_types_for_doc = _determine_relation_types_for_doc(
                doc,
                cli_relation_types=cli_relation_types,
            )

            # --------- NEW: augment DocRED relation types as "P159 : headquarters location" ---------
            if docred_rel_info is not None:
                rel_types_for_doc = _augment_docred_relation_types(
                    rel_types_for_doc,
                    docred_rel_info,
                )

            # Pseudocode logic
            if "docred_causal" in args.dataset_key.lower() and docred_rel_info is not None:
                pred_relations = strategy.predict_relations(doc, relation_types=rel_types_for_doc)
            else:
                rel_types = sorted(doc.get("relations", {}).keys())
                pred_relations = strategy.predict_relations(doc, relation_types=rel_types)
            
            # Decide what to write based on output-format
            if args.output_format == "pred-only":
                # Minimal object: keep identifier + predictions only
                out_obj: Dict[str, Any] = {
                    "document_id": doc.get("document_id") or doc.get("id"),
                    "pred_relations": pred_relations,
                }
            else:
                # Default: keep full original doc and just add predictions
                doc["pred_relations"] = pred_relations
                out_obj = doc

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            num_docs += 1

    print(f"[baseline] Done. Processed {num_docs} documents.")


if __name__ == "__main__":
    main()
