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
        raise ValueError(f"No model defined for backend '{backend_name}' and no --model override given.")

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Single-LLM baseline for relation extraction, without RAG or ontology. "
            "Reads a preprocessed JSONL and writes the same JSONL with an added "
            "'pred_relations' field mirroring the 'relations' structure."
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
        help="Key under datasets.preprocessed in the config (e.g. 'maven_ere').",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="LLM backend to use (e.g. 'ollama', 'openrouter'). If omitted, use llm.baseline.default_backend.",
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
            "(e.g. 'CAUSE,PRECONDITION'). "
            "If omitted, relation types are inferred from doc['relations'] or "
            f"fall back to '{DEFAULT_FALLBACK_RELATION_TYPE}'."
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
            pred_relations = strategy.predict_relations(
                doc,
                relation_types=rel_types_for_doc,
            )

            # Attach predictions using the same structure as 'relations'
            doc["pred_relations"] = pred_relations

            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            num_docs += 1

    print(f"[baseline] Done. Processed {num_docs} documents.")


if __name__ == "__main__":
    main()
