# ragtree/processing/orchestrators/relations_runner.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from tqdm import tqdm

from ragtree.core.config import load_config
from ragtree.processing.rag.base_strategy import BaseRelationStrategy, LLMBackendConfig
from ragtree.processing.rag.strategies.baseline_relations import (
    DEFAULT_FALLBACK_RELATION_TYPE,
)

DocTypeFilter = Union[str, Sequence[str]]


@dataclass
class RunnerLLMSections:
    """
    Helper to specify where to find LLM + prompt config in default.yaml.
    """
    llm_section: str = "baseline"        # under cfg["llm"][llm_section]
    prompt_section: str = "baseline"     # under cfg["prompts"][prompt_section]
    system_prompt_key: str = "causal_relations"


@dataclass
class PreparedContext:
    """
    Result of an optional experiment-specific preparation step.

    - strategy_kwargs: passed to the Strategy constructor
    - predict_kwargs: passed to each call to `strategy.predict_relations`
    """
    strategy_kwargs: Dict[str, Any]
    predict_kwargs: Dict[str, Any]


def _resolve_paths_and_config(
    config_path: Optional[Path],
    dataset_key: str,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """
    Resolve:
      - input JSONL (datasets.preprocessed[dataset_key])
      - output root directory (paths.data_processed)
      - full config dict
    """
    cfg = load_config(config_path)

    try:
        ds_pre = cfg["datasets"]["preprocessed"]
    except KeyError as e:
        raise KeyError(f"Config missing 'datasets.preprocessed' section: {e}")

    if dataset_key in ds_pre:
        input_path = Path(ds_pre[dataset_key])
    else:
        # Fallback: treat dataset_key as a filename stem living under paths.data_preprocessed
        try:
            pre_root = Path(cfg["paths"]["data_preprocessed"])
        except KeyError as e:
            raise KeyError(
                f"Unknown dataset key '{dataset_key}' and config missing paths.data_preprocessed to resolve it."
            ) from e

        candidate = pre_root / f"{dataset_key}.jsonl"
        if not candidate.exists():
            available = ", ".join(sorted(ds_pre.keys()))
            raise KeyError(
                f"Unknown dataset key '{dataset_key}'. Available preprocessed datasets: {available}. "
                f"Also tried file: {candidate} (not found)."
            )
        input_path = candidate

    try:
        processed_root = Path(cfg["paths"]["data_processed"])
    except KeyError as e:
        raise KeyError(f"Config missing 'paths.data_processed' section: {e}")

    processed_root.mkdir(parents=True, exist_ok=True)
    return input_path, processed_root, cfg


def _build_llm_config_from_yaml(
    cfg: Dict[str, Any],
    *,
    sections: RunnerLLMSections,
    backend_override: Optional[str],
    model_override: Optional[str],
) -> LLMBackendConfig:
    """
    Construct LLMBackendConfig from default.yaml and optional CLI overrides.

    Expected structure (for llm_section='baseline'):

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
        llm_cfg = cfg["llm"][sections.llm_section]
    except KeyError as e:
        raise KeyError(f"Config missing 'llm.{sections.llm_section}' section: {e}")

    default_backend: str = llm_cfg.get("default_backend", "ollama")
    backend_name = backend_override or default_backend

    try:
        backend_cfg = llm_cfg["backends"][backend_name]
    except KeyError as e:
        raise KeyError(
            f"Config missing backends entry for backend '{backend_name}' in "
            f"'llm.{sections.llm_section}': {e}"
        )

    model_name = model_override or backend_cfg.get("model")
    if not model_name:
        raise ValueError(
            f"No model defined for backend '{backend_name}' in 'llm.{sections.llm_section}' "
            f"and no --model override given."
        )

    temperature = float(backend_cfg.get("temperature", 0.0))
    max_tokens = int(backend_cfg.get("max_tokens", 512))

    system_prompt_key = llm_cfg.get("system_prompt_key", sections.system_prompt_key)

    try:
        system_prompt = cfg["prompts"][sections.prompt_section][system_prompt_key]
    except KeyError as e:
        raise KeyError(
            f"Config missing prompts.{sections.prompt_section}['{system_prompt_key}']: {e}"
        )

    return LLMBackendConfig(
        backend=backend_name,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )


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

    return rel_info


def _determine_relation_types_for_doc(
    doc: Dict[str, Any],
    cli_relation_types: Optional[Sequence[str]],
) -> List[str]:
    """
    Decide which relation types to use for a single doc.

    Priority:
      1. If CLI list is provided (e.g. --relation-types CAUSE,PRECONDITION) -> use those.
      2. Else if doc["relations"] exists and is a dict -> use its keys (even if lists are empty).
      3. Else -> [DEFAULT_FALLBACK_RELATION_TYPE].
    """
    if cli_relation_types:
        return list(cli_relation_types)

    rels = doc.get("relations")
    if isinstance(rels, dict) and rels:
        return list(rels.keys())

    return [DEFAULT_FALLBACK_RELATION_TYPE]


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


def _doc_type_matches(doc: Dict[str, Any], doc_type_filter: DocTypeFilter) -> bool:
    """
    Check if doc['type'] matches the filter.

    - If filter is 'all' (string) -> always True.
    - If filter is a string -> doc['type'] must equal that string.
    - If filter is a sequence -> doc['type'] must be in that collection.
    """
    if isinstance(doc_type_filter, str):
        if doc_type_filter == "all":
            return True
        return doc.get("type") == doc_type_filter

    # sequence of types
    allowed = set(doc_type_filter)
    return doc.get("type") in allowed


def _normalize_predict_kwargs(
    obj: Optional[PreparedContext],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if obj is None:
        return {}, {}
    return obj.strategy_kwargs, obj.predict_kwargs


def run_relation_experiment(
    strategy_cls: Type[BaseRelationStrategy],
    *,
    config_path: Optional[Path],
    dataset_key: str,
    backend: Optional[str],
    model: Optional[str],
    cli_relation_types: Optional[List[str]],
    output_format: str,
    doc_type_filter: DocTypeFilter,
    skip: int = 0,
    limit: Optional[int] = None,
    sections: Optional[RunnerLLMSections] = None,
    prepare_context_fn: Optional[
        Callable[[Path, Dict[str, Any]], PreparedContext]
    ] = None,
) -> None:
    """
    Generic runner for relation extraction experiments.

    Parameters
    ----------
    strategy_cls : subclass of BaseRelationStrategy
        Strategy to instantiate (e.g. BaselineRelationStrategy, ICLRelationStrategy, ...)

    config_path : optional Path
        Path to default.yaml (or similar). If omitted, load_config() should resolve it.

    dataset_key : str
        Key in cfg["datasets"]["preprocessed"].

    backend, model : str | None
        Overrides for backend/model from config.

    cli_relation_types : Optional[List[str]]
        If provided, fixed list of relation types to use for all documents.

    output_format : {"full", "pred-only"}
        Whether to keep full documents or only {document_id, pred_relations}.

    doc_type_filter : "all" | str | Sequence[str]
        Filter on doc["type"]. If "all", process everything.

    sections : RunnerLLMSections
        Specifies which llm/prompt sections to use in the config.

    prepare_context_fn : callable
        Optional function (input_path, cfg) -> PreparedContext.
        Used e.g. for ICL few-shot selection, ontology loading, etc.
    """
    if sections is None:
        sections = RunnerLLMSections()

    input_path, processed_root, cfg = _resolve_paths_and_config(
        config_path=config_path,
        dataset_key=dataset_key,
    )

    llm_config = _build_llm_config_from_yaml(
        cfg=cfg,
        sections=sections,
        backend_override=backend,
        model_override=model,
    )

    # Decide a default output path that includes method label (= llm_section) and backend
    method_label = sections.llm_section
    output_filename = f"{dataset_key}.{method_label}.{llm_config.backend}.jsonl"
    output_path = processed_root / output_filename

    print(f"[runner] dataset-key={dataset_key}")
    print(f"[runner] method={method_label}")
    print(f"[runner] backend={llm_config.backend}, model={llm_config.model}")
    print(f"[runner] input={input_path}")
    print(f"[runner] output={output_path}")
    print(f"[runner] output-format={output_format}")
    print(f"[runner] doc-type-filter={doc_type_filter}")
    print(f"[runner] skip={skip}, limit={limit}")


    # Load DocRED rel_info if relevant
    docred_rel_info: Optional[Dict[str, str]] = None
    if "docred_causal" in dataset_key.lower():
        try:
            docred_rel_info = _load_docred_rel_info(cfg)
            print(f"[runner] Loaded DocRED rel_info with {len(docred_rel_info)} entries.")
        except Exception as e:
            print(f"[runner] Warning: could not load DocRED rel_info.json: {e}")
            docred_rel_info = None

    # Optional experiment-specific context (few-shots, ontology, KG client, ...)
    prepared = prepare_context_fn(input_path, cfg) if prepare_context_fn is not None else None
    strategy_kwargs, predict_kwargs = _normalize_predict_kwargs(prepared)

    strategy = strategy_cls(llm_config=llm_config, **strategy_kwargs)

    num_docs = 0
    num_skipped_type = 0
    
    seen_after_type_filter = 0   # counts docs that pass doc-type filter
    written = 0                 # counts docs actually processed/written


    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        for line in tqdm(
            fin,
            desc=f"Running {method_label} on {dataset_key}",
            unit="doc",
        ):
            line = line.strip()
            if not line:
                continue

            doc = json.loads(line)

            # Filter on doc['type'] if requested
            if not _doc_type_matches(doc, doc_type_filter):
                num_skipped_type += 1
                continue

            # After type filter: apply skip/limit
            seen_after_type_filter += 1

            if skip and seen_after_type_filter <= skip:
                continue

            if limit is not None and written >= limit:
                break

            # Determine relation types for this doc
            relation_types_for_doc = _determine_relation_types_for_doc(
                doc,
                cli_relation_types=cli_relation_types,
            )

            # DocRED augmentation
            if (
                docred_rel_info is not None
                and "docred_causal" in dataset_key.lower()
            ):
                relation_types_for_doc = _augment_docred_relation_types(
                    relation_types_for_doc,
                    docred_rel_info,
                )

            # Predict relations with the final relation_types_for_doc
            pred_relations = strategy.predict_relations(
                doc,
                relation_types=relation_types_for_doc,
                **predict_kwargs,
            )

            # Decide what to write based on output-format
            if output_format == "pred-only":
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
            written += 1

    print(f"[runner] Done. Processed {written} documents.")
    print(f"[runner] docs_after_type_filter={seen_after_type_filter}, skip={skip}, limit={limit}")
    if isinstance(doc_type_filter, str):
        if doc_type_filter != "all":
            print(f"[runner] Skipped {num_skipped_type} documents due to doc-type filter.")
    else:
        print(f"[runner] Skipped {num_skipped_type} documents due to doc-type filter.")


__all__ = [
    "RunnerLLMSections",
    "PreparedContext",
    "run_relation_experiment",
]
