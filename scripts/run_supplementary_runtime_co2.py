#!/usr/bin/env python3
"""
Supplementary runtime and CO2 estimator for RAGTree relation-extraction methods.

This script is intentionally additive: it does not modify any existing RAGTree file.
Place it under:

    scripts/run_supplementary_runtime_co2.py

It measures one eligible document for each method/dataset configuration used in the
processing notebooks, then extrapolates the measured one-document cost to the full
number of eligible documents found in the corresponding JSONL file.

Outputs are written under:

    data/suplementary_metrics/

The name uses "suplementary" because this is the path requested for the paper
artifacts.

Important scientific interpretation
-----------------------------------
The reported values are approximate cost indicators. They are not exact
measurements of the original full benchmark execution. The extrapolation assumes
that the measured document is representative of the other documents in the same
method/dataset condition.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:
    import yaml
except ImportError as exc:  # pragma: no cover - user environment guard
    raise SystemExit(
        "PyYAML is required because RAGTree uses YAML configuration files. "
        "Install it with: pip install pyyaml"
    ) from exc


DocTypeFilter = Union[str, Sequence[str]]


# ---------------------------------------------------------------------------
# Method/dataset specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MethodSpec:
    """One method/dataset run copied from the processing notebooks."""

    method_family: str
    method: str
    dataset: str
    script: str
    args: Tuple[str, ...]
    doc_filter_arg: str
    doc_filter_value: str
    run_mode: str = "script"  # "script" or "direct_llm"
    notebook_source: str = "notebook"
    note: str = ""

    @property
    def notebook_like_command(self) -> str:
        pieces = ["%run", f'"{self.script}"', *self.args]
        return " ".join(pieces)


def _t(*items: str) -> Tuple[str, ...]:
    """Small helper to keep specs readable."""

    return tuple(items)


def build_method_specs(include_inferred_baseline_eventstoryline: bool = True) -> List[MethodSpec]:
    """
    Return the method configurations used by the notebooks.

    The commands below follow the parameters found in:
      - processing_maven_ere.ipynb
      - ontology.ipynb
      - kg.ipynb
      - agentic.ipynb

    For the three LLM-only methods, the original scripts do not expose --skip and
    --limit. They are therefore executed through the same internal orchestrator
    with skip/limit supplied by this supplementary script. This preserves the
    notebook parameters while allowing one-document measurement without modifying
    existing files.
    """

    specs: List[MethodSpec] = []

    # ------------------------------------------------------------------
    # LLM-only baselines
    # ------------------------------------------------------------------
    baseline_datasets = [
        ("docred_causal", "dev", "processing_maven_ere.ipynb cell 9"),
        ("fincausal", "all", "processing_maven_ere.ipynb cell 4"),
        ("maven_ere", "all", "processing_maven_ere.ipynb cell 3/cell 8"),
        ("causalbank", "all", "processing_maven_ere.ipynb cell 6"),
    ]
    if include_inferred_baseline_eventstoryline:
        baseline_datasets.insert(
            1,
            (
                "eventstoryline",
                "all",
                "inferred from the baseline pattern because eventstoryline is present for CoT/ICL/RAG methods",
            ),
        )

    for dataset, doc_type, source in baseline_datasets:
        specs.append(
            MethodSpec(
                method_family="LLM-only",
                method="baseline",
                dataset=dataset,
                script="scripts/run_single_llm_baseline.py",
                args=_t(
                    "--dataset-key",
                    dataset,
                    "--backend",
                    "vllm",
                    "--doc-type",
                    doc_type,
                ),
                doc_filter_arg="--doc-type",
                doc_filter_value=doc_type,
                run_mode="direct_llm",
                notebook_source=source,
                note=(
                    "EventStoryLine baseline is inferred for method/dataset completeness. "
                    "Remove it with --no-inferred-baseline-eventstoryline if strict notebook-only execution is needed."
                    if dataset == "eventstoryline"
                    else ""
                ),
            )
        )

    for dataset, doc_type, source in [
        ("docred_causal", "dev", "processing_maven_ere.ipynb cell 9"),
        ("eventstoryline", "all", "processing_maven_ere.ipynb cell 11"),
        ("fincausal", "all", "processing_maven_ere.ipynb cell 11"),
        ("causalbank", "all", "processing_maven_ere.ipynb cell 11"),
        ("maven_ere", "all", "processing_maven_ere.ipynb cell 11"),
    ]:
        specs.append(
            MethodSpec(
                method_family="LLM-only",
                method="cot",
                dataset=dataset,
                script="scripts/run_cot_baseline.py",
                args=_t(
                    "--dataset-key",
                    dataset,
                    "--backend",
                    "vllm",
                    "--doc-type",
                    doc_type,
                ),
                doc_filter_arg="--doc-type",
                doc_filter_value=doc_type,
                run_mode="direct_llm",
                notebook_source=source,
            )
        )

    for dataset, train_type, predict_types, source in [
        ("docred_causal", "train_annotated", "dev", "processing_maven_ere.ipynb cell 9"),
        ("eventstoryline", "full", "all", "processing_maven_ere.ipynb cell 7"),
        ("fincausal", "train.csv", "all", "processing_maven_ere.ipynb cell 7"),
        ("maven_ere", "train", "all", "processing_maven_ere.ipynb cell 7/cell 8"),
        ("causalbank", "resulted_from", "all", "processing_maven_ere.ipynb cell 7"),
    ]:
        specs.append(
            MethodSpec(
                method_family="LLM-only",
                method="icl",
                dataset=dataset,
                script="scripts/run_icl_baseline.py",
                args=_t(
                    "--dataset-key",
                    dataset,
                    "--backend",
                    "vllm",
                    "--icl-train-type",
                    train_type,
                    "--icl-train-num",
                    "3",
                    "--icl-predict-types",
                    predict_types,
                ),
                doc_filter_arg="--icl-predict-types",
                doc_filter_value=predict_types,
                run_mode="direct_llm",
                notebook_source=source,
            )
        )

    # ------------------------------------------------------------------
    # Ontology-guided RAG
    # ------------------------------------------------------------------
    growlrag = [
        (
            "docred_causal",
            "docred_causal_olink_llm_embedding_onto_docredontology",
            "dev",
            "dev",
            "ontology.ipynb cell 8",
        ),
        (
            "eventstoryline",
            "eventstoryline_olink_llm_embedding_onto_owltime",
            "all",
            "all",
            "ontology.ipynb cell 10",
        ),
        (
            "causalbank",
            "causalbank_olink_llm_embedding_onto_wordnetfull",
            "all",
            "resulted_from",
            "ontology.ipynb cell 31",
        ),
        (
            "fincausal",
            "fincausal_olink_llm_embedding_onto_fibocoreplus",
            "all",
            "train.csv",
            "ontology.ipynb cell 41",
        ),
        (
            "maven_ere",
            "maven_ere_olink_llm_embedding_onto_EventKG",
            "all",
            "train.csv",
            "ontology.ipynb cell 51",
        ),
    ]
    for dataset, olink_key, doc_filter, shot_type, source in growlrag:
        specs.append(
            MethodSpec(
                method_family="Ontology-guided RAG",
                method="growlrag",
                dataset=dataset,
                script="scripts/run_growlrag_relations.py",
                args=_t(
                    "--dataset-key",
                    olink_key,
                    "--backend",
                    "vllm",
                    "--doc-type-filter",
                    doc_filter,
                    "--growlrag-shot-type",
                    shot_type,
                    "--growlrag-shot-num",
                    "3",
                ),
                doc_filter_arg="--doc-type-filter",
                doc_filter_value=doc_filter,
                notebook_source=source,
            )
        )

    ontology_specs = {
        "docred_causal": {
            "ontology_key": "docredontology",
            "links": "data/preprocessed/docred_causal_olink_llm_embedding_onto_docredontology.jsonl",
            "doc_types": "dev",
            "chunk_index": "data/indices/chunk_orag/docredontology/BAAI__bge-m3/hier_sizes=512-128_leaf=1_sha=36863800a0d8",
            "chunk_ontology_key": "docredontology",
            "shot_type": "train",
            "source_ograg": "ontology.ipynb cell 12",
            "source_chunk": "ontology.ipynb cell 16",
        },
        "eventstoryline": {
            "ontology_key": "owltime",
            "links": "data/preprocessed/eventstoryline_olink_llm_embedding_onto_owltime.jsonl",
            "doc_types": "all",
            "chunk_index": "data/indices/chunk_orag/owltime/BAAI__bge-m3/hier_sizes=512-128_leaf=1_sha=5e35260c5c36",
            "chunk_ontology_key": "owltime",
            "shot_type": "all",
            "source_ograg": "ontology.ipynb cell 23",
            "source_chunk": "ontology.ipynb cell 27",
        },
        "causalbank": {
            "ontology_key": "wordnetfull",
            "links": "data/preprocessed/causalbank_olink_llm_embedding_onto_wordnetfull.jsonl",
            "doc_types": "all",
            "chunk_index": "data/indices/chunk_orag/wordnet/BAAI__bge-m3/hier_sizes=512-128_leaf=1_sha=05764d42922d",
            "chunk_ontology_key": "wordnetfull",
            "shot_type": "all",
            "source_ograg": "ontology.ipynb cell 33",
            "source_chunk": "ontology.ipynb cell 37",
        },
        "fincausal": {
            "ontology_key": "fibocoreplus",
            "links": "data/preprocessed/fincausal_olink_llm_embedding_onto_fibocoreplus.jsonl",
            "doc_types": "all",
            "chunk_index": "data/indices/chunk_orag/fibocoreplus/BAAI__bge-m3/hier_sizes=512-128_leaf=1_sha=ceb6e240dbb6",
            "chunk_ontology_key": "fibocoreplus",
            "shot_type": "all",
            "source_ograg": "ontology.ipynb cell 43",
            "source_chunk": "ontology.ipynb cell 47",
        },
        "maven_ere": {
            "ontology_key": "EventKG",
            "links": "data/preprocessed/maven_ere_olink_llm_embedding_onto_EventKG.jsonl",
            "doc_types": "all",
            "chunk_index": "data/indices/chunk_orag/EventKG/BAAI__bge-m3/hier_sizes=512-128_leaf=1_sha=5b25e82e8316",
            "chunk_ontology_key": "EventKG",
            "shot_type": "all",
            "source_ograg": "ontology.ipynb cell 53",
            "source_chunk": "ontology.ipynb cell 57",
        },
    }

    for dataset, params in ontology_specs.items():
        doc_types = str(params["doc_types"])
        specs.append(
            MethodSpec(
                method_family="Ontology-guided RAG",
                method="ograg",
                dataset=dataset,
                script="scripts/run_ograg_relations.py",
                args=_t(
                    "--dataset-key",
                    dataset,
                    "--ontology-key",
                    str(params["ontology_key"]),
                    "--ontology-links-path",
                    str(params["links"]),
                    "--backend",
                    "vllm",
                    "--doc-types",
                    doc_types,
                ),
                doc_filter_arg="--doc-types",
                doc_filter_value=doc_types,
                notebook_source=str(params["source_ograg"]),
            )
        )

        specs.append(
            MethodSpec(
                method_family="Ontology-guided RAG",
                method="chunk_orag",
                dataset=dataset,
                script="scripts/run_chunk_orag_relations.py",
                args=_t(
                    "--dataset-key",
                    dataset,
                    "--ontology-key",
                    str(params["chunk_ontology_key"]),
                    "--index-dir",
                    str(params["chunk_index"]),
                    "--doc-type-filter",
                    doc_types,
                    "--shot-type",
                    str(params["shot_type"]),
                    "--shot-num",
                    "3",
                    "--device",
                    "cpu",
                ),
                doc_filter_arg="--doc-type-filter",
                doc_filter_value=doc_types,
                notebook_source=str(params["source_chunk"]),
            )
        )

    # ------------------------------------------------------------------
    # KG-based RAG
    # ------------------------------------------------------------------
    kg_specs = {
        "docred_causal": {
            "kg_path": "data/kg/docred_causal__types=train_annotated_skip=0_limit=None__kg.json",
            "doc_filter_kg": "dev",
            "doc_filter_other": "dev",
            "shot_type_triple": "dev",
            "shot_type_community": "train",
        },
        "eventstoryline": {
            "kg_path": "data/kg/eventstoryline__types=full_skip=0_limit=10__kg.json",
            "doc_filter_kg": "full",
            "doc_filter_other": "all",
            "shot_type_triple": "all",
            "shot_type_community": "all",
        },
        "fincausal": {
            "kg_path": "data/kg/fincausal__types=train.csv_skip=0_limit=None__kg.json",
            "doc_filter_kg": "all",
            "doc_filter_other": "all",
            "shot_type_triple": "all",
            "shot_type_community": "all",
        },
        "maven_ere": {
            "kg_path": "data/kg/maven_ere__types=train_skip=0_limit=None__kg.json",
            "doc_filter_kg": "all",
            "doc_filter_other": "all",
            "shot_type_triple": "all",
            "shot_type_community": "all",
        },
        "causalbank": {
            "kg_path": "data/kg/causalbank__types=resulted_from_skip=0_limit=None__kg.json",
            "doc_filter_kg": "all",
            "doc_filter_other": "all",
            "shot_type_triple": "all",
            "shot_type_community": "all",
        },
    }

    for dataset, params in kg_specs.items():
        specs.append(
            MethodSpec(
                method_family="KG-based RAG",
                method="kg_rag",
                dataset=dataset,
                script="scripts/run_kg_rag_relations.py",
                args=_t(
                    "--dataset-key",
                    dataset,
                    "--backend",
                    "vllm",
                    "--doc-type-filter",
                    str(params["doc_filter_kg"]),
                    "--kg-path",
                    str(params["kg_path"]),
                ),
                doc_filter_arg="--doc-type-filter",
                doc_filter_value=str(params["doc_filter_kg"]),
                notebook_source="kg.ipynb",
            )
        )
        specs.append(
            MethodSpec(
                method_family="KG-based RAG",
                method="triple_kg_rag",
                dataset=dataset,
                script="scripts/run_triple_kg_rag_relations.py",
                args=_t(
                    "--dataset-key",
                    dataset,
                    "--backend",
                    "vllm",
                    "--doc-type-filter",
                    str(params["doc_filter_other"]),
                    "--kg-max-hops",
                    "1",
                    "--kg-max-triples",
                    "120",
                    "--shot-type",
                    str(params["shot_type_triple"]),
                    "--shot-num",
                    "3",
                ),
                doc_filter_arg="--doc-type-filter",
                doc_filter_value=str(params["doc_filter_other"]),
                notebook_source="kg.ipynb",
            )
        )
        specs.append(
            MethodSpec(
                method_family="KG-based RAG",
                method="community_kgrag",
                dataset=dataset,
                script="scripts/run_community_kgrag_relations.py",
                args=_t(
                    "--dataset-key",
                    dataset,
                    "--communitykg-root",
                    "data/kg_community",
                    "--backend",
                    "vllm",
                    "--doc-type-filter",
                    str(params["doc_filter_other"]),
                    "--shot-type",
                    str(params["shot_type_community"]),
                    "--shot-num",
                    "3",
                    "--top-communities",
                    "3",
                    "--top-sentences",
                    "3",
                    "--max-ctx-chars",
                    "3000",
                ),
                doc_filter_arg="--doc-type-filter",
                doc_filter_value=str(params["doc_filter_other"]),
                notebook_source="kg.ipynb",
            )
        )

    # ------------------------------------------------------------------
    # Agentic RAG
    # ------------------------------------------------------------------
    agentic_common = {
        "docred_causal": {
            "doc_types": "dev",
            "ontology_key": "docredontology",
            "links": "data/preprocessed/docred_causal_olink_llm_embedding_onto_docredontology.jsonl",
            "kg_path": "data/kg/docred_causal__types=train_annotated_skip=0_limit=None__kg.json",
            "shot_doc_types": "all",
        },
        "eventstoryline": {
            "doc_types": "all",
            "ontology_key": "owltime",
            "links": "data/preprocessed/eventstoryline_olink_llm_embedding_onto_owltime.jsonl",
            "kg_path": "data/kg/eventstoryline__types=full_skip=0_limit=10__kg.json",
            "shot_doc_types": "all",
        },
        "fincausal": {
            "doc_types": "all",
            "ontology_key": "fibocoreplus",
            "links": "data/preprocessed/fincausal_olink_llm_embedding_onto_fibocoreplus.jsonl",
            "kg_path": "data/kg/fincausal__types=train.csv_skip=0_limit=None__kg.json",
            "shot_doc_types": "all",
        },
        "maven_ere": {
            "doc_types": "all",
            "ontology_key": "EventKG",
            "links": "data/preprocessed/maven_ere_olink_llm_embedding_onto_EventKG.jsonl",
            "kg_path": "data/kg/maven_ere__types=train_skip=0_limit=None__kg.json",
            "shot_doc_types": "all",
        },
        "causalbank": {
            "doc_types": "all",
            "ontology_key": "wordnetfull",
            "links": "data/preprocessed/causalbank_olink_llm_embedding_onto_wordnetfull.jsonl",
            "kg_path": "data/kg/causalbank__types=resulted_from_skip=0_limit=None__kg.json",
            "shot_doc_types": "all",
        },
    }

    for dataset, params in agentic_common.items():
        doc_types = str(params["doc_types"])
        specs.append(
            MethodSpec(
                method_family="Agentic RAG",
                method="langgraph_agentic_simple",
                dataset=dataset,
                script="scripts/run_langgraph_agentic_simple_relations.py",
                args=_t(
                    "--dataset-key",
                    dataset,
                    "--backend",
                    "vllm",
                    "--doc-types",
                    doc_types,
                    "--shot-num",
                    "3",
                    "--shot-doc-types",
                    "all",
                    "--max-llm-calls",
                    "2",
                ),
                doc_filter_arg="--doc-types",
                doc_filter_value=doc_types,
                notebook_source="agentic.ipynb",
            )
        )
        specs.append(
            MethodSpec(
                method_family="Agentic RAG",
                method="agentic_hybrid",
                dataset=dataset,
                script="scripts/run_agentic_hybrid_relations.py",
                args=_t(
                    "--dataset-key",
                    dataset,
                    "--backend",
                    "vllm",
                    "--doc-types",
                    doc_types,
                    "--ontology-links-path",
                    str(params["links"]),
                    "--ontology-key",
                    str(params["ontology_key"]),
                    "--ontology-method",
                    "llm_embedding",
                    "--kg-path",
                    str(params["kg_path"]),
                    "--shot-num",
                    "3",
                    "--shot-doc-types",
                    str(params["shot_doc_types"]),
                    "--max-llm-calls",
                    "1",
                ),
                doc_filter_arg="--doc-types",
                doc_filter_value=doc_types,
                notebook_source="agentic.ipynb",
            )
        )

    # These two methods are present only for DocRED in the provided agentic notebook.
    specs.append(
        MethodSpec(
            method_family="Agentic RAG",
            method="langgraph_agentic_hybrid",
            dataset="docred_causal",
            script="scripts/run_langgraph_agentic_hybrid_relations.py",
            args=_t(
                "--dataset-key",
                "docred_causal",
                "--backend",
                "vllm",
                "--doc-types",
                "dev",
                "--ontology-links-path",
                "data/preprocessed/docred_causal_olink_llm_embedding_onto_docredontology.jsonl",
                "--ontology-key",
                "docredontology",
                "--ontology-method",
                "llm_embedding",
                "--kg-path",
                "data/kg/docred_causal__types=train_annotated_skip=0_limit=None__kg.json",
                "--kg-max-triples",
                "40",
                "--shot-num",
                "0",
                "--planner-mode",
                "llm",
                "--enable-web",
                "--enable-wikidata",
                "--max-llm-calls",
                "2",
            ),
            doc_filter_arg="--doc-types",
            doc_filter_value="dev",
            notebook_source="agentic.ipynb cell 13",
            note="Provided notebook contains this configuration only for DocRED.",
        )
    )
    specs.append(
        MethodSpec(
            method_family="Agentic RAG",
            method="marag",
            dataset="docred_causal",
            script="scripts/run_marag_relations.py",
            args=_t(
                "--dataset-key",
                "docred_causal",
                "--backend",
                "vllm",
                "--doc-types",
                "dev",
                "--ontology-key",
                "docredontology",
                "--ontology-links-path",
                "data/preprocessed/docred_causal_olink_llm_embedding_onto_docredontology.jsonl",
                "--kg-path",
                "data/kg/docred_causal__types=train_annotated_skip=0_limit=None__kg.json",
                "--enable-web",
                "--enable-wikidata",
                "--max-llm-calls",
                "10",
                "--shot-num",
                "3",
                "--shot-type",
                "dev",
            ),
            doc_filter_arg="--doc-types",
            doc_filter_value="dev",
            notebook_source="agentic.ipynb cell 16",
            note="Provided notebook contains this configuration only for DocRED.",
        )
    )

    return specs


# ---------------------------------------------------------------------------
# Config and JSONL helpers
# ---------------------------------------------------------------------------


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_doc_type_filter(value: str) -> DocTypeFilter:
    """Parse the same all-or-comma-separated semantics used in the runners."""

    if value is None or value == "" or value == "all":
        return "all"
    items = [x.strip() for x in value.split(",") if x.strip()]
    return items or "all"


def doc_type_matches(doc: Dict[str, Any], doc_filter: DocTypeFilter) -> bool:
    if doc_filter == "all":
        return True
    if isinstance(doc_filter, str):
        return doc.get("type") == doc_filter
    return doc.get("type") in set(doc_filter)


def document_identifier(doc: Dict[str, Any]) -> str:
    return str(doc.get("document_id") or doc.get("id") or doc.get("doc_id") or "")


def count_tokens_roughly(doc: Dict[str, Any]) -> int:
    """A lightweight token proxy used only for metadata in the output table."""

    text_parts: List[str] = []
    for key in ("text", "title"):
        value = doc.get(key)
        if isinstance(value, str):
            text_parts.append(value)
    sentences = doc.get("sentences")
    if isinstance(sentences, list):
        for sent in sentences:
            if isinstance(sent, str):
                text_parts.append(sent)
            elif isinstance(sent, list):
                text_parts.append(" ".join(str(x) for x in sent))
    tokens = doc.get("tokens")
    if isinstance(tokens, list):
        if tokens and isinstance(tokens[0], list):
            return sum(len(x) for x in tokens if isinstance(x, list))
        return len(tokens)
    return len(" ".join(text_parts).split())


def count_jsonl(path: Path, doc_filter_value: str) -> Tuple[int, int]:
    """Return total non-empty JSONL docs and eligible docs after type filtering."""

    total = 0
    eligible = 0
    doc_filter = parse_doc_type_filter(doc_filter_value)
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            if doc_type_matches(doc, doc_filter):
                eligible += 1
    return total, eligible


def get_eligible_doc(path: Path, doc_filter_value: str, sample_index: int) -> Tuple[Dict[str, Any], int]:
    """
    Return the selected eligible document and the effective skip index.

    If sample_index is outside the eligible range, the last eligible document is used.
    """

    doc_filter = parse_doc_type_filter(doc_filter_value)
    eligible_docs: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            if doc_type_matches(doc, doc_filter):
                eligible_docs.append(doc)

    if not eligible_docs:
        raise ValueError(f"No eligible document in {path} for doc filter '{doc_filter_value}'.")

    effective_index = max(0, min(sample_index, len(eligible_docs) - 1))
    return eligible_docs[effective_index], effective_index


def raw_config_path(root: Path, config_arg: Optional[str]) -> Path:
    if config_arg:
        p = Path(config_arg)
        return p if p.is_absolute() else root / p
    return root / "configs" / "default.yaml"


def write_runtime_config(root: Path, source_config: Path, metrics_dir: Path) -> Path:
    """
    Copy the YAML config content into data/suplementary_metrics and redirect
    paths.data_processed so prediction artifacts do not overwrite full benchmark
    outputs.
    """

    with source_config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("paths", {})
    cfg["paths"]["data_processed"] = "data/suplementary_metrics/runtime_predictions"

    runtime_predictions = root / "data" / "suplementary_metrics" / "runtime_predictions"
    runtime_predictions.mkdir(parents=True, exist_ok=True)

    out_path = metrics_dir / "runtime_measurement_config.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return out_path


def load_resolved_config(config_path: Path) -> Dict[str, Any]:
    """Use RAGTree's own config loader so path resolution matches the project."""

    from ragtree.core.config import load_config

    return load_config(config_path)


def resolve_input_path(cfg: Dict[str, Any], dataset_key: str, root: Path) -> Path:
    """Resolve a dataset key with the same conventions as the project runners."""

    def _abs(path_like: Any) -> Path:
        p = Path(path_like)
        return p if p.is_absolute() else root / p

    ds_pre = cfg.get("datasets", {}).get("preprocessed", {}) or {}
    if dataset_key in ds_pre:
        return _abs(ds_pre[dataset_key])

    p = _abs(dataset_key)
    if p.exists() and p.is_file():
        return p

    pre_root = Path(cfg.get("paths", {}).get("data_preprocessed", "data/preprocessed"))
    if not pre_root.is_absolute():
        pre_root = root / pre_root
    candidate = pre_root / f"{dataset_key}.jsonl"
    if candidate.exists():
        return candidate

    candidate2 = pre_root / dataset_key
    if candidate2.exists():
        return candidate2

    available = ", ".join(sorted(ds_pre.keys()))
    raise FileNotFoundError(
        f"Could not resolve dataset key '{dataset_key}'. Available config keys: {available}. "
        f"Tried {candidate} and {candidate2}."
    )


# ---------------------------------------------------------------------------
# LLM-only direct execution with skip/limit
# ---------------------------------------------------------------------------


def _parse_relation_types(arg: Optional[str]) -> Optional[List[str]]:
    if not arg:
        return None
    items = [x.strip() for x in arg.split(",") if x.strip()]
    return items or None


def _arg_value(args: Sequence[str], name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        idx = list(args).index(name)
    except ValueError:
        return default
    if idx + 1 >= len(args):
        return default
    return args[idx + 1]


def _has_flag(args: Sequence[str], name: str) -> bool:
    return name in set(args)


def _parse_predict_types(arg: str) -> DocTypeFilter:
    return parse_doc_type_filter(arg)


def _prepare_icl_context(input_path: Path, icl_train_type: str, icl_train_num: int):
    """Copy of the ICL context logic, kept here to avoid modifying existing scripts."""

    from ragtree.processing.orchestrators.relations_runner import PreparedContext

    few_shots: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            if doc.get("type") != icl_train_type:
                continue
            rels = doc.get("relations")
            if not isinstance(rels, dict) or not rels:
                continue
            few_shots.append(doc)
            if len(few_shots) >= icl_train_num:
                break

    print(f"[supplementary-icl] collected {len(few_shots)} examples of type '{icl_train_type}'.")
    return PreparedContext(strategy_kwargs={}, predict_kwargs={"few_shots": few_shots})


def run_direct_llm_spec(
    spec: MethodSpec,
    *,
    config_path: Path,
    skip: int,
    log_path: Path,
) -> Tuple[bool, Optional[str], int, float]:
    """
    Run baseline/cot/icl through the shared orchestrator with skip/limit=1.

    Returns: success, error_message, return_code, elapsed_seconds
    """

    # Imports are placed before the timer so the measurement focuses on the
    # method execution, not this supplementary script startup.
    from ragtree.processing.orchestrators.relations_runner import (
        PreparedContext,
        RunnerLLMSections,
        run_relation_experiment,
    )

    if spec.method == "baseline":
        from ragtree.processing.rag.strategies.baseline_relations import BaselineRelationStrategy

        strategy_cls = BaselineRelationStrategy
        sections = RunnerLLMSections(
            llm_section="baseline",
            prompt_section="baseline",
            system_prompt_key="causal_relations",
        )
        prepare_context_fn = None
        doc_type_filter = parse_doc_type_filter(str(_arg_value(spec.args, "--doc-type", "all")))

    elif spec.method == "cot":
        from ragtree.processing.rag.strategies.chain_of_thought import ChainOfThoughtRelationStrategy

        strategy_cls = ChainOfThoughtRelationStrategy
        sections = RunnerLLMSections(
            llm_section="cot",
            prompt_section="baseline",
            system_prompt_key="causal_relations",
        )
        print_cot = _has_flag(spec.args, "--print-cot")

        def prepare_context_fn(input_path: Path, cfg: Dict[str, Any]) -> PreparedContext:  # type: ignore[no-redef]
            return PreparedContext(strategy_kwargs={}, predict_kwargs={"print_cot": print_cot})

        doc_type_filter = parse_doc_type_filter(str(_arg_value(spec.args, "--doc-type", "all")))

    elif spec.method == "icl":
        from ragtree.processing.rag.strategies.baseline_icl import ICLRelationStrategy

        strategy_cls = ICLRelationStrategy
        sections = RunnerLLMSections(
            llm_section="icl",
            prompt_section="baseline",
            system_prompt_key="causal_relations",
        )
        icl_train_type = str(_arg_value(spec.args, "--icl-train-type", "train"))
        icl_train_num = int(str(_arg_value(spec.args, "--icl-train-num", "8")))

        def prepare_context_fn(input_path: Path, cfg: Dict[str, Any]) -> PreparedContext:  # type: ignore[no-redef]
            return _prepare_icl_context(input_path, icl_train_type, icl_train_num)

        doc_type_filter = _parse_predict_types(str(_arg_value(spec.args, "--icl-predict-types", "all")))

    else:
        raise ValueError(f"Unsupported direct LLM method: {spec.method}")

    dataset_key = str(_arg_value(spec.args, "--dataset-key", spec.dataset))
    backend = _arg_value(spec.args, "--backend", None)
    model = _arg_value(spec.args, "--model", None)
    relation_types = _parse_relation_types(_arg_value(spec.args, "--relation-types", None))
    output_format = str(_arg_value(spec.args, "--output-format", "full"))

    start = time.perf_counter()
    success = True
    error: Optional[str] = None
    return_code = 0

    with log_path.open("w", encoding="utf-8") as log_f, contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
        try:
            run_relation_experiment(
                strategy_cls=strategy_cls,
                config_path=config_path,
                dataset_key=dataset_key,
                backend=backend,
                model=model,
                cli_relation_types=relation_types,
                output_format=output_format,
                doc_type_filter=doc_type_filter,
                skip=skip,
                limit=1,
                sections=sections,
                prepare_context_fn=prepare_context_fn,
            )
        except Exception:
            success = False
            return_code = 1
            error = traceback.format_exc()
            print(error)

    elapsed = time.perf_counter() - start
    return success, error, return_code, elapsed


# ---------------------------------------------------------------------------
# Script execution for non-LLM methods
# ---------------------------------------------------------------------------


def command_for_spec(spec: MethodSpec, *, config_path: Path, skip: int) -> List[str]:
    """Build the original script command plus measurement-only skip/limit/config."""

    cmd = [sys.executable, spec.script]
    cmd.extend(spec.args)
    cmd.extend(["--config", str(config_path), "--skip", str(skip), "--limit", "1"])
    return cmd


def run_script_spec(
    spec: MethodSpec,
    *,
    root: Path,
    config_path: Path,
    skip: int,
    log_path: Path,
    timeout: Optional[int],
) -> Tuple[bool, Optional[str], int, float]:
    cmd = command_for_spec(spec, config_path=config_path, skip=skip)
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(root) if not existing_pythonpath else f"{root}{os.pathsep}{existing_pythonpath}"
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            cwd=root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        elapsed = time.perf_counter() - start
        log_path.write_text(completed.stdout or "", encoding="utf-8")
        success = completed.returncode == 0
        error = None if success else f"Command failed with return code {completed.returncode}. See log: {log_path}"
        return success, error, completed.returncode, elapsed
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        log_path.write_text(str(output), encoding="utf-8")
        return False, f"Timeout after {timeout} seconds. See log: {log_path}", 124, elapsed
    except Exception:
        elapsed = time.perf_counter() - start
        err = traceback.format_exc()
        log_path.write_text(err, encoding="utf-8")
        return False, err, 1, elapsed


# ---------------------------------------------------------------------------
# Metrics and output
# ---------------------------------------------------------------------------


def compute_energy_and_co2(
    elapsed_seconds: float,
    n_eligible: int,
    *,
    power_kw: float,
    carbon_intensity_kg_per_kwh: float,
) -> Dict[str, float]:
    runtime_hours = elapsed_seconds / 3600.0
    energy_kwh_doc = runtime_hours * power_kw
    co2_kg_doc = energy_kwh_doc * carbon_intensity_kg_per_kwh

    estimated_total_seconds = elapsed_seconds * n_eligible
    estimated_total_energy_kwh = energy_kwh_doc * n_eligible
    estimated_total_co2_kg = co2_kg_doc * n_eligible

    return {
        "elapsed_seconds_doc": elapsed_seconds,
        "elapsed_minutes_doc": elapsed_seconds / 60.0,
        "energy_kwh_doc": energy_kwh_doc,
        "co2_kg_doc": co2_kg_doc,
        "co2_g_doc": co2_kg_doc * 1000.0,
        "estimated_total_seconds": estimated_total_seconds,
        "estimated_total_minutes": estimated_total_seconds / 60.0,
        "estimated_total_hours": estimated_total_seconds / 3600.0,
        "estimated_total_energy_kwh": estimated_total_energy_kwh,
        "estimated_total_co2_kg": estimated_total_co2_kg,
        "estimated_total_co2_g": estimated_total_co2_kg * 1000.0,
    }


def safe_slug(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def make_latex_table_rows(rows: List[Dict[str, Any]]) -> str:
    """Create a small LaTeX-friendly text file for quick copy/paste."""

    lines = []
    for row in rows:
        if not row.get("success"):
            continue
        lines.append(
            "{} & {} & {} & {:.2f} & {:.2f} & {:.4f} \\\\".format(
                row.get("method", ""),
                row.get("dataset", ""),
                int(row.get("n_eligible_documents") or 0),
                float(row.get("elapsed_seconds_doc") or 0.0),
                float(row.get("estimated_total_minutes") or 0.0),
                float(row.get("estimated_total_co2_g") or 0.0),
            )
        )
    return "\n".join(lines) + ("\n" if lines else "")


# ---------------------------------------------------------------------------
# Main measurement loop
# ---------------------------------------------------------------------------


def filter_specs(specs: List[MethodSpec], args: argparse.Namespace) -> List[MethodSpec]:
    out = specs
    if args.only_family:
        wanted = {x.strip().lower() for x in args.only_family.split(",") if x.strip()}
        out = [s for s in out if s.method_family.lower() in wanted]
    if args.only_method:
        wanted = {x.strip().lower() for x in args.only_method.split(",") if x.strip()}
        out = [s for s in out if s.method.lower() in wanted]
    if args.only_dataset:
        wanted = {x.strip().lower() for x in args.only_dataset.split(",") if x.strip()}
        out = [s for s in out if s.dataset.lower() in wanted]
    return out


def print_spec_list(specs: Sequence[MethodSpec]) -> None:
    print(f"Configured method/dataset runs: {len(specs)}")
    for i, spec in enumerate(specs, start=1):
        print(
            f"{i:03d}. {spec.method_family:22s} | {spec.method:26s} | "
            f"{spec.dataset:18s} | filter={spec.doc_filter_value:8s} | mode={spec.run_mode}"
        )


def run_measurements(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    metrics_dir = root / "data" / "suplementary_metrics"
    logs_dir = metrics_dir / "logs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    source_cfg = raw_config_path(root, args.config)
    if not source_cfg.exists():
        raise FileNotFoundError(f"Config file not found: {source_cfg}")

    runtime_cfg = write_runtime_config(root, source_cfg, metrics_dir)
    resolved_cfg = load_resolved_config(runtime_cfg)

    specs = build_method_specs(
        include_inferred_baseline_eventstoryline=not args.no_inferred_baseline_eventstoryline
    )
    specs = filter_specs(specs, args)

    if args.list:
        print_spec_list(specs)
        return 0

    if not specs:
        print("No method/dataset run selected.")
        return 1

    print(f"[supplementary] root={root}")
    print(f"[supplementary] source config={source_cfg}")
    print(f"[supplementary] runtime config={runtime_cfg}")
    print(f"[supplementary] output dir={metrics_dir}")
    print(f"[supplementary] selected runs={len(specs)}")
    print(f"[supplementary] power_kw={args.power_kw}")
    print(f"[supplementary] carbon_intensity_kg_per_kwh={args.carbon_intensity}")

    rows: List[Dict[str, Any]] = []
    failures = 0

    for idx, spec in enumerate(specs, start=1):
        # The dataset key used by the command may differ from spec.dataset for
        # ontology-linked datasets such as GrOWL-RAG.
        command_dataset_key = str(_arg_value(spec.args, "--dataset-key", spec.dataset))

        try:
            input_path = resolve_input_path(resolved_cfg, command_dataset_key, root)
            total_docs, eligible_docs = count_jsonl(input_path, spec.doc_filter_value)
            sample_doc, effective_skip = get_eligible_doc(
                input_path,
                spec.doc_filter_value,
                int(args.sample_index),
            )
            sample_doc_id = document_identifier(sample_doc)
            sample_doc_type = str(sample_doc.get("type", ""))
            sample_doc_tokens = count_tokens_roughly(sample_doc)
        except Exception as exc:
            failures += 1
            row = {
                "success": False,
                "error": f"Input resolution/counting failed: {exc}",
                "method_family": spec.method_family,
                "method": spec.method,
                "dataset": spec.dataset,
                "command_dataset_key": command_dataset_key,
                "script": spec.script,
                "notebook_source": spec.notebook_source,
                "notebook_like_command": spec.notebook_like_command,
                "created_at_utc": now_utc_iso(),
            }
            rows.append(row)
            print(f"[{idx}/{len(specs)}] FAILED before run: {spec.method}/{spec.dataset}: {exc}")
            if args.stop_on_error:
                break
            continue

        log_name = f"{idx:03d}_{safe_slug(spec.method)}__{safe_slug(spec.dataset)}.log"
        log_path = logs_dir / log_name

        if spec.run_mode == "direct_llm":
            cmd_for_display = [
                "<direct-llm-orchestrator>",
                *spec.args,
                "--config",
                str(runtime_cfg),
                "--skip",
                str(effective_skip),
                "--limit",
                "1",
            ]
        else:
            cmd_for_display = command_for_spec(spec, config_path=runtime_cfg, skip=effective_skip)

        print(
            f"[{idx}/{len(specs)}] {spec.method_family} | {spec.method} | {spec.dataset} | "
            f"eligible={eligible_docs} | sample_skip={effective_skip} | doc_id={sample_doc_id or 'N/A'}"
        )

        if args.dry_run:
            success = True
            error = None
            return_code = 0
            elapsed = 0.0
        else:
            if spec.run_mode == "direct_llm":
                success, error, return_code, elapsed = run_direct_llm_spec(
                    spec,
                    config_path=runtime_cfg,
                    skip=effective_skip,
                    log_path=log_path,
                )
            else:
                success, error, return_code, elapsed = run_script_spec(
                    spec,
                    root=root,
                    config_path=runtime_cfg,
                    skip=effective_skip,
                    log_path=log_path,
                    timeout=args.timeout,
                )

        if not success:
            failures += 1

        energy = compute_energy_and_co2(
            elapsed,
            eligible_docs,
            power_kw=float(args.power_kw),
            carbon_intensity_kg_per_kwh=float(args.carbon_intensity),
        )

        row = {
            "success": success,
            "return_code": return_code,
            "error": error or "",
            "method_family": spec.method_family,
            "method": spec.method,
            "dataset": spec.dataset,
            "command_dataset_key": command_dataset_key,
            "script": spec.script,
            "run_mode": spec.run_mode,
            "doc_filter_arg": spec.doc_filter_arg,
            "doc_filter_value": spec.doc_filter_value,
            "n_total_jsonl_documents": total_docs,
            "n_eligible_documents": eligible_docs,
            "sample_index_requested": int(args.sample_index),
            "sample_index_effective": effective_skip,
            "sample_document_id": sample_doc_id,
            "sample_document_type": sample_doc_type,
            "sample_document_token_proxy": sample_doc_tokens,
            "power_kw": float(args.power_kw),
            "carbon_intensity_kg_per_kwh": float(args.carbon_intensity),
            "carbon_intensity_g_per_kwh": float(args.carbon_intensity) * 1000.0,
            "input_jsonl": str(input_path),
            "runtime_config": str(runtime_cfg),
            "log_path": str(log_path),
            "notebook_source": spec.notebook_source,
            "note": spec.note,
            "command": " ".join(cmd_for_display),
            "notebook_like_command": spec.notebook_like_command,
            "created_at_utc": now_utc_iso(),
        }
        row.update(energy)
        rows.append(row)

        status = "OK" if success else "FAILED"
        print(
            f"    -> {status}: elapsed={elapsed:.2f}s, "
            f"estimated_total={energy['estimated_total_minutes']:.2f}min, "
            f"estimated_total_co2={energy['estimated_total_co2_g']:.4f}g"
        )

        # Persist after every run so partial results survive interruptions.
        payload = {
            "metadata": {
                "created_at_utc": now_utc_iso(),
                "root": str(root),
                "source_config": str(source_cfg),
                "runtime_config": str(runtime_cfg),
                "power_kw": float(args.power_kw),
                "carbon_intensity_kg_per_kwh": float(args.carbon_intensity),
                "carbon_intensity_g_per_kwh": float(args.carbon_intensity) * 1000.0,
                "sample_index": int(args.sample_index),
                "dry_run": bool(args.dry_run),
                "interpretation": (
                    "One eligible document is measured per method/dataset and multiplied by the number "
                    "of eligible JSONL documents. Values are approximate cost indicators, not exact "
                    "full-benchmark measurements."
                ),
            },
            "rows": rows,
        }
        write_json(metrics_dir / "runtime_co2_by_method_dataset.json", payload)
        write_csv(metrics_dir / "runtime_co2_by_method_dataset.csv", rows)
        (metrics_dir / "runtime_co2_latex_rows.txt").write_text(
            make_latex_table_rows(rows),
            encoding="utf-8",
        )

        if args.stop_on_error and not success:
            break

    payload = {
        "metadata": {
            "created_at_utc": now_utc_iso(),
            "root": str(root),
            "source_config": str(source_cfg),
            "runtime_config": str(runtime_cfg),
            "output_dir": str(metrics_dir),
            "n_selected_runs": len(specs),
            "n_completed_rows": len(rows),
            "n_failures": failures,
            "power_kw": float(args.power_kw),
            "carbon_intensity_kg_per_kwh": float(args.carbon_intensity),
            "carbon_intensity_g_per_kwh": float(args.carbon_intensity) * 1000.0,
            "sample_index": int(args.sample_index),
            "dry_run": bool(args.dry_run),
        },
        "method_specs": [asdict(spec) for spec in specs],
        "rows": rows,
    }
    write_json(metrics_dir / "runtime_co2_by_method_dataset.json", payload)
    write_csv(metrics_dir / "runtime_co2_by_method_dataset.csv", rows)
    (metrics_dir / "runtime_co2_latex_rows.txt").write_text(
        make_latex_table_rows(rows),
        encoding="utf-8",
    )

    print("[supplementary] wrote:")
    print(f"  - {metrics_dir / 'runtime_co2_by_method_dataset.csv'}")
    print(f"  - {metrics_dir / 'runtime_co2_by_method_dataset.json'}")
    print(f"  - {metrics_dir / 'runtime_co2_latex_rows.txt'}")
    print(f"  - logs: {logs_dir}")

    return 1 if failures else 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Measure one-document runtime and estimated CO2 for RAGTree methods/datasets."
    )
    p.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="RAGTree project root. Default: parent of scripts/.",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Config path relative to root. Default: configs/default.yaml.",
    )
    p.add_argument(
        "--power-kw",
        type=float,
        default=0.300,
        help="Estimated system power draw in kW. Default: 0.300.",
    )
    p.add_argument(
        "--carbon-intensity",
        type=float,
        default=0.045,
        help="Carbon intensity in kgCO2/kWh. Default: 0.045, i.e. 45 gCO2/kWh.",
    )
    p.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Eligible document index to measure after doc-type filtering. Default: 0.",
    )
    p.add_argument(
        "--only-method",
        default=None,
        help="Comma-separated method filter, e.g. baseline,cot,rag_tree_method.",
    )
    p.add_argument(
        "--only-dataset",
        default=None,
        help="Comma-separated dataset filter, e.g. docred_causal,fincausal.",
    )
    p.add_argument(
        "--only-family",
        default=None,
        help='Comma-separated family filter, e.g. "LLM-only,KG-based RAG".',
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional timeout in seconds per external script run.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve docs and print commands, but do not execute method scripts.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List configured method/dataset runs and exit.",
    )
    p.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop at the first failed method/dataset run. Default: keep partial results.",
    )
    p.add_argument(
        "--no-inferred-baseline-eventstoryline",
        action="store_true",
        help=(
            "Disable the inferred EventStoryLine baseline run. The provided notebook contains "
            "EventStoryLine for CoT/ICL/RAG methods, but not an explicit baseline cell."
        ),
    )
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    raise SystemExit(run_measurements(args))


if __name__ == "__main__":
    main()
