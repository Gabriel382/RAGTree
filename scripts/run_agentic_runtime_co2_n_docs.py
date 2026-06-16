#!/usr/bin/env python3
"""
Run a controlled runtime/CO2 measurement for the three agentic RAG methods
on the SAME document slice for each dataset.

Designed to be placed under:

    scripts/run_agentic_runtime_co2_n_docs.py

The script is additive only: it creates supplementary metric files under
root/data/suplementary_metrics and does not modify existing benchmark code.

Methods measured:
    ARAG              -> scripts/run_agentic_hybrid_relations.py
    SimpleAgenticRAG  -> scripts/run_langgraph_agentic_simple_relations.py
    MARAG             -> scripts/run_marag_relations.py

For each dataset, all three methods are run with the same doc-type filter,
the same --skip value, and the same --limit value. This makes the comparison
more reliable than measuring a single arbitrary document per method.

Default test configuration:
    --skip 0 --n-docs 10

Outputs:
    data/suplementary_metrics/runtime_co2_agentic_n_docs_raw.csv
    data/suplementary_metrics/runtime_co2_agentic_n_docs_raw.json
    data/suplementary_metrics/runtime_co2_agentic_n_docs_selected_docs.csv
    data/suplementary_metrics/runtime_co2_agentic_n_docs_selected_docs.json
    data/suplementary_metrics/runtime_co2_agentic_n_docs_table.tex
    data/suplementary_metrics/logs_agentic_n_docs/
    data/suplementary_metrics/runtime_predictions_agentic_n_docs/<run-label>/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc


# ---------------------------------------------------------------------------
# Specifications
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    """Dataset-specific parameters shared by ARAG, SimpleAgenticRAG, and MARAG."""

    dataset: str
    display_dataset: str
    doc_types: str
    ontology_key: str
    ontology_links_path: str
    kg_path: str
    marag_shot_type: str
    notebook_source: str


DATASETS: List[DatasetSpec] = [
    DatasetSpec(
        dataset="maven_ere",
        display_dataset="MavenERE",
        doc_types="all",
        ontology_key="EventKG",
        ontology_links_path="data/preprocessed/maven_ere_olink_llm_embedding_onto_EventKG.jsonl",
        kg_path="data/kg/maven_ere__types=train_skip=0_limit=None__kg.json",
        marag_shot_type="train",
        notebook_source="agentic.ipynb MavenERE agentic cells",
    ),
    DatasetSpec(
        dataset="eventstoryline",
        display_dataset="EventStoryLine",
        doc_types="all",
        ontology_key="owltime",
        ontology_links_path="data/preprocessed/eventstoryline_olink_llm_embedding_onto_owltime.jsonl",
        kg_path="data/kg/eventstoryline__types=full_skip=0_limit=10__kg.json",
        marag_shot_type="full",
        notebook_source="agentic.ipynb EventStoryLine agentic cells",
    ),
    DatasetSpec(
        dataset="fincausal",
        display_dataset="FinCausal",
        doc_types="all",
        ontology_key="fibocoreplus",
        ontology_links_path="data/preprocessed/fincausal_olink_llm_embedding_onto_fibocoreplus.jsonl",
        kg_path="data/kg/fincausal__types=train.csv_skip=0_limit=None__kg.json",
        marag_shot_type="train.csv",
        notebook_source="agentic.ipynb FinCausal agentic cells",
    ),
    DatasetSpec(
        dataset="docred_causal",
        display_dataset="DocRED",
        doc_types="dev",
        ontology_key="docredontology",
        ontology_links_path="data/preprocessed/docred_causal_olink_llm_embedding_onto_docredontology.jsonl",
        kg_path="data/kg/docred_causal__types=train_annotated_skip=0_limit=None__kg.json",
        marag_shot_type="dev",
        notebook_source="agentic.ipynb DocRED agentic cells",
    ),
    DatasetSpec(
        dataset="causalbank",
        display_dataset="CausalBank",
        doc_types="all",
        ontology_key="wordnetfull",
        ontology_links_path="data/preprocessed/causalbank_olink_llm_embedding_onto_wordnetfull.jsonl",
        kg_path="data/kg/causalbank__types=resulted_from_skip=0_limit=None__kg.json",
        marag_shot_type="resulted_from",
        notebook_source="agentic.ipynb CausalBank agentic cells",
    ),
]

DATASET_ORDER = [d.dataset for d in DATASETS]
DATASET_DISPLAY = {d.dataset: d.display_dataset for d in DATASETS}

METHOD_ORDER = ["agentic_hybrid", "langgraph_agentic_simple", "marag"]
METHOD_DISPLAY = {
    "agentic_hybrid": "ARAG",
    "langgraph_agentic_simple": "SimpleAgenticRAG",
    "marag": "MARAG",
}
METHOD_FAMILY = "Agentic RAG"


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def find_repo_root(start: Optional[Path] = None) -> Path:
    """Find the RAGTree repository root from the current working directory."""
    cur = (start or Path.cwd()).resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "configs" / "default.yaml").exists() and (candidate / "scripts").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find RAGTree root. Run this script from the repository root "
        "or one of its subdirectories."
    )


def raw_config_path(root: Path, config_arg: Optional[str]) -> Path:
    """Resolve the source YAML config path."""
    if config_arg:
        p = Path(config_arg)
        return p if p.is_absolute() else root / p
    return root / "configs" / "default.yaml"


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML mapping."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML file must contain a mapping: {path}")
    return data


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    """Write a YAML mapping."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def write_runtime_config(
    *,
    root: Path,
    source_config: Path,
    metrics_dir: Path,
    run_label: str,
) -> Tuple[Path, Path]:
    """
    Copy the original config and redirect paths.data_processed to a metrics-only
    prediction folder. Existing benchmark predictions are not overwritten.
    """
    cfg = load_yaml(source_config)
    cfg.setdefault("paths", {})

    runtime_predictions_rel = f"data/suplementary_metrics/runtime_predictions_agentic_n_docs/{run_label}"
    cfg["paths"]["data_processed"] = runtime_predictions_rel

    runtime_predictions_dir = root / runtime_predictions_rel
    runtime_predictions_dir.mkdir(parents=True, exist_ok=True)

    runtime_cfg = metrics_dir / f"runtime_measurement_agentic_n_docs_{run_label}.yaml"
    write_yaml(runtime_cfg, cfg)
    return runtime_cfg, runtime_predictions_dir


def load_resolved_config(config_path: Path) -> Dict[str, Any]:
    """Use the project config loader so relative paths are resolved consistently."""
    from ragtree.core.config import load_config

    return load_config(config_path)


def as_repo_path(root: Path, path_like: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    p = Path(path_like)
    return p if p.is_absolute() else root / p


def resolve_input_path(cfg: Dict[str, Any], dataset_key: str, root: Path) -> Path:
    """Resolve an input JSONL from cfg['datasets']['preprocessed']."""
    ds_pre = cfg.get("datasets", {}).get("preprocessed", {}) or {}
    if dataset_key in ds_pre:
        return as_repo_path(root, ds_pre[dataset_key])

    pre_root = Path(cfg.get("paths", {}).get("data_preprocessed", "data/preprocessed"))
    pre_root = pre_root if pre_root.is_absolute() else root / pre_root
    candidate = pre_root / f"{dataset_key}.jsonl"
    if candidate.exists():
        return candidate

    available = ", ".join(sorted(ds_pre.keys()))
    raise FileNotFoundError(
        f"Could not resolve dataset key '{dataset_key}'. Available config keys: {available}. "
        f"Tried {candidate}."
    )


def parse_doc_types(arg: str) -> Sequence[str] | str:
    """Parse a doc-type filter such as 'all' or 'dev,test'."""
    if not arg or arg == "all":
        return "all"
    items = [x.strip() for x in str(arg).split(",") if x.strip()]
    return items or "all"


def should_keep_doc(doc: Dict[str, Any], doc_types: Sequence[str] | str) -> bool:
    """Return whether a document matches the filter."""
    if doc_types == "all":
        return True
    return doc.get("type") in set(doc_types)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def document_id(doc: Dict[str, Any]) -> str:
    """Return a stable document identifier for logging."""
    return str(doc.get("document_id") or doc.get("id") or "")


def token_proxy(doc: Dict[str, Any]) -> int:
    """Estimate document size with a simple token proxy."""
    chunks: List[str] = []
    for key in ("title", "text"):
        value = doc.get(key)
        if isinstance(value, str):
            chunks.append(value)
    sentences = doc.get("sentences")
    if isinstance(sentences, list):
        for sent in sentences:
            if isinstance(sent, str):
                chunks.append(sent)
            elif isinstance(sent, list):
                chunks.append(" ".join(str(x) for x in sent))
    tokens = doc.get("tokens")
    if isinstance(tokens, list):
        flattened: List[str] = []
        for item in tokens:
            if isinstance(item, list):
                flattened.extend(str(x) for x in item)
            else:
                flattened.append(str(item))
        if flattened:
            return len(flattened)
    text_blob = "\n".join(chunks)
    return len(text_blob.split()) if text_blob else 0


def count_and_select_docs(
    *,
    input_path: Path,
    doc_types: str,
    skip: int,
    n_docs: int,
) -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Count eligible docs and select the exact shared slice used by all methods.

    Returns:
        n_eligible, selected_docs, selected_doc_summary
    """
    parsed_types = parse_doc_types(doc_types)
    eligible_index = -1
    selected_docs: List[Dict[str, Any]] = []
    selected_summary: List[Dict[str, Any]] = []

    for doc in iter_jsonl(input_path):
        if not should_keep_doc(doc, parsed_types):
            continue
        eligible_index += 1
        if eligible_index < skip:
            continue
        if len(selected_docs) >= n_docs:
            continue
        selected_docs.append(doc)
        selected_summary.append(
            {
                "eligible_index": eligible_index,
                "document_id": document_id(doc),
                "document_type": doc.get("type"),
                "token_proxy": token_proxy(doc),
            }
        )

    n_eligible = eligible_index + 1
    return n_eligible, selected_docs, selected_summary


def tail_text(path: Path, n_lines: int) -> str:
    """Return the last n_lines of a text file."""
    if n_lines <= 0 or not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(lines[-n_lines:])


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write rows to CSV, preserving all encountered keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: Any) -> None:
    """Write JSON with indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Endpoint compatibility launcher
# ---------------------------------------------------------------------------

def child_launcher_command(script: str, script_args: Sequence[str], endpoint_mode: str, endpoint_verbose: bool) -> List[str]:
    """
    Return a Python command that monkeypatches requests.post in the child process.

    This preserves the original project scripts while allowing compatibility with
    local OpenAI-compatible servers that expose /v1/completions instead of
    /v1/chat/completions.
    """
    launcher = r'''
import os
import runpy
import sys
from typing import Any, Dict, List, Sequence, Tuple


def _chat_endpoint_candidates(url: str, mode: str) -> List[Tuple[str, str]]:
    if mode in {"none", "no-fallback"}:
        return [(url, "chat")]
    base = url
    for suffix in ("/v1/chat/completions", "/chat/completions", "/api/v1/chat/completions"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    candidates: List[Tuple[str, str]] = []
    if mode in {"auto", "v1"}:
        candidates.append((f"{base}/v1/chat/completions", "chat"))
    if mode in {"auto", "root"}:
        candidates.append((f"{base}/chat/completions", "chat"))
    if mode in {"auto", "api-v1"}:
        candidates.append((f"{base}/api/v1/chat/completions", "chat"))
    if mode in {"auto", "responses"}:
        candidates.append((f"{base}/v1/responses", "responses"))
    if mode in {"auto", "completions"}:
        candidates.append((f"{base}/v1/completions", "completions"))
    out: List[Tuple[str, str]] = []
    seen = set()
    for item in candidates:
        if item[0] in seen:
            continue
        seen.add(item[0])
        out.append(item)
    return out or [(url, "chat")]


def _messages_to_prompt(messages: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = str(msg.get("content", ""))
        if content:
            parts.append(f"{role.upper()}: {content}")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)


def _payload_for_endpoint(payload: Dict[str, Any], endpoint_mode: str) -> Dict[str, Any]:
    if endpoint_mode == "chat":
        return dict(payload)
    messages = payload.get("messages") or []
    prompt = _messages_to_prompt(messages if isinstance(messages, list) else [])
    model = payload.get("model")
    temperature = payload.get("temperature")
    max_tokens = payload.get("max_tokens") or payload.get("max_completion_tokens")
    if endpoint_mode == "responses":
        out: Dict[str, Any] = {"model": model, "input": messages if messages else prompt}
        if temperature is not None:
            out["temperature"] = temperature
        if max_tokens is not None:
            out["max_output_tokens"] = max_tokens
        return {k: v for k, v in out.items() if v is not None}
    if endpoint_mode == "completions":
        out = {"model": model, "prompt": prompt}
        if temperature is not None:
            out["temperature"] = temperature
        if max_tokens is not None:
            out["max_tokens"] = max_tokens
        return {k: v for k, v in out.items() if v is not None}
    return dict(payload)


class _ChatCompatResponse:
    def __init__(self, response: Any, endpoint_mode: str, url: str) -> None:
        self._response = response
        self._endpoint_mode = endpoint_mode
        self._url = url
        self.status_code = getattr(response, "status_code", None)
        self.text = getattr(response, "text", "")
        self.headers = getattr(response, "headers", {})
        self.content = getattr(response, "content", b"")

    def raise_for_status(self) -> None:
        return self._response.raise_for_status()

    def json(self) -> Dict[str, Any]:
        data = self._response.json()
        if self._endpoint_mode == "chat":
            return data
        content = ""
        if self._endpoint_mode == "responses":
            content = str(data.get("output_text") or "")
            if not content and isinstance(data.get("output"), list):
                chunks: List[str] = []
                for item in data.get("output", []):
                    if isinstance(item, dict):
                        for c in item.get("content", []) or []:
                            if isinstance(c, dict):
                                chunks.append(str(c.get("text") or c.get("content") or ""))
                content = "".join(chunks)
        elif self._endpoint_mode == "completions":
            choices = data.get("choices") or []
            if choices and isinstance(choices[0], dict):
                content = str(choices[0].get("text") or "")
        return {"choices": [{"message": {"content": content}}], "_raw_response": data, "_endpoint_url": self._url}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)


def install(mode: str, verbose: bool) -> None:
    if mode in {"none", "no-fallback"}:
        return
    import requests
    original_post = requests.post

    def patched_post(url: str, *pargs: Any, **kwargs: Any) -> Any:
        url_str = str(url)
        if not url_str.endswith("/chat/completions"):
            return original_post(url, *pargs, **kwargs)
        original_json = kwargs.get("json")
        if not isinstance(original_json, dict):
            return original_post(url, *pargs, **kwargs)
        last_response = None
        last_exc = None
        for candidate_url, endpoint_mode in _chat_endpoint_candidates(url_str, mode):
            call_kwargs = dict(kwargs)
            call_kwargs["json"] = _payload_for_endpoint(original_json, endpoint_mode)
            try:
                response = original_post(candidate_url, *pargs, **call_kwargs)
            except Exception as exc:
                last_exc = exc
                if mode != "auto":
                    raise
                continue
            last_response = response
            if getattr(response, "status_code", None) == 404 and mode == "auto":
                if verbose:
                    print(f"[endpoint-fallback] 404 at {candidate_url}; trying next candidate.")
                continue
            if verbose and candidate_url != url_str:
                print(f"[endpoint-fallback] using {candidate_url} for chat completion calls.")
            return _ChatCompatResponse(response, endpoint_mode, candidate_url)
        if last_response is not None:
            return last_response
        if last_exc is not None:
            raise last_exc
        return original_post(url, *pargs, **kwargs)

    requests.post = patched_post


mode = os.environ.get("RAGTREE_SUPPLEMENTARY_CHAT_ENDPOINT_MODE", "auto")
verbose = os.environ.get("RAGTREE_SUPPLEMENTARY_ENDPOINT_VERBOSE", "0") == "1"
install(mode, verbose)
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
'''
    env_mode = str(endpoint_mode)
    env_verbose = "1" if endpoint_verbose else "0"
    return [
        sys.executable,
        "-c",
        launcher,
        script,
        *script_args,
    ]


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------

def build_script_args(
    *,
    root: Path,
    runtime_config: Path,
    spec: DatasetSpec,
    method: str,
    backend: str,
    skip: int,
    n_docs: int,
    marag_max_llm_calls: int,
    arag_max_llm_calls: int,
    simple_max_llm_calls: int,
) -> Tuple[str, List[str], str]:
    """
    Build the existing RAGTree script command for one method/dataset pair.

    Returns:
        script_path, args, notebook_like_command
    """
    common_config = ["--config", str(runtime_config), "--skip", str(skip), "--limit", str(n_docs)]

    if method == "agentic_hybrid":
        script = "scripts/run_agentic_hybrid_relations.py"
        args = [
            "--dataset-key", spec.dataset,
            "--backend", backend,
            "--doc-types", spec.doc_types,
            "--ontology-links-path", spec.ontology_links_path,
            "--ontology-key", spec.ontology_key,
            "--ontology-method", "llm_embedding",
            "--kg-path", spec.kg_path,
            "--shot-num", "3",
            "--shot-doc-types", "all",
            "--max-llm-calls", str(arag_max_llm_calls),
            *common_config,
        ]
    elif method == "langgraph_agentic_simple":
        script = "scripts/run_langgraph_agentic_simple_relations.py"
        args = [
            "--dataset-key", spec.dataset,
            "--backend", backend,
            "--doc-types", spec.doc_types,
            "--shot-num", "3",
            "--shot-doc-types", "all",
            "--max-llm-calls", str(simple_max_llm_calls),
            *common_config,
        ]
    elif method == "marag":
        script = "scripts/run_marag_relations.py"
        args = [
            "--dataset-key", spec.dataset,
            "--backend", backend,
            "--doc-types", spec.doc_types,
            "--ontology-key", spec.ontology_key,
            "--ontology-links-path", spec.ontology_links_path,
            "--kg-path", spec.kg_path,
            "--enable-web",
            "--enable-wikidata",
            "--max-llm-calls", str(marag_max_llm_calls),
            "--shot-num", "3",
            "--shot-type", spec.marag_shot_type,
            *common_config,
        ]
    else:
        raise ValueError(f"Unknown method: {method}")

    notebook_like = "%run \"{}\" {}".format(script, " ".join(args))
    return script, args, notebook_like


def expected_output_path(runtime_predictions_dir: Path, spec: DatasetSpec, method: str, backend: str) -> Path:
    """Return the output path used by the existing method scripts."""
    if method == "agentic_hybrid":
        label = "agentic_hybrid"
    elif method == "langgraph_agentic_simple":
        label = "langgraph_agentic_simple"
    elif method == "marag":
        label = "marag"
    else:
        raise ValueError(method)
    return runtime_predictions_dir / f"{spec.dataset}.{label}.{backend}.jsonl"


def count_jsonl_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL output file."""
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def fmt_float(value: Any, ndigits: int = 2) -> str:
    """Format a numeric value or return -- for missing values."""
    if value is None or value == "":
        return "--"
    try:
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return "--"


def latex_escape(text: str) -> str:
    """Escape a minimal subset of LaTeX-sensitive characters."""
    return str(text).replace("_", "\\_")


def write_latex_table(path: Path, rows: List[Dict[str, Any]], selected_by_dataset: Dict[str, Dict[str, Any]]) -> None:
    """Write a final LaTeX table grouped by dataset."""
    row_map = {(r["dataset"], r["method"]): r for r in rows if r.get("status") == "OK"}
    lines: List[str] = []
    lines.append(r"\begin{table*}[pos=htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Runtime and estimated CO$_2$ indicators for the agentic RAG methods grouped by dataset. Each method is evaluated on the same document slice for a given dataset. Each entry reports average time per document, extrapolated total runtime in hours, and extrapolated CO$_2$ in grams.}")
    lines.append(r"\label{tab:runtime_co2_agentic_n_docs_grouped_by_dataset}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{8pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Method} & \textbf{Avg. time/doc. (s)} & \textbf{Est. total (h)} & \textbf{Est. CO$_2$ (g)} \\")
    lines.append(r"\midrule")

    first_dataset = True
    for dataset in DATASET_ORDER:
        selected = selected_by_dataset.get(dataset, {})
        display = DATASET_DISPLAY.get(dataset, dataset)
        docs_text = str(selected.get("n_eligible_documents", "?"))
        measured_text = str(selected.get("n_selected_documents", "?"))
        if not first_dataset:
            lines.append(r"\midrule")
        first_dataset = False
        lines.append(
            rf"\multicolumn{{4}}{{l}}{{\textbf{{{latex_escape(display)}}} \hfill \textbf{{{measured_text} measured documents / {docs_text} eligible documents}}}} \\"
        )
        lines.append(r"\midrule")
        for method in METHOD_ORDER:
            row = row_map.get((dataset, method))
            method_name = METHOD_DISPLAY[method]
            if row is None:
                lines.append(f"{method_name} & -- & -- & -- \\\\")
            else:
                lines.append(
                    rf"{method_name} & {fmt_float(row.get('elapsed_seconds_per_doc'))} & "
                    rf"{fmt_float(row.get('estimated_total_hours'))} & "
                    f"{fmt_float(row.get('estimated_total_co2_g'))} \\\\" 
                )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main measurement loop
# ---------------------------------------------------------------------------

def run_measurements(args: argparse.Namespace) -> int:
    """Run all selected method/dataset measurements."""
    root = find_repo_root()
    source_cfg = raw_config_path(root, args.config)
    metrics_dir = root / "data" / "suplementary_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    run_label = args.run_label or f"skip{args.skip}_n{args.n_docs}"
    runtime_cfg, runtime_predictions_dir = write_runtime_config(
        root=root,
        source_config=source_cfg,
        metrics_dir=metrics_dir,
        run_label=run_label,
    )
    resolved_cfg = load_resolved_config(runtime_cfg)

    logs_dir = metrics_dir / "logs_agentic_n_docs" / run_label
    logs_dir.mkdir(parents=True, exist_ok=True)

    selected_specs = [d for d in DATASETS if args.only_dataset in (None, "", d.dataset, d.display_dataset)]
    selected_methods = [m for m in METHOD_ORDER if args.only_method in (None, "", m, METHOD_DISPLAY[m])]

    selected_by_dataset: Dict[str, Dict[str, Any]] = {}
    selected_doc_rows: List[Dict[str, Any]] = []

    print(f"[agentic-n-docs] root={root}")
    print(f"[agentic-n-docs] source config={source_cfg}")
    print(f"[agentic-n-docs] runtime config={runtime_cfg}")
    print(f"[agentic-n-docs] output dir={metrics_dir}")
    print(f"[agentic-n-docs] runtime predictions={runtime_predictions_dir}")
    print(f"[agentic-n-docs] selected datasets={len(selected_specs)}")
    print(f"[agentic-n-docs] selected methods={', '.join(METHOD_DISPLAY[m] for m in selected_methods)}")
    print(f"[agentic-n-docs] skip={args.skip}")
    print(f"[agentic-n-docs] n_docs={args.n_docs}")
    print(f"[agentic-n-docs] power_kw={args.power_kw}")
    print(f"[agentic-n-docs] carbon_intensity_kg_per_kwh={args.carbon_intensity}")
    print(f"[agentic-n-docs] chat_endpoint_mode={args.chat_endpoint_mode}")

    for spec in selected_specs:
        input_path = resolve_input_path(resolved_cfg, spec.dataset, root)
        n_eligible, selected_docs, selected_summary = count_and_select_docs(
            input_path=input_path,
            doc_types=spec.doc_types,
            skip=int(args.skip),
            n_docs=int(args.n_docs),
        )
        selected_by_dataset[spec.dataset] = {
            "dataset": spec.dataset,
            "display_dataset": spec.display_dataset,
            "input_jsonl": str(input_path),
            "doc_types": spec.doc_types,
            "skip": int(args.skip),
            "n_requested": int(args.n_docs),
            "n_selected_documents": len(selected_docs),
            "n_eligible_documents": n_eligible,
            "selected_documents": selected_summary,
        }
        for selected in selected_summary:
            selected_doc_rows.append(
                {
                    "dataset": spec.dataset,
                    "display_dataset": spec.display_dataset,
                    "doc_types": spec.doc_types,
                    "skip": int(args.skip),
                    "n_requested": int(args.n_docs),
                    "n_selected_documents": len(selected_docs),
                    "n_eligible_documents": n_eligible,
                    **selected,
                }
            )

    all_pairs: List[Tuple[DatasetSpec, str]] = [(d, m) for d in selected_specs for m in selected_methods]
    print(f"[agentic-n-docs] selected runs={len(all_pairs)}")

    rows: List[Dict[str, Any]] = []

    for run_idx, (spec, method) in enumerate(all_pairs, start=1):
        selected = selected_by_dataset[spec.dataset]
        n_selected = int(selected["n_selected_documents"])
        n_eligible = int(selected["n_eligible_documents"])
        method_name = METHOD_DISPLAY[method]

        print(
            f"[{run_idx}/{len(all_pairs)}] {spec.display_dataset} | {method_name} | "
            f"eligible={n_eligible} | selected={n_selected} | skip={args.skip} | n={args.n_docs}"
        )

        script, script_args, notebook_like = build_script_args(
            root=root,
            runtime_config=runtime_cfg,
            spec=spec,
            method=method,
            backend=str(args.backend),
            skip=int(args.skip),
            n_docs=int(args.n_docs),
            marag_max_llm_calls=int(args.marag_max_llm_calls),
            arag_max_llm_calls=int(args.arag_max_llm_calls),
            simple_max_llm_calls=int(args.simple_max_llm_calls),
        )
        log_path = logs_dir / f"{run_idx:03d}_{method}__{spec.dataset}.log"
        output_path = expected_output_path(runtime_predictions_dir, spec, method, str(args.backend))

        base_row: Dict[str, Any] = {
            "status": "DRY-RUN" if args.dry_run else "PENDING",
            "dry_run": bool(args.dry_run),
            "success": False,
            "return_code": "",
            "error": "",
            "error_type": "",
            "error_message": "",
            "method_family": METHOD_FAMILY,
            "method": method,
            "method_display": method_name,
            "dataset": spec.dataset,
            "display_dataset": spec.display_dataset,
            "doc_types": spec.doc_types,
            "skip": int(args.skip),
            "n_requested": int(args.n_docs),
            "n_selected_documents": n_selected,
            "n_eligible_documents": n_eligible,
            "selected_document_ids_json": json.dumps([x["document_id"] for x in selected["selected_documents"]], ensure_ascii=False),
            "selected_documents_json": json.dumps(selected["selected_documents"], ensure_ascii=False),
            "power_kw": float(args.power_kw),
            "carbon_intensity_kg_per_kwh": float(args.carbon_intensity),
            "carbon_intensity_g_per_kwh": float(args.carbon_intensity) * 1000.0,
            "input_jsonl": selected["input_jsonl"],
            "runtime_config": str(runtime_cfg),
            "runtime_predictions_dir": str(runtime_predictions_dir),
            "output_jsonl": str(output_path),
            "log_path": str(log_path),
            "notebook_source": spec.notebook_source,
            "command": " ".join([sys.executable, script, *script_args]),
            "notebook_like_command": notebook_like,
            "created_at_utc": utc_now(),
        }

        if n_selected <= 0:
            base_row.update(
                {
                    "status": "FAILED",
                    "error": "No selected documents for this dataset/filter/skip/n combination.",
                    "error_type": "NoSelectedDocuments",
                    "error_message": "No selected documents.",
                }
            )
            rows.append(base_row)
            print("    -> FAILED: no selected documents.")
            if args.stop_on_error:
                break
            continue

        if args.dry_run:
            rows.append(base_row)
            print("    -> DRY-RUN: command resolved but not executed.")
            continue

        env = os.environ.copy()
        env["RAGTREE_SUPPLEMENTARY_CHAT_ENDPOINT_MODE"] = str(args.chat_endpoint_mode)
        env["RAGTREE_SUPPLEMENTARY_ENDPOINT_VERBOSE"] = "1" if args.endpoint_fallback_verbose else "0"

        command = child_launcher_command(
            script,
            script_args,
            endpoint_mode=str(args.chat_endpoint_mode),
            endpoint_verbose=bool(args.endpoint_fallback_verbose),
        )

        start = time.perf_counter()
        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                proc = subprocess.run(
                    command,
                    cwd=str(root),
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            elapsed = time.perf_counter() - start
            n_processed = count_jsonl_lines(output_path)
            divisor = n_processed if n_processed > 0 else n_selected
            elapsed_per_doc = elapsed / max(1, divisor)
            energy_kwh_doc = (elapsed_per_doc / 3600.0) * float(args.power_kw)
            co2_kg_doc = energy_kwh_doc * float(args.carbon_intensity)
            co2_g_doc = co2_kg_doc * 1000.0
            estimated_total_seconds = elapsed_per_doc * n_eligible
            estimated_total_hours = estimated_total_seconds / 3600.0
            estimated_total_energy_kwh = (estimated_total_seconds / 3600.0) * float(args.power_kw)
            estimated_total_co2_kg = estimated_total_energy_kwh * float(args.carbon_intensity)
            estimated_total_co2_g = estimated_total_co2_kg * 1000.0

            if proc.returncode == 0:
                base_row.update(
                    {
                        "status": "OK",
                        "success": True,
                        "return_code": proc.returncode,
                        "n_processed_documents": n_processed,
                        "elapsed_seconds_batch": elapsed,
                        "elapsed_minutes_batch": elapsed / 60.0,
                        "elapsed_seconds_per_doc": elapsed_per_doc,
                        "elapsed_minutes_per_doc": elapsed_per_doc / 60.0,
                        "energy_kwh_doc": energy_kwh_doc,
                        "co2_kg_doc": co2_kg_doc,
                        "co2_g_doc": co2_g_doc,
                        "estimated_total_seconds": estimated_total_seconds,
                        "estimated_total_minutes": estimated_total_seconds / 60.0,
                        "estimated_total_hours": estimated_total_hours,
                        "estimated_total_energy_kwh": estimated_total_energy_kwh,
                        "estimated_total_co2_kg": estimated_total_co2_kg,
                        "estimated_total_co2_g": estimated_total_co2_g,
                    }
                )
                print(
                    f"    -> OK: batch={elapsed:.2f}s | processed={n_processed} | "
                    f"avg/doc={elapsed_per_doc:.2f}s | estimated_total={estimated_total_hours:.2f}h | "
                    f"estimated_total_co2={estimated_total_co2_g:.2f}g"
                )
            else:
                log_tail = tail_text(log_path, int(args.show_error_tail))
                base_row.update(
                    {
                        "status": "FAILED",
                        "success": False,
                        "return_code": proc.returncode,
                        "n_processed_documents": n_processed,
                        "elapsed_seconds_batch": elapsed,
                        "error": log_tail,
                        "error_type": "SubprocessFailed",
                        "error_message": f"Return code {proc.returncode}",
                    }
                )
                print(f"    -> FAILED: elapsed={elapsed:.2f}s, return_code={proc.returncode}")
                if log_tail:
                    print("    ---- log tail ----")
                    for line in log_tail.splitlines():
                        print(f"    {line}")
        except Exception as exc:  # pragma: no cover - defensive error capture.
            elapsed = time.perf_counter() - start
            err = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            base_row.update(
                {
                    "status": "FAILED",
                    "success": False,
                    "return_code": "",
                    "elapsed_seconds_batch": elapsed,
                    "error": err,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            print(f"    -> FAILED: {type(exc).__name__}: {exc}")

        rows.append(base_row)
        if base_row.get("status") == "FAILED" and args.stop_on_error:
            break

    # Write outputs.
    prefix = "runtime_co2_agentic_n_docs"
    raw_csv = metrics_dir / f"{prefix}_raw.csv"
    raw_json = metrics_dir / f"{prefix}_raw.json"
    selected_csv = metrics_dir / f"{prefix}_selected_docs.csv"
    selected_json = metrics_dir / f"{prefix}_selected_docs.json"
    table_tex = metrics_dir / f"{prefix}_table.tex"

    write_csv(raw_csv, rows)
    write_json(raw_json, rows)
    write_csv(selected_csv, selected_doc_rows)
    write_json(selected_json, selected_by_dataset)
    write_latex_table(table_tex, rows, selected_by_dataset)

    print("[agentic-n-docs] wrote:")
    print(f"  - {raw_csv}")
    print(f"  - {raw_json}")
    print(f"  - {selected_csv}")
    print(f"  - {selected_json}")
    print(f"  - {table_tex}")
    print(f"  - logs: {logs_dir}")
    print(f"  - predictions: {runtime_predictions_dir}")

    return 0 if all(r.get("status") in {"OK", "DRY-RUN"} for r in rows) else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run ARAG, SimpleAgenticRAG, and MARAG on the same N documents per dataset and estimate runtime/CO2."
    )
    parser.add_argument("--config", default=None, help="Path to configs/default.yaml. Defaults to repository configs/default.yaml.")
    parser.add_argument("--backend", default="vllm", help="LLM backend to pass to the method scripts.")
    parser.add_argument("--skip", type=int, default=0, help="Skip N eligible documents after doc-type filtering.")
    parser.add_argument("--n-docs", type=int, default=10, help="Number of eligible documents to run per method/dataset.")
    parser.add_argument("--run-label", default=None, help="Optional label for runtime prediction/log folders.")
    parser.add_argument("--only-dataset", default=None, help="Restrict to one dataset key or display name.")
    parser.add_argument("--only-method", default=None, help="Restrict to one method key or display name.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve commands and selected docs without executing methods.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop after the first failed method/dataset run.")
    parser.add_argument("--show-error-tail", type=int, default=80, help="Print this many log lines when a run fails.")
    parser.add_argument("--power-kw", type=float, default=0.300, help="Estimated system power draw in kW.")
    parser.add_argument("--carbon-intensity", type=float, default=0.045, help="Grid carbon intensity in kgCO2/kWh.")
    parser.add_argument("--arag-max-llm-calls", type=int, default=1, help="Max LLM calls for ARAG/agentic_hybrid.")
    parser.add_argument("--simple-max-llm-calls", type=int, default=2, help="Max LLM calls for SimpleAgenticRAG.")
    parser.add_argument("--marag-max-llm-calls", type=int, default=10, help="Max LLM calls for MARAG.")
    parser.add_argument(
        "--chat-endpoint-mode",
        default="auto",
        choices=["auto", "v1", "root", "api-v1", "responses", "completions", "none", "no-fallback"],
        help="Endpoint fallback mode for local OpenAI-compatible servers.",
    )
    parser.add_argument(
        "--endpoint-fallback-verbose",
        action="store_true",
        help="Print endpoint fallback messages from child processes.",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()
    raise SystemExit(run_measurements(args))


if __name__ == "__main__":
    main()
