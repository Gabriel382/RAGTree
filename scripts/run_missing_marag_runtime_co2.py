#!/usr/bin/env python3
"""
Run the missing MA-RAG supplementary runtime/CO2 measurements.

This script is intentionally additive: it creates new supplementary metric files and
never edits the existing benchmark scripts or existing result files. It is designed
for placement under:

    scripts/run_missing_marag_runtime_co2.py

It measures the MA-RAG method on the four datasets that were missing in the first
runtime/CO2 pass:

    MavenERE, EventStoryLine, FinCausal, CausalBank

For each dataset, it runs one eligible document with --skip K --limit 1 using the
same MARAG script and the same MARAG-style parameters as the agentic notebook.
It then extrapolates the observed runtime to the full number of eligible JSONL
records for that dataset.

Outputs are written under:

    data/suplementary_metrics/

Main outputs:

    runtime_co2_missing_marag.csv
    runtime_co2_missing_marag.json
    runtime_co2_agentic_only_complete.csv
    runtime_co2_agentic_only_complete.json
    runtime_co2_agentic_only_table.tex

The combined agentic table keeps only:

    ARAG              -> method = agentic_hybrid
    SimpleAgenticRAG  -> method = langgraph_agentic_simple
    MARAG             -> method = marag
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
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import yaml
except Exception as exc:  # pragma: no cover - only triggered if PyYAML is missing.
    raise RuntimeError(
        "This script requires PyYAML because it creates a temporary runtime config. "
        "Install with: pip install pyyaml"
    ) from exc


# ---------------------------------------------------------------------------
# Dataset and method specifications
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MaragSpec:
    """Configuration for one missing MARAG dataset measurement."""

    dataset: str
    display_dataset: str
    doc_types: str
    ontology_key: str
    ontology_links_path: str
    kg_path: str
    shot_type: str
    notebook_source: str


MISSING_MARAG_SPECS: List[MaragSpec] = [
    MaragSpec(
        dataset="maven_ere",
        display_dataset="MavenERE",
        doc_types="all",
        ontology_key="EventKG",
        ontology_links_path="data/preprocessed/maven_ere_olink_llm_embedding_onto_EventKG.jsonl",
        kg_path="data/kg/maven_ere__types=train_skip=0_limit=None__kg.json",
        shot_type="train",
        notebook_source="inferred from agentic.ipynb MavenERE hybrid configuration + MARAG DocRED configuration",
    ),
    MaragSpec(
        dataset="eventstoryline",
        display_dataset="EventStoryLine",
        doc_types="all",
        ontology_key="owltime",
        ontology_links_path="data/preprocessed/eventstoryline_olink_llm_embedding_onto_owltime.jsonl",
        kg_path="data/kg/eventstoryline__types=full_skip=0_limit=10__kg.json",
        shot_type="full",
        notebook_source="inferred from agentic.ipynb EventStoryLine hybrid configuration + MARAG DocRED configuration",
    ),
    MaragSpec(
        dataset="fincausal",
        display_dataset="FinCausal",
        doc_types="all",
        ontology_key="fibocoreplus",
        ontology_links_path="data/preprocessed/fincausal_olink_llm_embedding_onto_fibocoreplus.jsonl",
        kg_path="data/kg/fincausal__types=train.csv_skip=0_limit=None__kg.json",
        shot_type="train.csv",
        notebook_source="inferred from agentic.ipynb FinCausal hybrid configuration + MARAG DocRED configuration",
    ),
    MaragSpec(
        dataset="causalbank",
        display_dataset="CausalBank",
        doc_types="all",
        ontology_key="wordnetfull",
        ontology_links_path="data/preprocessed/causalbank_olink_llm_embedding_onto_wordnetfull.jsonl",
        kg_path="data/kg/causalbank__types=resulted_from_skip=0_limit=None__kg.json",
        shot_type="resulted_from",
        notebook_source="inferred from agentic.ipynb CausalBank hybrid configuration + MARAG DocRED configuration",
    ),
]

DATASET_ORDER = ["maven_ere", "eventstoryline", "fincausal", "docred_causal", "causalbank"]
DATASET_DISPLAY = {
    "maven_ere": "MavenERE",
    "eventstoryline": "EventStoryLine",
    "fincausal": "FinCausal",
    "docred_causal": "DocRED",
    "causalbank": "CausalBank",
}
METHOD_ORDER = ["agentic_hybrid", "langgraph_agentic_simple", "marag"]
METHOD_DISPLAY = {
    "agentic_hybrid": "ARAG",
    "langgraph_agentic_simple": "SimpleAgenticRAG",
    "marag": "MARAG",
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def find_repo_root(start: Optional[Path] = None) -> Path:
    """Find the RAGTree repository root from the current working directory."""
    cur = (start or Path.cwd()).resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "configs" / "default.yaml").exists() and (candidate / "scripts").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find RAGTree root. Run this script from the repository root "
        "or from one of its subdirectories."
    )


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file as a dictionary."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML config must be a mapping: {path}")
    return data


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    """Write a dictionary to YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def as_repo_path(root: Path, path_like: str | Path) -> Path:
    """Resolve a possibly relative path against the repository root."""
    p = Path(path_like)
    return p if p.is_absolute() else root / p


def parse_doc_types(arg: str) -> Sequence[str] | str:
    """Parse a doc-type filter like 'all' or 'dev,test'."""
    if not arg or arg == "all":
        return "all"
    items = [x.strip() for x in arg.split(",") if x.strip()]
    return items or "all"


def should_keep_doc(doc: Dict[str, Any], doc_types: Sequence[str] | str) -> bool:
    """Return True when a JSONL document matches the requested doc types."""
    if doc_types == "all":
        return True
    return doc.get("type") in set(doc_types)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield dictionaries from a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def doc_id(doc: Dict[str, Any]) -> str:
    """Return a stable document identifier for logging."""
    return str(doc.get("document_id") or doc.get("id") or "")


def token_proxy(doc: Dict[str, Any]) -> int:
    """Estimate document size with a lightweight token proxy."""
    chunks: List[str] = []
    text = doc.get("text")
    if isinstance(text, str):
        chunks.append(text)
    title = doc.get("title")
    if isinstance(title, str):
        chunks.append(title)
    sentences = doc.get("sentences")
    if isinstance(sentences, list):
        for s in sentences:
            if isinstance(s, str):
                chunks.append(s)
            elif isinstance(s, list):
                chunks.append(" ".join(str(x) for x in s))
    tokens = doc.get("tokens")
    if isinstance(tokens, list):
        flat: List[str] = []
        for item in tokens:
            if isinstance(item, str):
                flat.append(item)
            elif isinstance(item, list):
                flat.extend(str(x) for x in item)
        if flat:
            return len(flat)
    return len("\n".join(chunks).split())


def eligible_docs(input_path: Path, doc_types_arg: str) -> List[Tuple[int, Dict[str, Any]]]:
    """Return eligible docs with their eligible index after doc-type filtering."""
    doc_types = parse_doc_types(doc_types_arg)
    out: List[Tuple[int, Dict[str, Any]]] = []
    eligible_index = 0
    for doc in iter_jsonl(input_path):
        if should_keep_doc(doc, doc_types):
            out.append((eligible_index, doc))
            eligible_index += 1
    return out


def count_jsonl(path: Path) -> int:
    """Count non-empty JSONL rows."""
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def write_runtime_config(root: Path, source_config: Path, metrics_dir: Path) -> Path:
    """
    Create a runtime config that redirects processed outputs to supplementary metrics.

    This avoids overwriting benchmark predictions under the normal data_processed path.
    """
    cfg = load_yaml(source_config)
    cfg.setdefault("paths", {})["data_processed"] = str(metrics_dir / "runtime_predictions")
    cfg["paths"]["supplementary_metrics"] = str(metrics_dir)
    out = metrics_dir / "runtime_measurement_marag_missing_config.yaml"
    write_yaml(out, cfg)
    return out


def route_error(error_message: str) -> bool:
    """Detect endpoint route errors that should not trigger doc retries."""
    msg = error_message or ""
    return "404 Client Error" in msg and "/chat/completions" in msg


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoint fallback for subprocesses
# ---------------------------------------------------------------------------

def child_launcher_command(script: str, script_args: Sequence[str], endpoint_mode: str, endpoint_verbose: bool) -> List[str]:
    """Run an existing script under a small launcher that installs an endpoint fallback."""
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
    env = os.environ.copy()
    env["RAGTREE_SUPPLEMENTARY_CHAT_ENDPOINT_MODE"] = endpoint_mode
    env["RAGTREE_SUPPLEMENTARY_ENDPOINT_VERBOSE"] = "1" if endpoint_verbose else "0"
    return [sys.executable, "-c", launcher, script, *script_args]


# ---------------------------------------------------------------------------
# Measurement and outputs
# ---------------------------------------------------------------------------

def build_marag_command(spec: MaragSpec, runtime_cfg: Path, skip: int, limit: int, backend: str, max_llm_calls: int) -> List[str]:
    """Build the script arguments for one MARAG run."""
    return [
        "scripts/run_marag_relations.py",
        "--dataset-key", spec.dataset,
        "--backend", backend,
        "--doc-types", spec.doc_types,
        "--ontology-key", spec.ontology_key,
        "--ontology-links-path", spec.ontology_links_path,
        "--kg-path", spec.kg_path,
        "--enable-web",
        "--enable-wikidata",
        "--max-llm-calls", str(max_llm_calls),
        "--shot-num", "3",
        "--shot-type", spec.shot_type,
        "--config", str(runtime_cfg),
        "--skip", str(skip),
        "--limit", str(limit),
    ]


def run_subprocess(
    command_args: List[str],
    *,
    root: Path,
    log_path: Path,
    endpoint_mode: str,
    endpoint_verbose: bool,
) -> Tuple[int, float, str, str, str]:
    """Execute a command with endpoint fallback and return status plus captured output."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    script = command_args[0]
    script_args = command_args[1:]
    cmd = child_launcher_command(script, script_args, endpoint_mode, endpoint_verbose)
    env = os.environ.copy()
    env["RAGTREE_SUPPLEMENTARY_CHAT_ENDPOINT_MODE"] = endpoint_mode
    env["RAGTREE_SUPPLEMENTARY_ENDPOINT_VERBOSE"] = "1" if endpoint_verbose else "0"

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.perf_counter() - t0
    output = proc.stdout or ""
    log_path.write_text(output, encoding="utf-8")

    error_type = ""
    error_message = ""
    if proc.returncode != 0:
        # Keep this simple and robust. The logs contain the full traceback.
        error_type = "subprocess.CalledProcessError"
        tail_lines = [line for line in output.splitlines() if line.strip()][-20:]
        error_message = " | ".join(tail_lines[-4:]) if tail_lines else f"return code {proc.returncode}"
        # Try to extract the final exception line from Python tracebacks.
        for line in reversed(tail_lines):
            if ":" in line and not line.startswith("File ") and not line.startswith("Traceback"):
                error_message = line.strip()
                break
    return proc.returncode, elapsed, output, error_type, error_message


def compute_metrics(elapsed: Optional[float], eligible: int, power_kw: float, carbon_intensity: float) -> Dict[str, Optional[float]]:
    """Compute per-doc and extrapolated energy/CO2 metrics."""
    if elapsed is None:
        return {
            "elapsed_seconds_doc": None,
            "elapsed_minutes_doc": None,
            "energy_kwh_doc": None,
            "co2_kg_doc": None,
            "co2_g_doc": None,
            "estimated_total_seconds": None,
            "estimated_total_minutes": None,
            "estimated_total_hours": None,
            "estimated_total_energy_kwh": None,
            "estimated_total_co2_kg": None,
            "estimated_total_co2_g": None,
        }
    energy_doc = (elapsed / 3600.0) * power_kw
    co2_kg_doc = energy_doc * carbon_intensity
    total_seconds = elapsed * eligible
    total_energy = (total_seconds / 3600.0) * power_kw
    total_co2_kg = total_energy * carbon_intensity
    return {
        "elapsed_seconds_doc": elapsed,
        "elapsed_minutes_doc": elapsed / 60.0,
        "energy_kwh_doc": energy_doc,
        "co2_kg_doc": co2_kg_doc,
        "co2_g_doc": co2_kg_doc * 1000.0,
        "estimated_total_seconds": total_seconds,
        "estimated_total_minutes": total_seconds / 60.0,
        "estimated_total_hours": total_seconds / 3600.0,
        "estimated_total_energy_kwh": total_energy,
        "estimated_total_co2_kg": total_co2_kg,
        "estimated_total_co2_g": total_co2_kg * 1000.0,
    }


def row_fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
    """Return stable CSV fieldnames for metric rows."""
    preferred = [
        "status", "dry_run", "success", "return_code", "error_type", "error_message",
        "method_family", "method", "dataset", "command_dataset_key", "script", "run_mode",
        "doc_filter_arg", "doc_filter_value", "n_total_jsonl_documents", "n_eligible_documents",
        "sample_index_requested", "sample_index_effective", "sample_document_id",
        "sample_document_type", "sample_document_token_proxy", "sample_strategy", "max_sample_attempts",
        "attempts", "failed_attempts", "attempt_summaries_json", "power_kw",
        "carbon_intensity_kg_per_kwh", "carbon_intensity_g_per_kwh", "input_jsonl",
        "runtime_config", "log_path", "notebook_source", "note", "command",
        "notebook_like_command", "created_at_utc", "elapsed_seconds_doc", "elapsed_minutes_doc",
        "energy_kwh_doc", "co2_kg_doc", "co2_g_doc", "estimated_total_seconds",
        "estimated_total_minutes", "estimated_total_hours", "estimated_total_energy_kwh",
        "estimated_total_co2_kg", "estimated_total_co2_g",
    ]
    keys = set()
    for row in rows:
        keys.update(row.keys())
    return [k for k in preferred if k in keys] + sorted(keys.difference(preferred))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = row_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write rows to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    """Read CSV rows if the file exists."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(row: Dict[str, Any], key: str) -> Optional[float]:
    val = row.get(key)
    if val is None or val == "":
        return None
    try:
        return float(val)
    except Exception:
        return None


def _int(row: Dict[str, Any], key: str) -> Optional[int]:
    val = row.get(key)
    if val is None or val == "":
        return None
    try:
        return int(float(val))
    except Exception:
        return None


def combine_agentic_rows(metrics_dir: Path, missing_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge the previous runtime CSV with the new MARAG rows and keep only
    ARAG, SimpleAgenticRAG, and MARAG.
    """
    base_path = metrics_dir / "runtime_co2_by_method_dataset.csv"
    base_rows = read_csv_rows(base_path)
    combined: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for row in base_rows:
        if row.get("method") in METHOD_ORDER and row.get("dataset") in DATASET_ORDER:
            combined[(row["dataset"], row["method"])] = row

    # New rows override any previous MARAG rows for the same dataset.
    for row in missing_rows:
        if row.get("method") in METHOD_ORDER and row.get("dataset") in DATASET_ORDER:
            combined[(row["dataset"], row["method"])] = row

    out: List[Dict[str, Any]] = []
    for ds in DATASET_ORDER:
        for method in METHOD_ORDER:
            row = combined.get((ds, method))
            if row is not None:
                out.append(row)
    return out


def fmt_number(value: Optional[float], digits: int = 2) -> str:
    """Format a number for LaTeX, using -- for missing values."""
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def docs_for_dataset(rows: List[Dict[str, Any]], dataset: str) -> str:
    """Pick the document count for a dataset block."""
    for row in rows:
        if row.get("dataset") == dataset:
            n = _int(row, "n_eligible_documents")
            if n is not None:
                return str(n)
    return "--"


def make_agentic_latex_table(rows: List[Dict[str, Any]]) -> str:
    """Build the compact LaTeX table with only ARAG, SimpleAgenticRAG, and MARAG."""
    by_key = {(r.get("dataset", ""), r.get("method", "")): r for r in rows}
    lines: List[str] = []
    lines.append(r"\begin{table*}[pos=htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Runtime and estimated CO2 indicators for the agentic RAG methods grouped by dataset. Each entry reports time per document, extrapolated total runtime in hours, and extrapolated CO2 in grams.}")
    lines.append(r"\label{tab:runtime_co2_agentic_grouped_by_dataset}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.05}")
    lines.append(r"\begin{tabularx}{\textwidth}{p{2.2cm} p{3.4cm} r r r}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Dataset} & \textbf{Method} & \textbf{Time/doc. (s)} & \textbf{Est. total (h)} & \textbf{Est. CO2 (g)} \\")
    lines.append(r"\midrule")

    for ds_i, dataset in enumerate(DATASET_ORDER):
        if ds_i > 0:
            lines.append(r"\midrule")
        display = DATASET_DISPLAY[dataset]
        docs = docs_for_dataset(rows, dataset)
        lines.append(rf"\multicolumn{{5}}{{l}}{{\textbf{{{display}}} \hfill \textbf{{{docs} documents}}}} \\")
        lines.append(r"\midrule")
        for method in METHOD_ORDER:
            row = by_key.get((dataset, method), {})
            name = METHOD_DISPLAY[method]
            t = fmt_number(_float(row, "elapsed_seconds_doc"), 2)
            h = fmt_number(_float(row, "estimated_total_hours"), 2)
            co2 = fmt_number(_float(row, "estimated_total_co2_g"), 2)
            lines.append(rf"{display} & {name} & {t} & {h} & {co2} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def make_notebook_like_command(spec: MaragSpec, backend: str, max_llm_calls: int) -> str:
    """Create a readable notebook-style command for traceability."""
    parts = [
        r'%run "scripts/run_marag_relations.py"',
        f"--dataset-key {spec.dataset}",
        f"--backend {backend}",
        f"--doc-types {spec.doc_types}",
        f"--ontology-key {spec.ontology_key}",
        f"--ontology-links-path {spec.ontology_links_path}",
        f"--kg-path {spec.kg_path}",
        "--enable-web",
        "--enable-wikidata",
        f"--max-llm-calls {max_llm_calls}",
        "--shot-num 3",
        f"--shot-type {spec.shot_type}",
    ]
    return " \\\n  ".join(parts)


def run_measurements(args: argparse.Namespace) -> int:
    """Main measurement routine."""
    root = find_repo_root()
    source_cfg = root / "configs" / "default.yaml"
    metrics_dir = root / "data" / "suplementary_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = metrics_dir / "logs_marag_missing"
    runtime_cfg = write_runtime_config(root, source_cfg, metrics_dir)

    cfg = load_yaml(runtime_cfg)
    ds_pre = cfg.get("datasets", {}).get("preprocessed", {})
    if not isinstance(ds_pre, dict):
        raise KeyError("Missing datasets.preprocessed in runtime config")

    selected_specs = MISSING_MARAG_SPECS
    if args.only_dataset:
        wanted = set(args.only_dataset)
        selected_specs = [s for s in selected_specs if s.dataset in wanted or s.display_dataset in wanted]

    print(f"[missing-marag] root={root}")
    print(f"[missing-marag] source config={source_cfg}")
    print(f"[missing-marag] runtime config={runtime_cfg}")
    print(f"[missing-marag] output dir={metrics_dir}")
    print(f"[missing-marag] selected missing MARAG runs={len(selected_specs)}")
    print(f"[missing-marag] power_kw={args.power_kw}")
    print(f"[missing-marag] carbon_intensity_kg_per_kwh={args.carbon_intensity}")
    print(f"[missing-marag] sample_strategy={args.sample_strategy}")
    print(f"[missing-marag] max_sample_attempts={args.max_sample_attempts}")

    rows: List[Dict[str, Any]] = []

    for run_idx, spec in enumerate(selected_specs, start=1):
        input_value = ds_pre.get(spec.dataset)
        if not input_value:
            raise KeyError(f"Missing datasets.preprocessed.{spec.dataset} in config")
        input_path = as_repo_path(root, input_value)
        total_docs = count_jsonl(input_path)
        eligible = eligible_docs(input_path, spec.doc_types)
        n_eligible = len(eligible)
        if n_eligible == 0:
            raise RuntimeError(f"No eligible docs for {spec.dataset} with doc-types={spec.doc_types}")

        start_idx = max(0, int(args.sample_index))
        candidates = eligible[start_idx : start_idx + max(1, int(args.max_sample_attempts))]
        if not candidates:
            candidates = eligible[: max(1, int(args.max_sample_attempts))]

        print(
            f"[{run_idx}/{len(selected_specs)}] MARAG | {spec.dataset} | "
            f"eligible={n_eligible} | first_sample_skip={candidates[0][0]} | "
            f"doc_id={doc_id(candidates[0][1])}"
        )

        final_row: Optional[Dict[str, Any]] = None
        attempt_summaries: List[Dict[str, Any]] = []
        failed_attempts = 0

        for attempt_no, (candidate_skip, candidate_doc) in enumerate(candidates, start=1):
            log_path = logs_dir / f"marag__{spec.dataset}__attempt{attempt_no:02d}__skip{candidate_skip}.log"
            command_args = build_marag_command(
                spec,
                runtime_cfg=runtime_cfg,
                skip=candidate_skip,
                limit=1,
                backend=args.backend,
                max_llm_calls=args.max_llm_calls,
            )
            command_str = " ".join(command_args)

            base_row: Dict[str, Any] = {
                "status": "DRY-RUN" if args.dry_run else "RUNNING",
                "dry_run": bool(args.dry_run),
                "success": False,
                "return_code": None,
                "error_type": "",
                "error_message": "",
                "method_family": "Agentic RAG",
                "method": "marag",
                "dataset": spec.dataset,
                "command_dataset_key": spec.dataset,
                "script": "scripts/run_marag_relations.py",
                "run_mode": "script",
                "doc_filter_arg": "--doc-types",
                "doc_filter_value": spec.doc_types,
                "n_total_jsonl_documents": total_docs,
                "n_eligible_documents": n_eligible,
                "sample_index_requested": int(args.sample_index),
                "sample_index_effective": candidate_skip,
                "sample_document_id": doc_id(candidate_doc),
                "sample_document_type": candidate_doc.get("type", ""),
                "sample_document_token_proxy": token_proxy(candidate_doc),
                "sample_strategy": args.sample_strategy,
                "max_sample_attempts": int(args.max_sample_attempts),
                "attempts": attempt_no,
                "failed_attempts": failed_attempts,
                "attempt_summaries_json": "[]",
                "power_kw": float(args.power_kw),
                "carbon_intensity_kg_per_kwh": float(args.carbon_intensity),
                "carbon_intensity_g_per_kwh": float(args.carbon_intensity) * 1000.0,
                "input_jsonl": str(input_path),
                "runtime_config": str(runtime_cfg),
                "log_path": str(log_path),
                "notebook_source": spec.notebook_source,
                "note": "Additional MARAG runtime/CO2 measurement for a method-dataset pair missing from the first supplementary pass.",
                "command": command_str,
                "notebook_like_command": make_notebook_like_command(spec, args.backend, args.max_llm_calls),
                "created_at_utc": utc_now(),
                "chat_endpoint_mode": args.chat_endpoint_mode,
            }

            if args.dry_run:
                base_row.update(compute_metrics(None, n_eligible, args.power_kw, args.carbon_intensity))
                base_row["status"] = "DRY-RUN"
                base_row["return_code"] = 0
                attempt_summaries.append({
                    "attempt": attempt_no,
                    "dry_run": True,
                    "sample_index_effective": candidate_skip,
                    "sample_document_id": doc_id(candidate_doc),
                })
                base_row["attempt_summaries_json"] = json.dumps(attempt_summaries, ensure_ascii=False)
                final_row = base_row
                print("    -> DRY-RUN: command resolved but not executed.")
                break

            return_code, elapsed, output, error_type, error_message = run_subprocess(
                command_args,
                root=root,
                log_path=log_path,
                endpoint_mode=args.chat_endpoint_mode,
                endpoint_verbose=args.endpoint_fallback_verbose,
            )
            success = return_code == 0
            attempt_summaries.append({
                "attempt": attempt_no,
                "success": success,
                "return_code": return_code,
                "elapsed_seconds": elapsed,
                "sample_index_effective": candidate_skip,
                "sample_document_id": doc_id(candidate_doc),
                "sample_document_type": candidate_doc.get("type", ""),
                "sample_document_token_proxy": token_proxy(candidate_doc),
                "log_path": str(log_path),
                "error_type": error_type,
                "error_message": error_message,
            })

            if success:
                base_row.update(compute_metrics(elapsed, n_eligible, args.power_kw, args.carbon_intensity))
                base_row.update({
                    "status": "OK",
                    "success": True,
                    "return_code": 0,
                    "error_type": "",
                    "error_message": "",
                    "attempts": attempt_no,
                    "failed_attempts": failed_attempts,
                    "attempt_summaries_json": json.dumps(attempt_summaries, ensure_ascii=False),
                })
                final_row = base_row
                print(
                    f"    -> OK: elapsed={elapsed:.2f}s, "
                    f"estimated_total={base_row['estimated_total_minutes']:.2f}min, "
                    f"estimated_total_co2={base_row['estimated_total_co2_g']:.4f}g"
                )
                break

            failed_attempts += 1
            print(f"    -> failed attempt {attempt_no}: elapsed={elapsed:.2f}s, error={error_message}")
            if route_error(error_message):
                print("    -> endpoint route error detected; not retrying more documents.")
                break
            if args.sample_strategy != "first-success":
                break
            if attempt_no < len(candidates):
                next_skip, next_doc = candidates[attempt_no]
                print(f"    -> retry attempt {attempt_no + 1}/{len(candidates)}: sample_skip={next_skip} | doc_id={doc_id(next_doc)}")

            base_row.update(compute_metrics(None, n_eligible, args.power_kw, args.carbon_intensity))
            base_row.update({
                "status": "FAILED",
                "success": False,
                "return_code": return_code,
                "error_type": error_type,
                "error_message": error_message,
                "attempts": attempt_no,
                "failed_attempts": failed_attempts,
                "attempt_summaries_json": json.dumps(attempt_summaries, ensure_ascii=False),
            })
            final_row = base_row

        if final_row is not None:
            rows.append(final_row)

    # Write missing MARAG rows.
    missing_csv = metrics_dir / "runtime_co2_missing_marag.csv"
    missing_json = metrics_dir / "runtime_co2_missing_marag.json"
    write_csv(missing_csv, rows)
    write_json(missing_json, rows)

    # Build combined agentic-only table, using the previous big CSV if available.
    combined_rows = combine_agentic_rows(metrics_dir, rows)
    combined_csv = metrics_dir / "runtime_co2_agentic_only_complete.csv"
    combined_json = metrics_dir / "runtime_co2_agentic_only_complete.json"
    combined_tex = metrics_dir / "runtime_co2_agentic_only_table.tex"
    write_csv(combined_csv, combined_rows)
    write_json(combined_json, combined_rows)
    combined_tex.write_text(make_agentic_latex_table(combined_rows), encoding="utf-8")

    print("[missing-marag] wrote:")
    print(f"  - {missing_csv}")
    print(f"  - {missing_json}")
    print(f"  - {combined_csv}")
    print(f"  - {combined_json}")
    print(f"  - {combined_tex}")
    print(f"  - logs: {logs_dir}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    p = argparse.ArgumentParser(description="Measure missing MARAG runtime/CO2 values for supplementary metrics.")
    p.add_argument("--dry-run", action="store_true", help="Resolve commands and write dry-run rows without executing MARAG.")
    p.add_argument("--only-dataset", nargs="*", default=None, help="Optional dataset keys/display names to run, e.g., maven_ere causalbank.")
    p.add_argument("--backend", default="vllm", help="LLM backend passed to run_marag_relations.py.")
    p.add_argument("--sample-index", type=int, default=0, help="Eligible-document index to measure.")
    p.add_argument("--sample-strategy", choices=["fixed", "first-success"], default="first-success", help="Retry later eligible documents if the selected sample fails.")
    p.add_argument("--max-sample-attempts", type=int, default=20, help="Maximum candidate docs to try in first-success mode.")
    p.add_argument("--max-llm-calls", type=int, default=10, help="MARAG --max-llm-calls value.")
    p.add_argument("--power-kw", type=float, default=0.300, help="Estimated system power draw in kW.")
    p.add_argument("--carbon-intensity", type=float, default=0.045, help="Carbon intensity in kgCO2/kWh.")
    p.add_argument(
        "--chat-endpoint-mode",
        choices=["auto", "none", "no-fallback", "v1", "root", "api-v1", "responses", "completions"],
        default="auto",
        help="Endpoint fallback mode for local OpenAI-compatible servers.",
    )
    p.add_argument("--endpoint-fallback-verbose", action="store_true", help="Print endpoint fallback route changes from child scripts.")
    return p


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()
    raise SystemExit(run_measurements(args))


if __name__ == "__main__":
    main()
