#!/usr/bin/env python3
"""Replay the *exact three failed KBS/DocRED requests* against local vLLM.

The original messages are loaded verbatim from KBS per-call traces. Only the
completion ceiling changes (4096 -> 8192 by default). This diagnostic never
modifies the KBS checkpoint database, original traces, datasets or protocol.

Usage (from your RAGTree repository root):
  python kbs_replay_failed_docred.py --repo .               # dry run, no inference
  python kbs_replay_failed_docred.py --repo . --execute     # EXACTLY 3 POST requests
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import re
import sys
import urllib.error
import urllib.request
import uuid

DEFAULT_SESSION = "304d4e7f88f641f6a9f4cd09860f7914"
TARGETS = (
    ("baseline", "DocRED - 832aa7b8fd581e20", "c360602695f194c45129f00e"),
    ("langgraph_agentic_simple", "DocRED - e8f3abcd8da9ebb2", "d47be6c460c815303acc7ce8"),
    ("langgraph_agentic_simple", "DocRED - e92387cc98263d80", "d47be6c460c815303acc7ce8"),
)
# Only parameters accepted by the OpenAI-compatible chat completions API. Local
# LiteLLM controls (timeout, retries, api_base, api_key) are never forwarded.
REQUEST_FIELDS = (
    "temperature", "top_p", "max_tokens", "seed", "stop",
    "frequency_penalty", "presence_penalty", "repetition_penalty",
    "reasoning_effort", "response_format", "tools", "tool_choice",
    "parallel_tool_calls", "min_p", "top_k",
)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def locate_trace(root: Path, method: str, doc_id: str, run_id: str, session: str):
    # The schema used by trace_call() stores the per-record SHA256 directory.
    folder = root / "local" / "main_gpt" / method / "docred_causal" / run_id / "calls" / sha256(doc_id.encode("utf-8")).hexdigest()[:24]
    if not folder.is_dir():
        raise FileNotFoundError(f"No per-document traces for {doc_id}: {folder}")
    candidates = []
    for path in sorted(folder.glob(f"{session}-*.json")):
        obj = read_json(path)
        req = obj.get("request") or {}
        resp = obj.get("response") or {}
        choices = resp.get("choices") or []
        choice = choices[0] if choices else {}
        content = (choice.get("message") or {}).get("content")
        if (req.get("messages") and choice.get("finish_reason") == "length" and
                not (isinstance(content, str) and content.strip())):
            candidates.append((path, obj))
    if not candidates:
        raise ValueError(f"No saved reasoning-only length-truncated request for {doc_id} in session {session}")
    # Agentic methods can make multiple calls; the last truncated one is the
    # call associated with the eventual error. Pick by recorded timestamp.
    path, trace = max(candidates, key=lambda it: (float(it[1].get("timestamp") or 0), it[0].name))
    return path, trace


def as_request(trace: dict, model: str, cap: int):
    saved = trace["request"]
    messages = saved.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Saved messages are missing or invalid")
    parameters = saved.get("parameters") or {}
    request = {"model": model, "messages": messages}
    for field in REQUEST_FIELDS:
        if field in parameters and parameters[field] is not None:
            request[field] = parameters[field]
    request["max_tokens"] = cap
    if "temperature" not in request:
        raise ValueError("Original temperature not recorded; refusing to infer")
    if "reasoning_effort" not in request:
        raise ValueError("Original reasoning_effort not recorded; refusing to infer")
    return request


def extract_content(response: dict):
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(x.get("text", "") for x in content if isinstance(x, dict) and isinstance(x.get("text"), str))
    return ""


def validate_answer(content: str, messages: list):
    result = {"nonempty": bool(content.strip()), "json_valid": False,
              "relation_pairs_well_formed": False, "entity_ids_present_in_prompt": None,
              "issues": []}
    if not content.strip():
        result["issues"].append("Empty final answer")
        return result
    cleaned = content.strip()
    match = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, flags=re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        result["issues"].append(f"JSON parsing failed: {exc}")
        return result
    result["json_valid"] = True
    if not isinstance(obj, dict):
        result["issues"].append("Top-level JSON must be a relation-label dictionary")
        return result
    problems = []
    ids = []
    total_pairs = 0
    for label, pairs in obj.items():
        if not isinstance(label, str) or not isinstance(pairs, list):
            problems.append(f"Invalid relation label/list: {label!r}")
            continue
        for i, pair in enumerate(pairs):
            if not (isinstance(pair, list) and len(pair) == 2 and
                    all(isinstance(value, str) for value in pair)):
                problems.append(f"Invalid pair at {label}[{i}]: {pair!r}")
                continue
            total_pairs += 1
            ids.extend(pair)
    result["pair_count"] = total_pairs
    result["relation_labels"] = sorted(str(x) for x in obj)
    result["relation_pairs_well_formed"] = not problems
    result["issues"].extend(problems[:20])
    prompt_text = "\n".join(json.dumps(msg, ensure_ascii=False) for msg in messages)
    missing = sorted(set(ident for ident in ids if ident not in prompt_text))
    result["entity_ids_present_in_prompt"] = len(missing) == 0
    if missing:
        result["issues"].append(f"Predicted IDs absent from original prompt: {missing[:10]}")
    # NOTE: This is structural validation, NOT the dataset's official schema or
    # gold-relation score. Those checks belong in the full evaluator.
    return result


def fetch_json(request: urllib.request.Request, timeout: float):
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read(1500).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--execute", action="store_true", help="Actually issue one local inference request per failed record")
    args = parser.parse_args(argv)
    if args.api_base.rstrip("/").split("/v1")[0] not in ("http://localhost:8000", "http://127.0.0.1:8000"):
        parser.error("Only local vLLM on port 8000 is allowed by this diagnostic")
    if not 1 <= args.max_tokens <= 16384:
        parser.error("--max-tokens must be between 1 and 16384")
    if not 1 <= args.timeout <= 1800:
        parser.error("--timeout must be between 1 and 1800 seconds")
    root = args.repo.resolve() / "results" / "kbs_journal"
    source = []
    for method, doc_id, run_id in TARGETS:
        path, trace = locate_trace(root, method, doc_id, run_id, args.session)
        source.append((method, doc_id, run_id, path, trace))
    print(f"Found all {len(source)} exact failed request traces. Replaying will NOT alter experiment checkpoints.")
    for method, doc_id, run_id, path, trace in source:
        params = trace["request"].get("parameters") or {}
        usage = (trace.get("response") or {}).get("usage") or {}
        print(f"  {method}: {doc_id} | original max={params.get('max_tokens')} | original output={usage.get('completion_tokens')} | source={path.name}")
    if not args.execute:
        print("DRY RUN: no model requests sent. Add --execute to replay exactly three requests.")
        return 0
    endpoint = args.api_base.rstrip("/")
    headers = {"Authorization": "Bearer EMPTY", "Content-Type": "application/json"}
    models = fetch_json(urllib.request.Request(endpoint + "/models", headers=headers), args.timeout)
    model_ids = [row.get("id") for row in models.get("data", [])]
    if "openai/gpt-oss-20b" not in model_ids:
        raise RuntimeError(f"The local server does not expose openai/gpt-oss-20b: {model_ids!r}")
    directory = root / "diagnostics" / ("failed_docred_replay_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    summary = []
    for i, (method, doc_id, run_id, path, trace) in enumerate(source, 1):
        safe_name = f"{i}_{method}"
        request_obj = as_request(trace, "openai/gpt-oss-20b", args.max_tokens)
        # Preserve the exact saved prompt; only model alias and max_tokens
        # change. LiteLLM-only parameters are not part of the vLLM request.
        prompt_sha = sha256(json.dumps(request_obj["messages"], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        print(f"[{i}/3] {method} {doc_id} | messages_sha256={prompt_sha[:16]}", flush=True)
        item = {"method": method, "document_id": doc_id, "run_id": run_id,
                "trace_file": str(path), "messages_sha256": prompt_sha,
                "original_parameters": trace["request"].get("parameters"),
                "replay_parameters": {k: v for k, v in request_obj.items() if k != "messages"}}
        try:
            req = urllib.request.Request(endpoint + "/chat/completions", method="POST",
                data=json.dumps(request_obj, ensure_ascii=False).encode("utf-8"), headers=headers)
            response = fetch_json(req, args.timeout)
            content = extract_content(response)
            validation = validate_answer(content, request_obj["messages"])
            choice = (response.get("choices") or [{}])[0]
            item.update({"status": "completed", "finish_reason": choice.get("finish_reason"),
                "usage": response.get("usage"), "validation": validation, "answer": content})
            # Preserve all returned tokens including reasoning, if vLLM exposed it.
            (directory / f"{safe_name}_response.json").write_text(json.dumps(response, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"   finish={item['finish_reason']!r} usage={item['usage']} JSON={validation['json_valid']} pairs={validation['relation_pairs_well_formed']} IDs_in_prompt={validation['entity_ids_present_in_prompt']}", flush=True)
        except Exception as exc:
            item.update({"status": "failed", "error": repr(exc)})
            print(f"   ERROR: {exc}", file=sys.stderr, flush=True)
        (directory / f"{safe_name}_summary.json").write_text(json.dumps(item, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        summary.append(item)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"\nResults saved in {directory}")
    n = sum(x.get("status") == "completed" and x.get("finish_reason") == "stop" and
            x.get("validation", {}).get("json_valid") and x.get("validation", {}).get("relation_pairs_well_formed") for x in summary)
    print(f"Valid, normally completed replays: {n}/3")
    if n < 3:
        print("Do not enable automatic fallback yet; inspect failed replay responses.")
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
