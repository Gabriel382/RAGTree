#!/usr/bin/env python3
"""Replay exactly one saved failed MAVEN-ERE prompt through OpenRouter DeepInfra.

Defaults to dry run. It never modifies the KBS SQLite database, checkpoints,
source files, protocol, or existing LLM traces. Requires OPENROUTER_API_KEY only
for --execute. This is a paid diagnostic when --execute is set.

Commands:
  python kbs_probe_failed_maven_deepinfra.py --repo .
  python kbs_probe_failed_maven_deepinfra.py --repo . --execute
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import urllib.error
import urllib.parse
import urllib.request

SESSION = "2621e33f04de452eb50a63baaf03f990"
MODEL = "openai/gpt-oss-20b"
RUN_ID = "cac752a698530190bd25ce0e"
ALLOW_PARAMS = (
    "temperature", "top_p", "max_tokens", "seed", "stop",
    "frequency_penalty", "presence_penalty", "reasoning_effort",
    "response_format", "tools", "tool_choice", "min_p", "top_k",
)


def json_read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def find_failure_trace(repo: Path, index: int, session: str):
    database = repo / "results/kbs_journal/checkpoints.sqlite"
    uri = "file:" + urllib.parse.quote(str(database.resolve())) + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        failures = conn.execute(
            "SELECT doc_id,run_id FROM failures WHERE session_id=? ORDER BY updated",
            (session,),
        ).fetchall()
    finally:
        conn.close()
    if len(failures) != 4:
        raise RuntimeError(f"Expected exactly 4 saved failures in session, found {len(failures)}")
    if not 1 <= index <= len(failures):
        raise ValueError("--index must be between 1 and 4")
    doc_id, run_id = failures[index - 1]
    if run_id != RUN_ID:
        raise RuntimeError("Unexpected run ID in failure record; refusing ambiguous replay")
    run = repo / "results/kbs_journal/openrouter/main_gpt/baseline/maven_ere" / run_id
    if not run.is_dir():
        raise FileNotFoundError(f"Expected run directory not found: {run}")
    folder = run / "calls" / hashlib.sha256(doc_id.encode()).hexdigest()[:24]
    candidates = sorted(folder.glob(session + "-*.json")) if folder.is_dir() else []
    if not candidates:
        for p in (run / "calls").glob("*/" + session + "-*.json"):
            try:
                if json_read(p).get("document_id") == doc_id:
                    candidates.append(p)
            except (OSError, ValueError):
                continue
    for p in sorted(candidates):
        trace = json_read(p)
        if (trace.get("request") or {}).get("messages"):
            return doc_id, p, trace
    raise RuntimeError(f"No saved prompt trace for {doc_id}; refusing reconstruction")


def build_request(trace: dict, cap: int, provider: str) -> dict:
    saved = trace["request"]
    messages = saved.get("messages")
    parameters = saved.get("parameters") or {}
    if not isinstance(messages, list) or not messages:
        raise ValueError("Saved messages are missing")
    if parameters.get("reasoning_effort") is None:
        raise ValueError("Saved reasoning effort is missing")
    request = {"model": MODEL, "messages": messages}
    for key in ALLOW_PARAMS:
        if parameters.get(key) is not None:
            request[key] = parameters[key]
    request["max_tokens"] = cap
    request["provider"] = {
        "only": [provider],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    return request


def get_json(url: str, headers: dict, payload: dict | None = None, timeout: int = 180):
    req = urllib.request.Request(
        url,
        headers=headers,
        method="POST" if payload is not None else "GET",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return json.load(res)
    except urllib.error.HTTPError as exc:
        err = exc.read(1200).decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code}: {err}") from exc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--session", default=SESSION)
    ap.add_argument("--index", type=int, default=1, help="Which of the four failure rows (1 to 4)")
    ap.add_argument("--provider", default="deepinfra", choices=["deepinfra", "coreweave"],
                    help="Provider to pin; no fallback to other providers")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    if not 1 <= args.max_tokens <= 8192:
        ap.error("--max-tokens must be between 1 and 8192")
    doc_id, path, trace = find_failure_trace(args.repo.resolve(), args.index, args.session)
    request = build_request(trace, args.max_tokens, args.provider)
    messages_hash = hashlib.sha256(json.dumps(request["messages"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    print(f"Selected failure: {args.index}/4 | {doc_id}")
    print(f"Saved trace: {path.name} | messages_sha256={messages_hash}")
    print(f"Model: {request['model']} | provider only: {args.provider} | allow_fallbacks: false")
    print(f"max_tokens={request['max_tokens']} | reasoning_effort={request.get('reasoning_effort')}")
    if not args.execute:
        print("DRY RUN. No inference, cost, or checkpoint edits. Add --execute for ONE paid request.")
        return 0
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        ap.error("OPENROUTER_API_KEY not loaded; source ~/.config/ragtree/openrouter.env")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    result = get_json("https://openrouter.ai/api/v1/chat/completions", headers, request, timeout=600)
    choice = (result.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    valid_json = False
    if isinstance(content, str) and content.strip():
        try:
            valid_json = isinstance(json.loads(content.strip()), dict)
        except (ValueError, TypeError):
            pass
    usage = result.get("usage") or {}
    gen_id = result.get("id")
    provider_info = {}
    if gen_id:
        try:
            meta = get_json("https://openrouter.ai/api/v1/generation?" + urllib.parse.urlencode({"id": gen_id}),
                            headers, timeout=30)
            provider_info = meta.get("data") or {}
        except Exception as exc:
            provider_info = {"lookup_error": str(exc)}
    print("finish_reason:", choice.get("finish_reason"))
    print("completion_tokens:", usage.get("completion_tokens"))
    print("reasoning_tokens:", (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))
    print("visible_chars:", len(content) if isinstance(content, str) else 0)
    print("JSON_object:", valid_json)
    print("provider:", provider_info.get("provider_name", "not reported"))
    print("OpenRouter total_cost_usd:", provider_info.get("total_cost", "not reported"))
    # No prompt content in summary. Retain raw response for audit, outside tracked source.
    out = args.repo.resolve() / "results/kbs_journal/diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"failed_maven_provider_probe_{timestamp}_{args.index}"
    (out / f"{stem}.json").write_text(json.dumps({
        "document_id": doc_id, "trace_source": str(path), "model": MODEL,
        "messages_sha256": messages_hash, "request_parameters": {k: v for k, v in request.items() if k != "messages"},
        "response": result, "provider_metadata": {k: provider_info.get(k) for k in
            ("provider_name", "model", "total_cost", "native_tokens_completion")},
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Saved diagnostic:", out / f"{stem}.json")
    ok = choice.get("finish_reason") == "stop" and valid_json and len(content.strip()) > 0
    print("Usable response:", ok)
    print("NOTE: This replay does NOT mark the failed checkpoint done; only the pipeline can do so.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
