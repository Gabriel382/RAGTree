#!/usr/bin/env python3
"""Read-only inspection of OpenRouter KBS failure traces (no model requests).

Usage:
  python kbs_openrouter_failure_audit.py --session 2621e33f04de452eb50a63baaf03f990
  python kbs_openrouter_failure_audit.py --session 2621e33f04de452eb50a63baaf03f990 --lookup-provider

The optional provider lookup calls only GET /api/v1/generation (metadata), never chat/completions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sqlite3
import urllib.parse
import urllib.request
from pathlib import Path


def get_provider_metadata(generation_id: str) -> dict:
    key = os.environ.get('OPENROUTER_API_KEY', '')
    if not key:
        return {'lookup_error': 'OPENROUTER_API_KEY not loaded in this shell'}
    url = 'https://openrouter.ai/api/v1/generation?' + urllib.parse.urlencode({'id': generation_id})
    request = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + key,
                                                   'Accept': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = json.load(response)
        value = body.get('data') or {}
        return {k: value.get(k) for k in ('provider_name', 'model', 'finish_reason',
                 'native_finish_reason', 'native_tokens_completion',
                 'native_tokens_reasoning', 'tokens_completion', 'total_cost')}
    except Exception as exc:
        return {'lookup_error': f'{type(exc).__name__}: {exc}'}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', default='.')
    p.add_argument('--session', required=True)
    p.add_argument('--lookup-provider', action='store_true',
                   help='Read-only OpenRouter generation metadata lookup; no inference')
    args = p.parse_args()
    root = Path(args.repo).resolve()
    db_path = root / 'results/kbs_journal/checkpoints.sqlite'
    if not db_path.exists():
        raise SystemExit(f'Missing database: {db_path}')
    conn = sqlite3.connect(f'file:{urllib.parse.quote(str(db_path))}?mode=ro', uri=True)
    try:
        failures = conn.execute(
            'SELECT doc_id, run_id, detail FROM failures WHERE session_id=? ORDER BY updated',
            (args.session,)
        ).fetchall()
    finally:
        conn.close()
    print(f'Session {args.session} | recorded failed documents: {len(failures)}')
    if not failures:
        return 0
    lookup_cache: dict[str, dict] = {}
    anomaly_count = 0
    for index, (doc_id, run_id, detail) in enumerate(failures, 1):
        print(f'\n[{index}/{len(failures)}] {doc_id} | run_id={run_id}')
        base = root / 'results/kbs_journal'
        runs = list(base.glob(f'*/main_gpt/baseline/maven_ere/{run_id}'))
        runs += list(base.glob(f'*/main_qwen/baseline/maven_ere/{run_id}'))
        if not runs:
            runs = [x.parent for x in base.rglob('manifest.json') if x.parent.name == run_id]
        if not runs:
            print('  Trace directory not found for this run; database error:')
            print(' ', detail.splitlines()[-1] if detail else '(none)')
            continue
        call_dir = runs[0] / 'calls' / hashlib.sha256(doc_id.encode('utf-8')).hexdigest()[:24]
        traces = sorted(call_dir.glob(args.session + '-*.json'))
        if not traces:
            # Identity hotfix may use internal UID; inspect only the current run's
            # calls folder and filter by the original identifier if present.
            traces = [f for f in (runs[0] / 'calls').glob('*/' + args.session + '-*.json')
                      if f.is_file() and _matches_doc(f, doc_id)]
        print('  saved API traces:', len(traces))
        for trace_path in traces:
            try:
                trace = json.loads(trace_path.read_text(encoding='utf-8'))
            except (OSError, ValueError) as exc:
                print('  unreadable trace:', trace_path.name, type(exc).__name__)
                continue
            req = trace.get('request') or {}
            rsp = trace.get('response') or {}
            prm = req.get('parameters') or {}
            usage = rsp.get('usage') or {}
            details = usage.get('completion_tokens_details') or usage.get('output_tokens_details') or {}
            choices = rsp.get('choices') or []
            choice = choices[0] if choices else {}
            if not isinstance(choice, dict):
                choice = {}
            message = choice.get('message') or {}
            content = message.get('content') or ''
            completion = usage.get('completion_tokens', usage.get('output_tokens'))
            cap = prm.get('max_completion_tokens', prm.get('max_tokens'))
            reasons = details.get('reasoning_tokens')
            print('  trace:', trace_path.name)
            print('   model:', req.get('model'), '| response model:', rsp.get('model'))
            print('   max_tokens:', prm.get('max_tokens'), 'max_completion_tokens:', prm.get('max_completion_tokens'),
                  'reasoning_effort:', prm.get('reasoning_effort'), 'reasoning:', prm.get('reasoning'))
            print('   finish:', choice.get('finish_reason'), '| output:', completion,
                  '| reasoning:', reasons, '| visible chars:', len(str(content).strip()),
                  '| configured cap:', cap)
            print('   response ID:', rsp.get('id', '(not saved)'),
                  '| gateway trace error:', trace.get('error'),
                  '| LiteLLM cost USD:', rsp.get('kbs_litellm_reported_cost_usd'))
            if isinstance(cap, (int, float)) and isinstance(completion, (int, float)) and completion > cap:
                anomaly_count += 1
                print('   *** ANOMALY: reported completion exceeds configured cap ***')
            gen_id = rsp.get('id')
            if args.lookup_provider and isinstance(gen_id, str) and gen_id:
                if gen_id not in lookup_cache:
                    lookup_cache[gen_id] = get_provider_metadata(gen_id)
                print('   OpenRouter generation metadata:', lookup_cache[gen_id])
    print(f'\nCalls with output above configured cap: {anomaly_count}')
    print('Read-only: no checkpoint edits, no LLM inference requests, no secret values printed.')
    return 0


def _matches_doc(path: Path, doc_id: str) -> bool:
    try:
        entry = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError):
        return False
    return entry.get('document_id') == doc_id


if __name__ == '__main__':
    raise SystemExit(main())
