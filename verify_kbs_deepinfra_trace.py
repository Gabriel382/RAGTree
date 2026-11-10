#!/usr/bin/env python3
"""Read the most recent KBS GPT-OSS trace pinned to DeepInfra; no API calls."""
import json
from pathlib import Path

base = Path('results/kbs_journal/openrouter/main_gpt/baseline/maven_ere')
traces = []
for path in base.glob('*/calls/*/*.json'):
    try:
        d = json.loads(path.read_text(encoding='utf-8'))
        p = (d.get('request') or {}).get('parameters') or {}
        pin = (p.get('extra_body') or {}).get('provider') or {}
        if pin.get('only') == ['deepinfra'] and pin.get('allow_fallbacks') is False and pin.get('require_parameters') is True:
            traces.append((path.stat().st_mtime, path, d))
    except (OSError, ValueError, TypeError, AttributeError):
        continue
if not traces:
    raise SystemExit('NO pinned DeepInfra trace found. Check the installed gateway and dry run before a full run.')
_, path, d = max(traces, key=lambda row: row[0])
res = d.get('response') or {}
choices = res.get('choices') or []
choice = choices[0] if choices else {}
message = choice.get('message') or {}
usage = res.get('usage') or {}
print('Trace:', path)
print('Document:', d.get('document_id'))
print('Request routing:', (d['request']['parameters']['extra_body']['provider']))
print('Response provider:', res.get('provider') or '(not carried in LiteLLM response; verify in OpenRouter activity)')
print('Finish:', choice.get('finish_reason'))
print('Content length:', len(message.get('content') or ''))
print('Output tokens:', usage.get('completion_tokens'))
print('Recorded error:', d.get('error'))
