#!/usr/bin/env python3
"""Read-only diagnostic of the most recent KBS LiteLLM response; never prints prompt text."""
from pathlib import Path
import argparse, json

p=argparse.ArgumentParser()
p.add_argument('--repo',type=Path,default=Path('.'))
p.add_argument('--dataset',default='maven_ere')
p.add_argument('--method',default='baseline')
a=p.parse_args()
base=a.repo/'results/kbs_journal/local/main_gpt'/a.method/a.dataset
traces=sorted((f for f in base.glob('*/calls/*/*.json') if f.is_file()),key=lambda x:x.stat().st_mtime,reverse=True)
if not traces:
    raise SystemExit(f'No saved LLM call traces under {base} (check whether KBS gateway was invoked)')
for f in traces[:10]:
    try:r=json.loads(f.read_text(encoding='utf8'))
    except (OSError,ValueError):continue
    result=r.get('response')
    if result is None:continue
    calls=result.get('choices') or []
    ch=calls[0] if calls else {}
    msg=ch.get('message') or {}
    u=result.get('usage') or {}
    d=u.get('completion_tokens_details') or {}
    content=msg.get('content')
    print('Trace:',f)
    print('Model:',result.get('model'))
    print('Request parameters:',r.get('request',{}).get('parameters'))
    print('Finish reason:',ch.get('finish_reason'))
    print('Content type:',type(content).__name__,'length:',len(str(content)) if content is not None else 0)
    print('Reasoning field present:',bool(msg.get('reasoning_content') or msg.get('reasoning')))
    print('Usage:',u)
    print('Error:',r.get('error'))
    break
else:
    raise SystemExit('Found traces, but no saved provider response; inspect error traces and vLLM logs')
