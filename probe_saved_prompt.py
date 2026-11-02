#!/usr/bin/env python3
"""Single non-checkpointed vLLM diagnostic against the actual failed journal prompt.

NEVER writes experiment results or changes protocols. Uses only localhost by default.
"""
from pathlib import Path
import argparse,json,urllib.request

p=argparse.ArgumentParser()
p.add_argument('--repo',type=Path,default=Path('.'))
p.add_argument('--api-base',default='http://127.0.0.1:8000/v1')
p.add_argument('--max-tokens',type=int,default=4096)
p.add_argument('--reasoning-effort',choices=['low','medium','high'],default='low')
p.add_argument('--timeout',type=int,default=240)
a=p.parse_args()
if not 1<=a.max_tokens<=16384:raise SystemExit('Refusing max_tokens outside [1,16384]')
base=a.repo/'results/kbs_journal/local/main_gpt/baseline/maven_ere'
traces=sorted((f for f in base.glob('*/calls/*/*.json') if f.is_file()),key=lambda x:x.stat().st_mtime,reverse=True)
request=None
for path in traces:
    obj=json.loads(path.read_text(encoding='utf8'))
    raw=obj.get('request') or {}
    if raw.get('messages'):
        request=raw
        break
if not request:raise SystemExit('No saved source prompt found')
if not a.api_base.startswith(('http://127.0.0.1:','http://localhost:')):
    raise SystemExit('This diagnostic is intentionally localhost-only')
api=a.api_base.rstrip('/')
def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'Authorization':'Bearer dummy'}),timeout=a.timeout) as r:
        return json.load(r)
models=get_json(api+'/models')
model=models['data'][0]['id']
data={'model':model,'messages':request['messages'], 'temperature':0,
      'max_tokens':a.max_tokens,'reasoning_effort':a.reasoning_effort}
b=json.dumps(data).encode('utf8')
http=urllib.request.Request(api+'/chat/completions',data=b,method='POST',headers={
    'Content-Type':'application/json','Authorization':'Bearer dummy'})
print('Probe is ONE real local LLM request, not a benchmark checkpoint')
print('Model:',model,'max_tokens:',a.max_tokens,'reasoning_effort:',a.reasoning_effort)
with urllib.request.urlopen(http,timeout=a.timeout) as resp:
    obj=json.load(resp)
choice=(obj.get('choices') or [{}])[0]
message=choice.get('message') or {}
content=message.get('content')
print('Finish reason:',choice.get('finish_reason'))
print('Content length:',len(content) if isinstance(content,str) else 0)
print('Reasoning field present:',bool(message.get('reasoning_content') or message.get('reasoning')))
print('Usage:',obj.get('usage'))
if isinstance(content,str):print('Answer preview:',repr(content[:200]))
