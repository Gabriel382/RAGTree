#!/usr/bin/env python3
"""Lossless FinCausal *document identity* repair for KBS checkpointing.

Do not deduplicate source rows: source text may repeat with distinct annotations.
Creates an immutable-meaning derived JSONL with unique document IDs and optional
config override with a backup of default.yaml. No relation/entity/text changes.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

p=argparse.ArgumentParser()
p.add_argument('--repo',type=Path,default=Path('.'))
p.add_argument('--apply',action='store_true',help='Write derived JSONL and mapping CSV')
p.add_argument('--update-config',action='store_true',help='Also update default.yaml, with backup (requires --apply)')
a=p.parse_args()
root=a.repo.resolve()
source=root/'data/preprocessed/fincausal.jsonl'
target=root/'data/preprocessed/fincausal_kbs_unique.jsonl'
mapping=root/'data/preprocessed/fincausal_kbs_unique_mapping.csv'
if a.update_config and not a.apply:raise SystemExit('--update-config requires --apply')
if not source.exists():raise SystemExit(f'Missing {source}')
rows=[]
for line_no,line in enumerate(source.read_text(encoding='utf8').splitlines(),1):
    if line.strip():
        doc=json.loads(line)
        ident=str(doc.get('document_id') or doc.get('id') or '')
        if not ident:raise SystemExit(f'Missing ID at source line {line_no}')
        rows.append((line_no,doc,ident))
counts=Counter(ident for _,_,ident in rows)
used=set();out=[];entries=[]
for n,doc,ident in rows:
    if counts[ident]>1:
        # Sequence number is stable because the exact preprocessed source bytes
        # are frozen by the experiment manifest/checksum.
        new_id=f'{ident} :: kbs-source-row-{n:06d}'
        doc=dict(doc)
        doc['kbs_original_document_id']=ident
        doc['document_id']=new_id
    else:new_id=ident
    if new_id in used:raise SystemExit(f'New identifier unexpectedly collided at line {n}')
    used.add(new_id)
    out.append(doc)
    entries.append((n,ident,new_id,str(doc.get('title',''))))
assert len(out)==len(rows)==len(used)
for (_,original,_),new in zip(rows,out):
    before=dict(original);after=dict(new)
    before.pop('document_id',None);after.pop('document_id',None);after.pop('kbs_original_document_id',None)
    assert before==after,'A scientific field unexpectedly changed!'
print(f'Records: {len(rows)}, unique original IDs: {len(counts)}, duplicates: {len(rows)-len(counts)}')
print(f'Collision groups: {sum(v>1 for v in counts.values())}; output unique IDs: {len(used)}')
print('Source SHA256:',hashlib.sha256(source.read_bytes()).hexdigest())
if not a.apply:
    print('DRY RUN: nothing written. To save derived file, pass --apply --update-config')
    raise SystemExit(0)
if target.exists():
    current=[json.loads(line) for line in target.read_text(encoding='utf8').splitlines() if line.strip()]
    if current!=out:raise SystemExit(f'{target} already exists but does not match generated records; refusing overwrite')
else:
    with target.open('x',encoding='utf8') as f:
        for doc in out:f.write(json.dumps(doc,ensure_ascii=False)+'\n')
if mapping.exists():
    print(f'Mapping already exists: {mapping}')
else:
    with mapping.open('x',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['source_line','original_document_id','kbs_document_id','title']);w.writerows(entries)
if a.update_config:
    config=root/'configs/default.yaml'
    config_text=config.read_text(encoding='utf8')
    old='data/preprocessed/fincausal.jsonl';new='data/preprocessed/fincausal_kbs_unique.jsonl'
    pat=r'(?m)^(\s*fincausal:\s*)'+re.escape(old)+r'(?=\s*(?:#.*)?$)'
    amended,num=re.subn(pat,lambda m:m.group(1)+new,config_text)
    if num==1:
        backup=config.with_suffix('.yaml.bak_kbs')
        if not backup.exists():backup.write_text(config_text,encoding='utf8')
        config.write_text(amended,encoding='utf8')
        print('Updated config:',config,'; backup:',backup)
    elif new in config_text:
        print('Config already points to derived FinCausal JSONL')
    else:raise SystemExit('Cannot identify exactly one FinCausal dataset path in configs/default.yaml; update manually')
print('Derived dataset:',target)
print('Source mapping:',mapping)
