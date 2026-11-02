#!/usr/bin/env python3
"""Safely integrate the KBS bounded fallback across earlier KBS patch revisions.

Usage: python install_compatible.py --repo . [--apply]
No change is made without --apply. Every changed file is backed up.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

CANONICAL = Path(__file__).with_name('gateway_retry_block.txt').read_text(encoding='utf-8')


def put_once(src, old, new, name):
    if src.count(old) != 1:
        raise RuntimeError(f'{name}: expected exactly one anchor, found {src.count(old)}')
    return src.replace(old, new, 1)


def edit_gateway(src):
    if 'KBS_EMPTY_LENGTH_FALLBACK' in src:
        if 'KBS_FALLBACK_MAX_TOKENS' not in src or 'generation_fallback' not in src:
            raise RuntimeError('gateway.py has a partial/different fallback implementation; refusing to overwrite')
        return src
    if 'class LiteLLMClient' not in src or 'def chat(' not in src or 'def embed(' not in src:
        raise RuntimeError('gateway.py is not a recognized KBS LiteLLM adapter')
    if 'from .runtime import trace_call, trace_event' not in src:
        raise RuntimeError('gateway.py does not have KBS trace helpers')
    if "if os.environ.get('KBS_REASONING_EFFORT'):" not in src:
        old = "        if os.environ.get('KBS_LITELLM_API_BASE'):"
        new = "        if os.environ.get('KBS_REASONING_EFFORT'):\n            data.setdefault('reasoning_effort',os.environ['KBS_REASONING_EFFORT'])\n" + old
        src = put_once(src, old, new, 'gateway reasoning')
    start_re = re.search(r'^        start\s*=\s*time\.monotonic\(\)\s*$', src, re.M)
    if not start_re:
        raise RuntimeError('gateway.py generation block has unknown structure; no changes made')
    end_re = re.search(r'^    def embed\(', src[start_re.end():], re.M)
    if not end_re:
        raise RuntimeError('gateway.py embed method missing; no changes made')
    end = start_re.end() + end_re.start()
    old_block = src[start_re.start():end]
    if 'litellm.completion(' not in old_block or 'trace_call(' not in old_block:
        raise RuntimeError('gateway.py call/trace implementation is unfamiliar; no changes made')
    return src[:start_re.start()] + CANONICAL + '\n' + src[end:]


def edit_protocol_py(src):
    if 'def output_config(' not in src or 'load_protocol()' not in src:
        raise RuntimeError('protocol.py lacks recognized KBS output configuration')
    if 'protocol_mode:str=' not in src and 'protocol_mode: str =' not in src:
        src = put_once(src,
          'def output_config(job:Job,run_root:Path,profile:dict,shard:int=0):',
          'def output_config(job:Job,run_root:Path,profile:dict,shard:int=0,protocol_mode:str="legacy_xquality"):',
          'protocol.py function signature')
    if 'generation = (load_protocol().get(' not in src:
        anchor="    model_id=profile['models'][job.model]\n"
        insert=("    generation = (load_protocol().get('generation', {}).get(job.model, {})\n"
                "                  if protocol_mode=='controlled_revision_v2' else {})\n")
        src=put_once(src, anchor, anchor+insert, 'protocol.py generation mapping')
        src=put_once(src,
           "'temperature':default.get('temperature',0.0),\n           'max_tokens':default.get('max_tokens',1024)}",
           "'temperature':generation.get('temperature',default.get('temperature',0.0)),\n"
           "           'max_tokens':generation.get('max_tokens',default.get('max_tokens',1024))}",
           'protocol.py litellm config')
    return src


def edit_engine(src):
    if not re.search(r'output_config\(job,root,profile,index,protocol_mode=protocol_mode', src):
        src = put_once(src, 'output_config(job,root,profile,index)',
                       'output_config(job,root,profile,index,protocol_mode=protocol_mode)',
                       'engine.py output_config call')
    if 'KBS_EMPTY_LENGTH_FALLBACK' in src:
        if 'KBS_FALLBACK_MAX_TOKENS' not in src or 'empty_length_fallback_max_tokens' not in src:
            raise RuntimeError('engine.py has an incomplete KBS fallback configuration')
        return src
    # Anchors occur inside the shard subprocess environment configuration.
    end_anchor = "            if job.phase=='budget_control':env['KBS_EVIDENCE_BUDGET_TOKENS']='2048'"
    if src.count(end_anchor) != 1:
        raise RuntimeError('engine.py budget_control anchor not found uniquely')
    comment = '            # Do not modify xquality generation behavior or use untracked'
    if comment in src:
        first = src.index(comment)
    else:
        # Old versions have no reasoning configuration block at all.
        first = src.index(end_anchor)
    last = src.index(end_anchor)
    if first > last:
        raise RuntimeError('engine.py unexpected environment ordering')
    previous = src[first:last]
    if previous and ('env.pop(' not in previous or 'KBS_REASONING_EFFORT' not in previous):
        raise RuntimeError('engine.py contains an unfamiliar environment block; refusing replacement')
    block=(
        "            # KBS controlled generation: never inherit untracked overrides.\n"
        "            for key in ('KBS_REASONING_EFFORT', 'KBS_EMPTY_LENGTH_FALLBACK',\n"
        "                        'KBS_FALLBACK_MAX_TOKENS'):\n"
        "                env.pop(key, None)\n"
        "            if protocol_mode=='controlled_revision_v2':\n"
        "                generation=load_protocol().get('generation',{}).get(job.model,{})\n"
        "                if generation.get('reasoning_effort'):\n"
        "                    env['KBS_REASONING_EFFORT']=str(generation['reasoning_effort'])\n"
        "                fallback_cap=generation.get('empty_length_fallback_max_tokens')\n"
        "                if fallback_cap is not None:\n"
        "                    if int(fallback_cap)<=int(generation.get('max_tokens',0)):\n"
        "                        raise ValueError('KBS fallback cap must exceed primary generation cap')\n"
        "                    env['KBS_EMPTY_LENGTH_FALLBACK']='1'\n"
        "                    env['KBS_FALLBACK_MAX_TOKENS']=str(int(fallback_cap))\n"
    )
    return src[:first] + block + src[last:]


def edit_yaml(src):
    if 'empty_length_fallback_max_tokens:' in src:
        if not re.search(r'^\s+empty_length_fallback_max_tokens:\s*8192\s*$', src, re.M):
            raise RuntimeError('protocol.yaml already defines a different fallback cap: inspect manually')
        return src
    if '\ngeneration:\n' not in src:
        if src.count('max_shard_seconds:') != 1:
            raise RuntimeError('protocol.yaml has no generation section and no known insertion point')
        addition=(
            '# KBS controlled revision generation; legacy xquality is unchanged.\n'
            'generation:\n'
            '  gpt_oss_20b:\n'
            '    temperature: 0.0\n'
            '    max_tokens: 4096\n'
            '    reasoning_effort: low\n'
            '    empty_length_fallback_max_tokens: 8192\n'
            '  qwen3_30b_a3b:\n'
            '    temperature: 0.0\n'
            '    max_tokens: 4096\n'
        )
        return put_once(src, 'max_shard_seconds:', addition+'max_shard_seconds:', 'protocol.yaml generation')
    match = re.search(r'(?m)^  gpt_oss_20b:\s*\n', src)
    if not match:
        raise RuntimeError('protocol.yaml generation section lacks gpt_oss_20b')
    # Never insert beyond the named model's settings.
    start=match.end()
    after=re.search(r'(?m)^  [A-Za-z0-9_]+:\s*$', src[start:])
    if not after:
        raise RuntimeError('protocol.yaml cannot identify end of GPT model settings')
    end=start+after.start()
    part=src[start:end]
    if not re.search(r'(?m)^    max_tokens:\s*4096\s*$',part):
        raise RuntimeError('GPT-OSS must be configured with max_tokens=4096 before adding fallback')
    return src[:end]+'    empty_length_fallback_max_tokens: 8192\n'+src[end:]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path('.'))
    p.add_argument('--apply', action='store_true', help='Write changes (default dry-run)')
    args=p.parse_args()
    repo=args.repo.resolve()
    spec=[
        ('ragtree/experiments/kbs/gateway.py',edit_gateway),
        ('ragtree/experiments/kbs/engine.py',edit_engine),
        ('ragtree/experiments/kbs/protocol.py',edit_protocol_py),
        ('configs/experiments/kbs_journal/protocol.yaml',edit_yaml),
    ]
    output={}
    for relative,editor in spec:
        path=repo/relative
        if not path.is_file():
            sys.exit(f'NOT APPLIED: missing {path}\nCheck that the KBS framework is installed in this checkout.')
        original=path.read_text(encoding='utf-8-sig')
        try:
            changed=editor(original)
            if relative.endswith('.py'):
                ast.parse(changed,filename=relative)
        except Exception as e:
            sys.exit(f'NOT APPLIED: incompatible {relative}: {e}\nShare this file to make a version-specific patch.')
        output[path]=(original,changed)
        state='unchanged' if original==changed else 'needs update'
        print(f'{state}: {relative}')
    if not args.apply:
        print('DRY RUN ONLY. Add --apply to update all four files and save backups.')
        return 0
    changed_files=[]
    try:
        for path,(original,changed) in output.items():
            if original==changed:continue
            backup=path.with_name(path.name+'.before_kbs_fallback.bak')
            if backup.exists():
                raise RuntimeError(f'Backup already exists: {backup}. Refusing to overwrite.')
            shutil.copy2(path,backup)
            temp=path.with_name(path.name+'.kbs_tmp')
            temp.write_text(changed,encoding='utf-8')
            os.replace(temp,path)
            changed_files.append((path,backup))
        print(f'APPLIED: {len(changed_files)} files updated. Backups: *.before_kbs_fallback.bak')
    except Exception as e:
        for path,backup in reversed(changed_files):
            shutil.copy2(backup,path)
        sys.exit(f'Rolled back due to error: {e}')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
