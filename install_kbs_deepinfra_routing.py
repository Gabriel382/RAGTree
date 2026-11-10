#!/usr/bin/env python3
"""Install *versioned* OpenRouter DeepInfra routing for the KBS GPT-OSS protocol.

Dry run by default. --apply writes files after checking all anchors and syntax.
Originals are backed up. This changes the KBS fingerprint, hence generates a
NEW run ID. The old checkpoints are preserved but deliberately not rekeyed.

Usage:
    python install_kbs_deepinfra_routing.py --repo .
    python install_kbs_deepinfra_routing.py --repo . --apply
"""
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import re
import shutil
import sys

GATEWAY_MARK = 'KBS_OPENROUTER_PROVIDER'
ENGINE_MARK = 'KBS_OPENROUTER_PROVIDER'
YAML_MARK = 'openrouter_provider_routing:'

GATEWAY_INSERT = '''        # Controlled KBS provider routing. The provider is configured in the
        # versioned protocol, propagated by execute_job, and saved in traces.
        route_provider = os.environ.get('KBS_OPENROUTER_PROVIDER', '').strip()
        if route_provider:
            if os.environ.get('KBS_PROTOCOL_MODE') != 'controlled_revision_v2':
                raise RuntimeError('KBS provider pin is only valid for controlled_revision_v2')
            if not str(model).startswith('openrouter/openai/gpt-oss-20b'):
                raise RuntimeError('KBS provider pin must not affect another model')
            extra_body = dict(data.get('extra_body') or {})
            if 'provider' in extra_body:
                raise RuntimeError('Conflicting OpenRouter provider configuration')
            extra_body['provider'] = {
                'only': [route_provider],
                'allow_fallbacks': False,
                'require_parameters': True,
            }
            data['extra_body'] = extra_body
'''

ENGINE_INSERT = '''            # Always clear any inherited provider override. This setting is
            # read from protocol.yaml so the checkpoint fingerprint tracks it.
            env.pop('KBS_OPENROUTER_PROVIDER', None)
            if (protocol_mode == 'controlled_revision_v2' and
                    profile_name == 'openrouter' and job.model == 'gpt_oss_20b'):
                routing = load_protocol().get('openrouter_provider_routing', {}).get('gpt_oss_20b', {})
                allowed = routing.get('only')
                if allowed is not None:
                    if allowed != ['deepinfra']:
                        raise ValueError('Unexpected OpenRouter GPT-OSS provider policy')
                    if routing.get('allow_fallbacks') is not False or routing.get('require_parameters') is not True:
                        raise ValueError('OpenRouter provider restrictions must be explicit')
                    env['KBS_OPENROUTER_PROVIDER'] = 'deepinfra'
'''

YAML_INSERT = '''# Pinned provider for the controlled GPT-OSS OpenRouter baseline. Prevents
# SiliconFlow/Darkbloom from serving a request after their observed failures.
# Qwen remains on its previous routing policy until independently validated.
openrouter_provider_routing:
  gpt_oss_20b:
    only: [deepinfra]
    allow_fallbacks: false
    require_parameters: true
'''


def _once(src: str, marker: str, add: str, filename: str) -> str:
    if src.count(marker) != 1:
        raise ValueError(f'{filename}: insertion anchor occurs {src.count(marker)} times, need exactly one')
    return src.replace(marker, add + marker, 1)


def edit_gateway(src: str) -> str:
    if GATEWAY_MARK in src:
        if 'allow_fallbacks' in src and 'route_provider' in src:
            return src
        raise ValueError('gateway already has a different KBS provider mechanism')
    if 'class LiteLLMClient' not in src or 'litellm.completion(' not in src:
        raise ValueError('Unrecognized LiteLLM gateway')
    # Insert before the bounded fallback's first attempt, not inside it.
    anchor = '        # An empty final answer after length termination'
    return _once(src, anchor, GATEWAY_INSERT + '\n', 'gateway.py')


def edit_engine(src: str) -> str:
    if ENGINE_MARK in src:
        if 'openrouter_provider_routing' in src and "env.pop('KBS_OPENROUTER_PROVIDER', None)" in src:
            return src
        raise ValueError('engine already has a different routing mechanism')
    if 'def execute_job(' not in src:
        raise ValueError('Unrecognized KBS engine')
    anchor = "            if job.phase=='budget_control':env['KBS_EVIDENCE_BUDGET_TOKENS']='2048'"
    return _once(src, anchor, ENGINE_INSERT, 'engine.py')


def edit_yaml(src: str) -> str:
    if YAML_MARK in src:
        if 'only: [deepinfra]' in src and 'require_parameters: true' in src:
            return src
        raise ValueError('protocol.yaml already defines a different provider routing')
    if 'revised_protocol: controlled_revision_v2' not in src:
        raise ValueError('Unexpected protocol revision')
    anchor = 'max_errors:'
    return _once(src, anchor, YAML_INSERT, 'protocol.yaml')


def install(repo: Path, apply: bool = False):
    spec = [
        ('ragtree/experiments/kbs/gateway.py', edit_gateway),
        ('ragtree/experiments/kbs/engine.py', edit_engine),
        ('configs/experiments/kbs_journal/protocol.yaml', edit_yaml),
    ]
    staged = []
    for rel, editor in spec:
        dest = repo / rel
        if not dest.is_file():
            raise RuntimeError(f'Missing {rel}')
        before = dest.read_text(encoding='utf-8-sig')
        after = editor(before)
        if dest.suffix == '.py':
            ast.parse(after, filename=str(dest))
        if dest.suffix == '.yaml':
            try:
                import yaml
                parsed = yaml.safe_load(after)
                assert parsed['openrouter_provider_routing']['gpt_oss_20b']['only'] == ['deepinfra']
            except ImportError:
                print('WARNING: PyYAML not installed; YAML contents not parsed')
        print(f'{"unchanged" if before == after else "needs update"}: {rel}')
        staged.append((dest, before, after))
    if not apply:
        print('DRY RUN ONLY. Add --apply to install. The run ID will change.')
        return
    planned = [(p,old,new) for p,old,new in staged if old != new]
    for p, *_ in planned:
        back = p.with_name(p.name + '.before_provider_pin.bak')
        if back.exists():
            raise RuntimeError(f'Backup exists: {back}; refusing to overwrite')
    completed = []
    try:
        for p, old, new in planned:
            backup = p.with_name(p.name + '.before_provider_pin.bak')
            shutil.copy2(p, backup)
            tmp = p.with_name(p.name + '.provider_pin_tmp')
            tmp.write_text(new, encoding='utf-8')
            os.replace(tmp, p)
            completed.append((p, backup))
    except Exception:
        for p, backup in reversed(completed):
            shutil.copy2(backup, p)
        raise
    print(f'APPLIED: {len(planned)} files. Backups: *.before_provider_pin.bak')
    print('NOTE: versioned code/protocol changed => NEW checkpoint run ID; previous checkpoints preserved.')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, default=Path('.'))
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args(argv)
    try:
        install(args.repo.resolve(), args.apply)
    except (ValueError, RuntimeError, SyntaxError) as e:
        print(f'NOT APPLIED: {e}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
