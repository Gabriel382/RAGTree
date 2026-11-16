#!/usr/bin/env python3
"""One-pass, no-inference quality audit of completed KBS campaign configurations.

Runs the existing KBS *report* command once per main phase, saves separately
named snapshots, cross-checks the exact completed run roots recorded in a
campaign report, and prints scores plus coverage/anomaly flags. No model calls.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import shutil
import subprocess
import sys

DEFAULT_SESSION = 'aed8f2dcd4d1470ba38ec1be968ebd93'


def jread(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def fnum(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pct(value):
    return f'{100 * value:6.2f}' if value is not None else '   N/A'


def csvout(path, data, names):
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=names, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, default=Path('.'))
    ap.add_argument('--session', default=DEFAULT_SESSION)
    ap.add_argument('--no-refresh', action='store_true',
                    help='Use existing saved per-run metrics; skip regenerating reports.')
    args = ap.parse_args()
    repo = args.repo.resolve()
    rep = repo / 'results/kbs_journal/reports'
    campfile = rep / f'campaign_{args.session}.json'
    if not campfile.is_file():
        ap.error(f'Campaign report missing: {campfile}')
    campaign = jread(campfile)
    if campaign.get('session_id') != args.session:
        ap.error('Campaign report has a different session_id.')
    finished = [r for r in campaign.get('results', []) if r.get('status') == 'completed']
    if not finished:
        ap.error('Campaign contains no completed configurations.')
    phases = sorted({r['phase'] for r in finished})
    if any(p not in ('main_gpt', 'main_qwen') for p in phases):
        ap.error(f'Unexpected phase(s): {phases}')

    dest = rep / f'quality_check_{args.session}'
    dest.mkdir(parents=True, exist_ok=True)
    snapshots = {}
    if not args.no_refresh:
        for phase in phases:
            command = [sys.executable, 'scripts/kbs_protocol.py', 'report',
                       '--profile', 'openrouter', '--phase', phase]
            print(f'\nRecalculate local metrics for {phase} (NO inference):', flush=True)
            p = subprocess.run(command, cwd=repo, check=False)
            if p.returncode:
                print(f'ERROR: report failed for {phase}: exit={p.returncode}', file=sys.stderr)
                return 2
            current_report = rep / 'metrics.json'
            if not current_report.is_file():
                print(f'ERROR: expected {current_report} not found', file=sys.stderr)
                return 2
            save_path = dest / f'metrics_{phase}_snapshot.json'
            shutil.copyfile(current_report, save_path)
            snapshots[phase] = jread(save_path)
    else:
        for phase in phases:
            saved = dest / f'metrics_{phase}_snapshot.json'
            snapshots[phase] = jread(saved) if saved.exists() else []

    records = []
    relation_rows = []
    findings = []
    for cell in sorted(finished, key=lambda x: (x['method'], x['dataset'], x['phase'])):
        phase, method, dataset = (cell[k] for k in ('phase', 'method', 'dataset'))
        expected = int(cell.get('selected_record_status', {}).get('expected', -1))
        rid = cell.get('run_id', '')
        rr = Path(cell.get('run_root') or '')
        if not rr.is_absolute():
            rr = repo / rr
        rootmetric = rr / 'metrics.json'
        candidates = []
        if rootmetric.is_file():
            candidates.append(('exact-run-root', jread(rootmetric)))
        for r in snapshots.get(phase, []):
            if r.get('phase') == phase and r.get('method') == method and r.get('dataset') == dataset:
                if r.get('stale_fingerprint'):
                    continue
                if r.get('run_id') and r['run_id'] != rid:
                    continue
                candidates.append(('report-snapshot', r))
        # Prioritize the exact run-root, but only when all row metadata matches.
        match = None
        source = None
        for label, r in candidates:
            if (r.get('phase'), r.get('method'), r.get('dataset')) != (phase, method, dataset):
                continue
            if r.get('run_id') and r['run_id'] != rid:
                continue
            if r.get('protocol_mode') and r['protocol_mode'] != 'controlled_revision_v2':
                continue
            if fnum(r.get('record_count')) != expected:
                continue
            if r.get('status') != 'complete':
                continue
            match, source = r, label
            break
        item = {'phase':phase, 'method':method, 'dataset':dataset, 'run_id':rid,
                'expected_docs':expected,'reported_docs':None,'coverage':None,
                'status':'NO_VALID_METRICS','P':None,'R':None,'F1':None,
                'macro_F1':None,'gold':None,'pred':None,'TP':None,'FP':None,'FN':None,
                'pred_gold_ratio':None,'llm_calls':None,'provider_cost_usd':None,
                'metric_source':source,'flags':''}
        flags=[]
        ck = cell.get('selected_record_status', {})
        if ck.get('done') != expected or ck.get('failed',0) or ck.get('missing',0):
            flags.append('CHECKPOINT_MISMATCH')
        if match is None:
            flags.append('METRICS_MISSING_OR_STALE')
        else:
            m = match.get('metrics') or {}
            mi = m.get('micro') or {}
            cov = m.get('coverage') or {}
            p,r,f = (fnum(mi.get(k)) for k in ('precision','recall','f1'))
            item.update({'reported_docs':match.get('record_count'),
                         'coverage':cov.get('successful'), 'status':match.get('status'),
                         'P':p,'R':r,'F1':f,'macro_F1':fnum(m.get('macro_f1')),
                         'llm_calls':match.get('llm_calls'),
                         'provider_cost_usd':match.get('provider_cost_usd'),
                         'metric_source':source})
            if (cov.get('expected') != expected or cov.get('successful') != expected
                    or cov.get('missing',0) != 0):
                flags.append('COVERAGE_MISMATCH')
            if match.get('score_error'):
                flags.append('SCORE_ERROR')
            if any(v is None or v < 0 or v > 1 for v in (p,r,f)):
                flags.append('INVALID_PRF')
            if f == 0:
                flags.append('ZERO_F1')
            labels = m.get('per_relation') or {}
            if not labels:
                flags.append('NO_PER_RELATION_METRICS')
            else:
                try:
                    tp=sum(int(v.get('tp',0)) for v in labels.values())
                    fp=sum(int(v.get('fp',0)) for v in labels.values())
                    fn=sum(int(v.get('fn',0)) for v in labels.values())
                    item.update({'TP':tp,'FP':fp,'FN':fn,'gold':tp+fn,'pred':tp+fp,
                                 'pred_gold_ratio':(tp+fp)/(tp+fn) if (tp+fn) else None})
                    pp=tp/(tp+fp) if tp+fp else 0.
                    rr2=tp/(tp+fn) if tp+fn else 0.
                    ff=2*pp*rr2/(pp+rr2) if pp+rr2 else 0.
                    if any(v is not None and abs(x-v)>1e-9 for v,x in ((p,pp),(r,rr2),(f,ff))):
                        flags.append('COUNTS_SCORE_MISMATCH')
                    if item['pred_gold_ratio'] is not None and item['pred_gold_ratio']<0.10:
                        flags.append('SEVERE_UNDERPREDICTION')
                except (ValueError, TypeError):
                    flags.append('INVALID_PER_LABEL_COUNTS')
                for label, d in sorted(labels.items()):
                    relation_rows.append({'phase':phase,'method':method,'dataset':dataset,
                        'relation':label, **{key:d.get(key) for key in
                         ('tp','fp','fn','precision','recall','f1','support_gold','support_pred')}})
        item['flags'] = ';'.join(flags)
        findings.extend(f'{phase}/{method}/{dataset}: {flag}' for flag in flags)
        records.append(item)

    columns=['phase','method','dataset','run_id','expected_docs','reported_docs','coverage',
             'status','P','R','F1','macro_F1','gold','pred','TP','FP','FN','pred_gold_ratio',
             'llm_calls','provider_cost_usd','metric_source','flags']
    csvout(dest/'completed_metrics.csv', records, columns)
    csvout(dest/'per_relation.csv', relation_rows,
           ['phase','method','dataset','relation','tp','fp','fn','precision','recall','f1','support_gold','support_pred'])

    pairs=defaultdict(dict)
    for r in records:
        pairs[(r['method'],r['dataset'])][r['phase']]=r
    comparative=[]
    for (method,dataset), group in sorted(pairs.items()):
        a,b=group.get('main_gpt'),group.get('main_qwen')
        comparative.append({'method':method,'dataset':dataset,
           'GPT_F1':a['F1'] if a else None,
           'Qwen_F1':b['F1'] if b else None,
           'Qwen_minus_GPT_pp':(100*(b['F1']-a['F1']) if a and b and
                             a['F1'] is not None and b['F1'] is not None else None)})
    csvout(dest/'backbone_comparison.csv',comparative,
           ['method','dataset','GPT_F1','Qwen_F1','Qwen_minus_GPT_pp'])

    output=[]
    output.append(f'# KBS completed-configuration sanity check — {args.session}')
    output.append('')
    output.append(f'Completed cells in selected campaign report: **{len(finished)}**')
    output.append(f'Cells with validated complete local metrics: **{sum(r["status"]=="complete" and not any(x in r["flags"] for x in ("MISMATCH","INVALID","ERROR","STALE")) for r in records)}**')
    output.append('')
    output.append('| Phase | Method | Dataset | Docs | P% | R% | F1% | Gold | Pred | Flags |')
    output.append('|---|---|---|---:|---:|---:|---:|---:|---:|---|')
    print('\nCOMPLETED CONFIGURATION SCORES (current run IDs only)')
    print(f'  {"PHASE":10} {"METHOD":16} {"DATASET":19} {"DOCS":>5} {"P%":>7} {"R%":>7} {"F1%":>7} {"GOLD":>7} {"PRED":>7}  FLAGS')
    for r in records:
        print(f'  {r["phase"]:10} {r["method"]:16} {r["dataset"]:19} '
              f'{r["expected_docs"]:5} {pct(r["P"])} {pct(r["R"])} {pct(r["F1"])} '
              f'{str(r["gold"] or "-"):>7} {str(r["pred"] or "-"):>7}  {r["flags"] or "OK"}')
        output.append(f'| {r["phase"]} | {r["method"]} | {r["dataset"]} | {r["expected_docs"]} | '
                      f'{pct(r["P"]).strip()} | {pct(r["R"]).strip()} | {pct(r["F1"]).strip()} | '
                      f'{r["gold"]} | {r["pred"]} | {r["flags"] or "OK"} |')
    output+=['','## Qwen versus GPT-OSS (F1 percentage-point difference)','',
             '| Method | Dataset | GPT F1% | Qwen F1% | Qwen - GPT (pp) |',
             '|---|---|---:|---:|---:|']
    print('\nCROSS-BACKBONE COMPARISON (same method and dataset)')
    for x in comparative:
        d=x['Qwen_minus_GPT_pp']
        print(f'  {x["method"]:16} {x["dataset"]:19} GPT={pct(x["GPT_F1"])}%  '
              f'Qwen={pct(x["Qwen_F1"])}%  Delta={d:+.2f}pp' if d is not None else
              f'  {x["method"]}/{x["dataset"]}: incomplete pair')
        output.append(f'| {x["method"]} | {x["dataset"]} | {pct(x["GPT_F1"]).strip()} | '
                      f'{pct(x["Qwen_F1"]).strip()} | {f"{d:+.2f}" if d is not None else "N/A"} |')
    output+=['','## Quality flags','']
    if findings:
        output.extend('- '+s for s in findings)
    else:
        output.append('No structural/coverage or severe underprediction flags. This does not prove semantic correctness.')
    output+=['','**Interpretation:** Complete checkpoints and matching scoring arithmetic establish technical consistency only. ',
            'Low or zero recall, label imbalance and method/backbone differences need scientific interpretation. ',
            'This command performs **no LLM inference** and does not modify the experiment checkpoints or protocol.']
    (dest/'QUALITY_CHECK.md').write_text('\n'.join(output)+'\n',encoding='utf-8')
    print(f'\nFiles: {dest}/QUALITY_CHECK.md, completed_metrics.csv, backbone_comparison.csv, per_relation.csv')
    print('No inference / no checkpoint mutations. The report CLI recalculates local metrics.')
    return 0 if not any('MISMATCH' in z or 'INVALID' in z or 'ERROR' in z or 'STALE' in z for z in findings) else 1


if __name__ == '__main__':
    raise SystemExit(main())
