#!/usr/bin/env python3
"""Targeted KBS run of the three historical DocRED pilot failures.

Dry run by default. --execute sends real model requests. Reuses the KBS scheduler,
checkpointing, and original model-method scripts; no shortcuts through raw LLM API.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3

TARGETS = {
    'baseline': ['DocRED - 832aa7b8fd581e20'],
    'langgraph_agentic_simple': [
        'DocRED - e8f3abcd8da9ebb2',
        'DocRED - e92387cc98263d80',
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Explicitly perform real inference')
    parser.add_argument('--profile', default='local', choices=['local', 'openrouter'])
    parser.add_argument('--max-errors', type=int, default=3)
    args = parser.parse_args()
    from ragtree.experiments.kbs.engine import OUT, execute_job, filtered_docs, campaign_lock
    from ragtree.experiments.kbs.protocol import job_matrix
    from ragtree.experiments.kbs.record_identity import checkpoint_id
    from ragtree.experiments.kbs.runtime import init_db

    plan = []
    for method, ids in TARGETS.items():
        matches = [job for job in job_matrix()
                   if job.phase == 'main_gpt' and job.method == method and job.dataset == 'docred_causal']
        if len(matches) != 1:
            raise SystemExit(f'Expected precisely one KBS main_gpt job for {method}; found {len(matches)}')
        job = matches[0]
        id_set = set(ids)
        actual = [doc for doc in filtered_docs(job, 'controlled_revision_v2', OUT)
                  if doc.get('document_id') in id_set]
        actual_ids = [doc['document_id'] for doc in actual]
        if len(actual) != len(ids) or set(actual_ids) != id_set:
            raise SystemExit(f'Source mismatch for {method}: expected {ids!r}, found {actual_ids!r}')
        selected = [checkpoint_id(doc) for doc in actual]
        if len(set(selected)) != len(selected):
            raise SystemExit(f'Private checkpoint key collision for {method}')
        plan.append((job, selected, ids))
    print('Exact document targets:', sum(len(s) for _,s,_ in plan))
    for job, selected, original in plan:
        print(f'  {job.method}/{job.dataset}: {len(selected)} record(s) {original!r}')
    if not args.execute:
        print('DRY RUN: no inference performed. Add --execute to run exactly these 3 records.')
        return 0

    results=[]
    with campaign_lock(OUT):
        init_db(OUT/'checkpoints.sqlite')
        for job, selected, original in plan:
            outcome=execute_job(
                job, profile_name=args.profile, mode='sequential', workers=1,
                max_errors=args.max_errors, selected_ids=selected,
                retry_failed=True, protocol_mode='controlled_revision_v2',
                results_root=OUT,
            )
            row={'method':job.method,'dataset':job.dataset, 'status':outcome['status'],
                 'run_id':outcome.get('run_id'), 'run_root':outcome.get('run_root')}
            if outcome.get('run_id'):
                with sqlite3.connect(OUT/'checkpoints.sqlite') as con:
                    placeholders=','.join('?' for _ in selected)
                    states=con.execute(
                        f'SELECT doc_id,status FROM records WHERE run_id=? AND doc_id IN ({placeholders})',
                        [outcome['run_id'], *selected],
                    ).fetchall()
                row['selected_status']={key:status for key,status in states}
                row['done_selected']=sum(status=='done' for _,status in states)
                row['expected_selected']=len(selected)
            results.append(row)
            print(json.dumps(row,ensure_ascii=False,indent=2))
            if outcome['status'] in ('failed','aborted','blocked'):
                print('Stopped after unsuccessful job; prior outputs remain intact.')
                break
    success=len(results)==len(plan) and all(
        x.get('done_selected')==x.get('expected_selected') for x in results)
    print('ALL THREE DOCUMENTS COMPLETED:', success)
    return 0 if success else 1

if __name__ == '__main__':
    raise SystemExit(main())
