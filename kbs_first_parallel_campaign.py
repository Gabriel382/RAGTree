#!/usr/bin/env python3
"""Bounded, checkpoint-safe *single pass* of KBS main benchmarks on OpenRouter.

Runs the two main phases (60 GPT-OSS + 60 Qwen) concurrently at two levels:
  - a limited number of independent method/dataset jobs;
  - a limited number of document shards per job.

No model request unless --execute is specified. A pilot with --documents-per-cell N
and the full pass use the SAME run IDs; completed documents are reused.

Place this file in the RAGTree repository root and run inside the project venv.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import uuid

from ragtree.experiments.kbs.engine import (
    OUT, campaign_lock, execute_job, filtered_docs, preflight, summarize_campaign,
)
from ragtree.experiments.kbs.estimate import sample_ids
from ragtree.experiments.kbs.protocol import job_matrix, load_protocol
from ragtree.experiments.kbs.runtime import init_db

PHASES = ('main_gpt', 'main_qwen')


def session_error_count(db: Path, session: str) -> int:
    with sqlite3.connect(db, timeout=30) as conn:
        return conn.execute('SELECT COUNT(*) FROM failures WHERE session_id=?', (session,)).fetchone()[0]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='One controlled KBS main-benchmark pass, with bounded concurrency across jobs and documents.'
    )
    parser.add_argument('--execute', action='store_true', help='Explicitly allow billable OpenRouter requests')
    parser.add_argument('--documents-per-cell', type=int, default=None,
                        help='Sample N documents per method/dataset for a cheap pilot; omit for the full dataset')
    parser.add_argument('--parallel-jobs', type=int, default=3, help='Max concurrent experiment cells (default 3)')
    parser.add_argument('--workers-per-job', type=int, default=3,
                        help='Max concurrent document shards per cell (default 3)')
    parser.add_argument('--max-errors', type=int, default=3,
                        help='Global terminal document failure threshold (default 3)')
    parser.add_argument('--max-shard-seconds', type=int, default=86400)
    parser.add_argument('--only-phase', choices=PHASES, default=None)
    parser.add_argument('--max-jobs', type=int, default=None,
                        help='Optional troubleshooting cap on number of jobs')
    parser.add_argument('--retry-failed', action='store_true',
                        help='Retry previously failed records; completed records remain skipped')
    parser.add_argument('--results-root', type=Path, default=OUT)
    parser.add_argument('--allow-unverified', action='store_true',
                        help='Explicitly allow unverified historical recipe mappings')
    args = parser.parse_args(argv)

    if not 1 <= args.parallel_jobs <= 8:
        parser.error('--parallel-jobs must be 1..8')
    if not 1 <= args.workers_per_job <= 16:
        parser.error('--workers-per-job must be 1..16')
    if args.parallel_jobs * args.workers_per_job > 32:
        parser.error('Total potential document processes must be <=32')
    if args.documents_per_cell is not None and args.documents_per_cell < 1:
        parser.error('--documents-per-cell must be at least 1')
    if args.max_errors < 0:
        parser.error('--max-errors must be nonnegative')
    if args.max_jobs is not None and args.max_jobs < 1:
        parser.error('--max-jobs must be at least 1')
    if args.max_shard_seconds < 1:
        parser.error('--max-shard-seconds must be positive')

    phases = (args.only_phase,) if args.only_phase else PHASES
    original_jobs = job_matrix()
    if args.only_phase:
        jobs = [j for j in original_jobs if j.phase == args.only_phase]
    else:
        # Interleave matching cells across the two backbones so BOTH models are
        # tested immediately, rather than doing all 60 GPT jobs before Qwen.
        gpt = [j for j in original_jobs if j.phase == 'main_gpt']
        qwen = [j for j in original_jobs if j.phase == 'main_qwen']
        if len(gpt) != 60 or len(qwen) != 60:
            raise RuntimeError('Unexpected KBS primary matrix size; stop before billing')
        if [(j.method, j.dataset) for j in gpt] != [(j.method, j.dataset) for j in qwen]:
            raise RuntimeError('Backbone job order differs; stop before billing')
        jobs = [job for pair in zip(gpt, qwen) for job in pair]
    if args.max_jobs is not None:
        jobs = jobs[:args.max_jobs]

    print('KBS PRIMARY CAMPAIGN')
    print('Phases:', dict(Counter(j.phase for j in jobs)))
    print('Experiment cells:', len(jobs), '/ 120 (full main matrix)')
    print('Model profile: OpenRouter; protocol: controlled_revision_v2')
    print('Documents per cell:', args.documents_per_cell if args.documents_per_cell else 'ALL')
    print('Parallel cells:', args.parallel_jobs)
    print('Shards per cell:', args.workers_per_job)
    print('Maximum concurrent subprocesses:', args.parallel_jobs * args.workers_per_job)
    print('Max terminal errors:', args.max_errors)
    print('No repeats, prompt variants, or sensitivity phases are selected.')
    print('Checkpoint identity is inherited from the existing RAGTree engine.')

    if not args.execute:
        print('\nDRY RUN. No model calls. Add --execute for billable inference.')
        return 0

    if not os.environ.get('OPENROUTER_API_KEY'):
        parser.error('OPENROUTER_API_KEY must be defined in the shell. Do not put the key in the script.')
    # Security: never echo, persist, or log the API key.
    if args.allow_unverified:
        print('WARNING: unverified historical mappings explicitly enabled.')

    root = args.results_root.resolve()
    db = root / 'checkpoints.sqlite'
    session = uuid.uuid4().hex
    started = time.monotonic()
    print('Session:', session, flush=True)

    def run_one(job):
        try:
            issue = preflight(job)
            reasons = list(issue['issues'])
            # Controlled revision recipe recovery was installed previously. Preserve
            # safety gating of the selected current implementation.
            if job.provenance != 'reconstructed' and not args.allow_unverified:
                from ragtree.experiments.kbs.protocol import load_protocol
                mode = load_protocol().get('revised_protocol')
                if mode != 'controlled_revision_v2':
                    reasons.append('unverified historical recipe')
            if reasons:
                return {'job': job.identity, 'phase': job.phase, 'method': job.method,
                        'dataset': job.dataset, 'status': 'blocked', 'reasons': reasons}

            selection = None
            if args.documents_per_cell is not None:
                selection = sample_ids(job, args.documents_per_cell)
            else:
                population = sum(1 for _ in filtered_docs(job))
                expected = load_protocol()['expected_records'][job.dataset]
                if population != expected:
                    return {'job': job.identity, 'phase': job.phase, 'method': job.method,
                            'dataset': job.dataset, 'status': 'blocked',
                            'reasons': [f'dataset count {population} != expected {expected}']}

            result = execute_job(
                job,
                profile_name='openrouter',
                mode='parallel',
                workers=args.workers_per_job,
                max_errors=args.max_errors,
                session_id=session,
                selected_ids=selection,
                retry_failed=args.retry_failed,
                allow_unverified=args.allow_unverified,
                max_shard_seconds=args.max_shard_seconds,
                results_root=root,
                protocol_mode='controlled_revision_v2',
            )
            result.setdefault('phase', job.phase)
            result.setdefault('method', job.method)
            result.setdefault('dataset', job.dataset)
            return result
        except Exception as ex:
            return {'job': job.identity, 'phase': job.phase, 'method': job.method,
                    'dataset': job.dataset, 'status': 'orchestration_error', 'reason': repr(ex)}

    all_results = []
    completed = 0
    launched = 0
    stop = False
    with campaign_lock(root):
        init_db(db)
        iterator = iter(jobs)
        with ThreadPoolExecutor(max_workers=args.parallel_jobs) as pool:
            pending = {}
            while True:
                while len(pending) < args.parallel_jobs and not stop:
                    if session_error_count(db, session) > args.max_errors:
                        stop = True
                        break
                    try:
                        next_job = next(iterator)
                    except StopIteration:
                        break
                    future = pool.submit(run_one, next_job)
                    pending[future] = next_job
                    launched += 1
                if not pending:
                    break
                done, _ = wait(pending, timeout=2.0, return_when=FIRST_COMPLETED)
                for future in done:
                    job = pending.pop(future)
                    result = future.result()
                    all_results.append(result)
                    completed += 1
                    status = result.get('status', '<missing>')
                    print(f'[{completed}/{len(jobs)}] {job.phase} {job.method}/{job.dataset}: '
                          f'{status} {result.get("counts", "")}', flush=True)
                    if status in ('failed', 'aborted', 'orchestration_error'):
                        stop = True
                        print('STOP: job-level failure. No new jobs will be scheduled.', flush=True)
                if session_error_count(db, session) > args.max_errors:
                    stop = True
                    print(f'STOP: more than {args.max_errors} terminal failures in this session.', flush=True)
        report = summarize_campaign(all_results, root, session, time.monotonic() - started)

    statuses = dict(Counter(row['status'] for row in all_results))
    print('\nCAMPAIGN COMPLETE')
    print('Started jobs:', launched, '/', len(jobs))
    print('Results:', statuses)
    print('Terminal failures:', session_error_count(db, session))
    print('Report:', report)
    print('To compute evaluation metrics:')
    print('  python scripts/kbs_protocol.py report --profile openrouter --phase main_gpt')
    print('  python scripts/kbs_protocol.py report --profile openrouter --phase main_qwen')
    if launched != len(jobs) or any(s in statuses for s in ('failed', 'aborted', 'orchestration_error', 'blocked')):
        print('ATTENTION: not all jobs finished successfully. Inspect logs/report before retrying.')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
