#!/usr/bin/env python3
"""Resume ONE KBS GPT-OSS/MAVEN-ERE cell, retrying only documented DeepInfra overloads.

- Dry-run by default. Add --execute for potentially billed inference.
- Uses the existing tqdm campaign runner unchanged; preserves its fingerprint.
- Every restart receives a new campaign session ID via the original runner.
- Retries only when every error in that session is DeepInfra shared-pool 429.
- Stops on different errors, unexpected run_id, incomplete reports, or retry cap.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

EXPECTED_PHASE = 'main_gpt'
EXPECTED_METHOD = 'baseline'
EXPECTED_DATASET = 'maven_ere'
EXPECTED_DOCUMENTS = 3516


def record_counts(db: Path, run_id: str) -> dict[str, int]:
    with sqlite3.connect(db, timeout=30) as con:
        return dict(con.execute(
            'SELECT status, COUNT(*) FROM records WHERE run_id=? GROUP BY status',
            (run_id,),
        ).fetchall())


def session_failures(db: Path, session: str, run_id: str) -> list[tuple[str, str]]:
    with sqlite3.connect(db, timeout=30) as con:
        return con.execute(
            'SELECT doc_id, detail FROM failures WHERE session_id=? AND run_id=?',
            (session, run_id),
        ).fetchall()


def all_deepinfra_upstream_429(failures: list[tuple[str, str]]) -> bool:
    return bool(failures) and all(
        ('429' in detail and
         'DeepInfra' in detail and
         'engine_overloaded' in detail and
         'upstream_provider_shared_pool' in detail)
        for _, detail in failures
    )


def new_report(directory: Path, before: set[str]) -> dict:
    added = sorted((p for p in directory.glob('campaign_*.json') if p.name not in before),
                   key=lambda p: p.stat().st_mtime)
    if len(added) != 1:
        raise RuntimeError(f'Expected exactly one new report, got {len(added)}. '
                           'Stopping to avoid retrying the wrong campaign.')
    return json.loads(added[0].read_text(encoding='utf-8'))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path('.'))
    p.add_argument('--execute', action='store_true')
    p.add_argument('--expect-run-id', required=True,
                   help='Confirmed checkpoint run_id; prevents unintended new runs')
    p.add_argument('--workers-per-job', type=int, default=3)
    p.add_argument('--max-errors', type=int, default=3)
    p.add_argument('--max-restarts', type=int, default=5,
                   help='Maximum re-launches after a run stopped ONLY on DeepInfra 429s')
    p.add_argument('--initial-backoff', type=float, default=60,
                   help='Wait after first 429 batch, seconds')
    p.add_argument('--max-backoff', type=float, default=300,
                   help='Upper bound on exponential wait, seconds')
    args = p.parse_args(argv)
    if not 1 <= args.workers_per_job <= 16 or not 0 <= args.max_errors <= 100:
        p.error('workers-per-job must be 1..16, max-errors 0..100')
    if not 0 <= args.max_restarts <= 20 or args.initial_backoff < 1 or args.max_backoff < args.initial_backoff:
        p.error('invalid restart/backoff limits')
    root = args.repo.resolve()
    runner = root / 'kbs_first_parallel_campaign_tqdm.py'
    db = root / 'results/kbs_journal/checkpoints.sqlite'
    reports = root / 'results/kbs_journal/reports'
    if not runner.is_file() or not db.is_file():
        p.error('Cannot find original tqdm runner or checkpoint database under --repo')

    # Before the first paid call, calculate the actual fingerprint from the current
    # repository. This catches code/protocol edits that would invalidate resume.
    sys.path.insert(0, str(root))
    from ragtree.experiments.kbs.engine import run_id
    from ragtree.experiments.kbs.protocol import job_matrix, load_protocol
    matches = [j for j in job_matrix() if j.phase == EXPECTED_PHASE
               and j.method == EXPECTED_METHOD and j.dataset == EXPECTED_DATASET]
    if len(matches) != 1:
        raise RuntimeError(f'Expected one {EXPECTED_PHASE} {EXPECTED_METHOD}/{EXPECTED_DATASET} job; got {len(matches)}')
    job = matches[0]
    actual_run_id, _ = run_id(job, 'openrouter', 'controlled_revision_v2')
    if actual_run_id != args.expect_run_id:
        raise RuntimeError(f'RUN ID CHANGED: expected {args.expect_run_id}, found {actual_run_id}. '
                           'STOP: do not restart; old done checkpoints would not be reused.')
    expected = int(load_protocol()['expected_records'][EXPECTED_DATASET])
    if expected != EXPECTED_DOCUMENTS:
        raise RuntimeError(f'Expected {EXPECTED_DOCUMENTS} records; protocol says {expected}')

    counts = record_counts(db, actual_run_id)
    print(f'Checkpoint {actual_run_id}: done={counts.get("done", 0)}, '
          f'failed={counts.get("failed", 0)}, target={expected}.', flush=True)
    if counts.get('done', 0) == expected and counts.get('failed', 0) == 0:
        print('Already complete; nothing to run.')
        return 0
    if not args.execute:
        print('DRY RUN ONLY. Add --execute to retry missing/failed documents.')
        return 0
    if not os.environ.get('OPENROUTER_API_KEY'):
        p.error('OPENROUTER_API_KEY is missing; source your environment file first')

    command = [sys.executable, str(runner), '--execute',
               '--only-phase', EXPECTED_PHASE, '--max-jobs', '1',
               '--parallel-jobs', '1', '--workers-per-job', str(args.workers_per_job),
               '--max-errors', str(args.max_errors), '--retry-failed']
    for attempt in range(args.max_restarts + 1):
        print(f'\n=== Attempt {attempt + 1}/{args.max_restarts + 1}: '
              f'{EXPECTED_PHASE} {EXPECTED_METHOD}/{EXPECTED_DATASET}, '
              f'{args.workers_per_job} parallel workers ===', flush=True)
        before = {x.name for x in reports.glob('campaign_*.json')}
        exit_code = subprocess.run(command, cwd=root, check=False).returncode
        try:
            report = new_report(reports, before)
        except Exception as ex:
            print(f'STOP: cannot safely identify campaign report: {ex}', file=sys.stderr)
            return 2
        if report.get('experiment_jobs') != 1 or len(report.get('results', [])) != 1:
            print('STOP: report unexpectedly contains more than one job.', file=sys.stderr)
            return 2
        row = report['results'][0]
        session = report.get('session_id', '')
        if row.get('run_id') != actual_run_id or any(
            row.get(k) != v for k, v in (
                ('phase', EXPECTED_PHASE), ('method', EXPECTED_METHOD),
                ('dataset', EXPECTED_DATASET))
        ):
            print('STOP: wrong job or run_id in report.', file=sys.stderr)
            return 2
        counts = record_counts(db, actual_run_id)
        done, failed = counts.get('done', 0), counts.get('failed', 0)
        print(f'\nCheckpoint after attempt: done={done}/{expected}, failed={failed}, '
              f'status={row.get("status")}, runner_exit={exit_code}', flush=True)
        if done == expected and failed == 0 and row.get('status') == 'completed':
            print('SUCCESS: complete dataset with zero failed checkpoints.')
            return 0
        failures = session_failures(db, session, actual_run_id)
        if row.get('status') not in ('aborted', 'failed', 'incomplete', 'completed_with_failures') \
                or not all_deepinfra_upstream_429(failures):
            print('STOP: at least one failure is not a confirmed DeepInfra shared-pool 429, '
                  'or the job stopped for another reason. Investigate before paying again.',
                  file=sys.stderr)
            return 1
        print(f'Transient provider overload: {len(failures)} DeepInfra 429 errors '
              'in this attempt; saved checkpoints will be reused.', flush=True)
        if attempt >= args.max_restarts:
            print('STOP: maximum automatic restarts reached. Resume manually after diagnosis.',
                  file=sys.stderr)
            return 1
        delay = min(args.max_backoff, args.initial_backoff * 2 ** attempt)
        print(f'Cooling down for {delay:.0f} seconds before safe restart. '
              'Ctrl+C stops only this controller.', flush=True)
        try:
            time.sleep(delay)
        except KeyboardInterrupt:
            print('\nStopped before next restart; checkpoints preserved.')
            return 130
    return 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nInterrupted; completed checkpoints remain saved.', file=sys.stderr)
        raise SystemExit(130)
