#!/usr/bin/env python3
"""Inspect and safely resume the RAGTree KBS 120-cell main campaign.

Only retries entire campaign when ALL errors in the previous session are
confirmed transient provider-overload HTTP 429 responses. The original tqdm
runner remains untouched, so checkpoint fingerprints and completed records stay
unchanged. Dry-run unless --execute is supplied.

Run at the RAGTree repository root after sourcing the OpenRouter env file.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time

DEFAULT_SESSION = '2d22578806f24ba4995b0435f59ede83'
PROFILE = 'openrouter'
PROTOCOL = 'controlled_revision_v2'


def classify(detail: str) -> str:
    """Only an unambiguous temporary provider overload may trigger paid retry."""
    d = detail.lower()
    if ('429' in d and (
        'engine_overloaded' in d or
        'upstream_provider_shared_pool' in d or
        'temporarily rate-limited upstream' in d or
        'temporarily unavailable' in d and ('provider' in d or 'upstream' in d)
    ) and not any(s in d for s in ('insufficient_quota', 'payment_required', 'invalid_api_key', 'credit limit'))):
        return 'TRANSIENT_UPSTREAM_429'
    if '429' in d or 'ratelimiterror' in d:
        return 'OTHER_429_CHECK_REQUIRED'
    if 'llm returned no usable final-answer text' in d:
        return 'EMPTY_MODEL_RESPONSE'
    if any(s in d for s in ('jsondecodeerror', 'invalid json', 'json parse', 'schema validation')):
        return 'PARSING_OR_SCHEMA'
    if any(s in d for s in ('timeout', 'timed out', 'connectionerror', 'connecterror')):
        return 'TIMEOUT_OR_NETWORK'
    if re.search(r'\b(500|502|503|504)\b', d):
        return 'PROVIDER_5XX'
    return 'OTHER_ERROR'


def short_reason(detail: str) -> str:
    """Return a sanitized, compact description without leaking request content."""
    d = detail.lower()
    if 'engine_overloaded' in d:
        return 'upstream engine overloaded'
    if 'upstream_provider_shared_pool' in d:
        return 'upstream shared-pool limit'
    if 'temporarily rate-limited upstream' in d:
        return 'provider temporarily rate-limited upstream'
    if 'llm returned no usable final-answer text' in d:
        return 'LLM response contains no usable final text'
    for marker in ('insufficient_quota', 'payment_required', 'context_length_exceeded',
                   'jsondecodeerror', 'timeout', 'unauthorized', 'invalid_api_key'):
        if marker in d:
            return marker
    match = re.search(r'([A-Za-z_.]+(?:Error|Exception))', detail)
    return match.group(1) if match else 'unclassified (inspect saved failure details)'


def get_report(reports: Path, session: str) -> dict:
    p = reports / f'campaign_{session}.json'
    if not p.is_file():
        raise RuntimeError(f'Campaign report not found: {p}')
    data = json.loads(p.read_text(encoding='utf8'))
    if data.get('session_id') != session:
        raise RuntimeError('Report session ID does not match file name; stopping')
    return data


def failures_for_session(db: Path, session: str) -> list[tuple[str, str, str]]:
    with sqlite3.connect(f'file:{db.as_posix()}?mode=ro', uri=True, timeout=30) as cx:
        return cx.execute(
            'SELECT run_id, doc_id, detail FROM failures WHERE session_id=?',
            (session,),
        ).fetchall()


def check_fingerprints(root: Path, report: dict) -> None:
    """Refuse paid calls if existing cells would silently acquire new run IDs."""
    sys.path.insert(0, str(root))
    from ragtree.experiments.kbs.protocol import job_matrix
    from ragtree.experiments.kbs.engine import run_id
    jobs = {(j.phase, j.method, j.dataset): j for j in job_matrix()}
    if len(jobs) != 120:
        raise RuntimeError(f'Unexpected main matrix size: {len(jobs)} (expected 120)')
    for row in report.get('results', []):
        k = (row.get('phase'), row.get('method'), row.get('dataset'))
        job = jobs.get(k)
        if job is None:
            raise RuntimeError(f'Previously reported cell is not in current matrix: {k}')
        expected_rid, _ = run_id(job, PROFILE, PROTOCOL)
        previous_rid = row.get('run_id')
        if previous_rid and expected_rid != previous_rid:
            raise RuntimeError(
                f'RUN ID CHANGED for {k}: prior {previous_rid}, current {expected_rid}. '
                'Do not launch: completed checkpoints would not be reused.'
            )


def audit_session(db: Path, reports: Path, session: str, *, verbose=True) -> tuple[dict, list, bool]:
    report = get_report(reports, session)
    failures = failures_for_session(db, session)
    statuses = Counter(str(row.get('status')) for row in report.get('results', []))
    categories = Counter(classify(detail) for _, _, detail in failures)
    by_run = defaultdict(Counter)
    labels = {r.get('run_id'): f'{r.get("phase")}/{r.get("method")}/{r.get("dataset")}'
              for r in report.get('results', [])}
    for rid, doc, detail in failures:
        by_run[labels.get(rid, 'unidentified:' + rid)][classify(detail)] += 1
    is_transient_only = bool(failures) and categories == {'TRANSIENT_UPSTREAM_429': len(failures)}
    if verbose:
        print(f'\n=== Campaign session {session} ===')
        print(f'Cells recorded: {len(report.get("results", []))} / 120; statuses: {dict(statuses)}')
        print(f'Failure rows: {len(failures)}; categorized: {dict(categories)}')
        for label, counts in sorted(by_run.items()):
            print(f'  {label}: {dict(counts)}')
        if failures:
            causes = Counter((classify(detail), short_reason(detail)) for _, _, detail in failures)
            print('Failure signatures (no prompts or credentials):')
            for (category, reason), count in causes.most_common(12):
                print(f'  {count:>4} x {category}: {reason}')
        print('Safe automatic retry:', 'YES (transient provider overload only)' if is_transient_only
              else 'NO (some errors are not verified upstream transient 429)')
    return report, failures, is_transient_only


def latest_new_report(reports: Path, known: set[str]) -> dict:
    new = [p for p in reports.glob('campaign_*.json') if p.name not in known]
    if len(new) != 1:
        raise RuntimeError(f'Expected exactly 1 newly generated report, found {len(new)}')
    return json.loads(new[0].read_text(encoding='utf8'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, default=Path('.'))
    ap.add_argument('--session', default=DEFAULT_SESSION,
                    help='Initial campaign session to inspect (default: last provided session)')
    ap.add_argument('--execute', action='store_true',
                    help='Explicit consent to resume the 120-cell campaign and spend OpenRouter credits')
    ap.add_argument('--parallel-jobs', type=int, default=2,
                    help='Concurrent cells (default 2; reduced load)')
    ap.add_argument('--workers-per-job', type=int, default=2,
                    help='Parallel document shards per cell (default 2)')
    ap.add_argument('--max-errors', type=int, default=100,
                    help='Session-wide terminal error budget (default 100)')
    ap.add_argument('--max-restarts', type=int, default=8,
                    help='Automatic retries following ONLY verified upstream 429 sessions')
    ap.add_argument('--initial-backoff', type=float, default=180,
                    help='First provider cooldown, seconds (default 180)')
    ap.add_argument('--max-backoff', type=float, default=1200,
                    help='Maximum provider cooldown, seconds (default 1200)')
    args = ap.parse_args()
    if not (1 <= args.parallel_jobs <= 8 and 1 <= args.workers_per_job <= 16
            and args.parallel_jobs * args.workers_per_job <= 32):
        ap.error('Invalid concurrency; max 32 subprocesses')
    if args.max_errors < 0 or not (0 <= args.max_restarts <= 30):
        ap.error('Invalid error/restart limit')
    if args.initial_backoff < 1 or args.max_backoff < args.initial_backoff:
        ap.error('Invalid cooldown limits')
    root = args.repo.resolve()
    results = root / 'results' / 'kbs_journal'
    db = results / 'checkpoints.sqlite'
    reports = results / 'reports'
    runner = root / 'kbs_first_parallel_campaign_tqdm.py'
    if not (db.is_file() and runner.is_file()):
        ap.error('Cannot find checkpoints.sqlite and the tqdm campaign runner in --repo')
    report, errors, safe = audit_session(db, reports, args.session)
    if not args.execute:
        print('\nREAD ONLY. No inference, no checkpoint edits, no charges.')
        if safe:
            print('Add --execute to resume automatically with cooldowns and full checkpoints.')
        else:
            print('STOP: address nontransient failure categories before restarting.')
        return 0
    if not safe:
        print('STOP: cannot safely auto-retry this campaign; errors are not all confirmed upstream 429.',
              file=sys.stderr)
        return 2
    if not os.environ.get('OPENROUTER_API_KEY'):
        print('STOP: OPENROUTER_API_KEY not defined. Source env file.', file=sys.stderr)
        return 2
    check_fingerprints(root, report)

    cmd = [sys.executable, str(runner), '--execute',
           '--parallel-jobs', str(args.parallel_jobs),
           '--workers-per-job', str(args.workers_per_job),
           '--max-errors', str(args.max_errors), '--retry-failed']
    for attempt in range(args.max_restarts + 1):
        if attempt:
            delay = min(args.max_backoff, args.initial_backoff * 2 ** (attempt - 1))
            print(f'\nCooling down for {delay:g}s before next session; Ctrl+C cancels restarts.',
                  flush=True)
            try:
                time.sleep(delay)
            except KeyboardInterrupt:
                print('Stopped during cooldown. Completed checkpoints remain saved.')
                return 130
        print(f'\n=== RESUME {attempt+1}/{args.max_restarts+1}: '
              f'{args.parallel_jobs} cells x {args.workers_per_job} workers, '
              f'{args.max_errors} session-wide errors ===', flush=True)
        known = {p.name for p in reports.glob('campaign_*.json')}
        try:
            exit_code = subprocess.run(cmd, cwd=root, check=False).returncode
        except KeyboardInterrupt:
            print('Interrupted. Check that the child executor has stopped before restarting.',
                  file=sys.stderr)
            return 130
        try:
            new_report = latest_new_report(reports, known)
        except Exception as e:
            print(f'STOP: cannot identify new report safely: {e}', file=sys.stderr)
            return 2
        sid = new_report.get('session_id', '')
        fresh_report, errors, safe = audit_session(db, reports, sid)
        print(f'Executor exit code: {exit_code}', flush=True)
        # The original runner returns 0 only when it has scheduled all 120 jobs
        # and found no incomplete or aborted jobs.
        if exit_code == 0 and len(fresh_report.get('results', [])) == 120 \
                and all(r.get('status') == 'completed' for r in fresh_report['results']):
            print('SUCCESS: all 120 main cells completed; no more inference needed.')
            return 0
        if not safe:
            print('STOP: encountered nontransient/unverified errors, or stopped for '
                  'another reason. Nothing will be retried automatically.', file=sys.stderr)
            return 1
        check_fingerprints(root, fresh_report)
        if attempt == args.max_restarts:
            print('STOP: automatic restart limit reached. Completed checkpoints are intact.',
                  file=sys.stderr)
            return 1
        print('All observed failures are confirmed temporary upstream overloads. '
              'Saved checkpoints remain intact; next session will resume missing work.',
              flush=True)
    return 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'STOP (no silent retries): {exc}', file=sys.stderr)
        raise SystemExit(2)
