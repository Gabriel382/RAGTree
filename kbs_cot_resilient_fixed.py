#!/usr/bin/env python3
"""Resumable CoT KBS GPT-OSS controller for transient OpenRouter/DeepInfra 429s.

This is a *scheduler*, not a replacement relation-extraction method. It calls
RAGTree's existing engine.execute_job() and scripts/run_cot_baseline.py in
controlled_revision_v2, with the same model, prompts, fingerprint and scoring.

- Read-only preflight by default; --execute permits billable inference.
- Never changes package code, YAML, the strategy, or existing done checkpoints.
- Runs small deterministic selections under a new session ID per batch, so
  a session-wide 100-error cutoff cannot repeatedly kill all remaining work.
- One confirmed upstream DeepInfra 429 triggers a global provider cooldown;
  then it probes one pending record before resuming full batches.
- Only confirmed upstream 429s are retried; any other error halts the run.
- Outage state is persisted across controller restarts.
- A final no-inference full-population pass restores manifest selected='all'.

No external controller can guarantee provider availability. Long outages cause
bounded pause and ultimately a safe stop with checkpoint preservation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
import uuid

EXPECTED_MAVEN_RUN_ID = '5da2154163b1983a9cd86212'  # confirmed in user's campaign report
PHASE = 'main_gpt'
METHOD = 'cot'
PROFILE = 'openrouter'
PROTOCOL_MODE = 'controlled_revision_v2'


def classify_failure(detail: str) -> str:
    d = detail.lower()
    if ('429' in d and 'deepinfra' in d and
            ('engine_overloaded' in d or 'upstream_provider_shared_pool' in d or
             'temporarily rate-limited upstream' in d)):
        return 'TRANSIENT_DEEPINFRA_429'
    return 'NONTRANSIENT_OR_UNKNOWN'


def db_status(db: Path, rid: str) -> dict[str, str]:
    # The SQLite database remains RAGTree's source of truth; this controller
    # never inserts, removes, or modifies its rows.
    with sqlite3.connect(db, timeout=30) as con:
        return {str(k): str(v) for k, v in con.execute(
            'SELECT doc_id,status FROM records WHERE run_id=?', (rid,))}


def session_failures(db: Path, rid: str, session: str):
    with sqlite3.connect(db, timeout=30) as con:
        return con.execute(
            'SELECT doc_id,detail FROM failures WHERE session_id=? AND run_id=?',
            (session, rid),
        ).fetchall()


def read_state(path: Path) -> dict:
    if not path.is_file():
        return {'outage_streak': 0, 'next_retry_after': 0}
    data = json.loads(path.read_text(encoding='utf8'))
    if data.get('phase') != PHASE or data.get('method') != METHOD:
        raise RuntimeError(f'Controller state has unexpected phase or method: {path}')
    return data


def write_state(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use atomic replace; never touch experiment manifests or SQLite.
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(json.dumps(state, indent=2) + '\n', encoding='utf8')
    os.replace(tmp, path)


def cooldown(seconds: float, message: str):
    if seconds <= 0:
        return
    print(f'COOLDOWN: {seconds:.0f}s — {message}. Ctrl+C stops safely.', flush=True)
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        time.sleep(min(1.0, end - time.monotonic()))


class LiveProgress:
    """Read-only SQLite progress while engine.execute_job runs synchronously."""
    def __init__(self, db: Path, rids: dict[str, str], expected: dict[str, int], interval: float):
        from tqdm import tqdm
        self.db, self.rids, self.expected = db, rids, expected
        self.interval = interval
        self.stop = threading.Event()
        self.thread = None
        self.total = sum(expected.values())
        self.initial = self.read_total()
        self.bar = tqdm(total=self.total, initial=self.initial, unit='docs',
                        desc='CoT GPT total', dynamic_ncols=True, mininterval=1)
        self.last = self.initial

    def read_total(self):
        total = 0
        with sqlite3.connect(self.db, timeout=30) as cx:
            for key, rid in self.rids.items():
                n = cx.execute('SELECT COUNT(*) FROM records WHERE run_id=? AND status=?',
                               (rid, 'done')).fetchone()[0]
                total += min(n, self.expected[key])
        return total

    def refresh(self):
        try:
            n = self.read_total()
            if n > self.last:
                self.bar.update(n - self.last)
                self.last = n
        except sqlite3.Error:
            pass

    def _poll(self):
        while not self.stop.wait(self.interval):
            self.refresh()

    def __enter__(self):
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, typ, exc, tb):
        self.stop.set()
        self.thread.join(timeout=self.interval + 1)
        self.refresh()
        self.bar.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, default=Path('.'))
    ap.add_argument('--execute', action='store_true', help='Allow paid requests (dry-run by default)')
    ap.add_argument('--dataset', default=None,
                    help='Optional single CoT dataset; default all five in the protocol order')
    ap.add_argument('--batch-size', type=int, default=48,
                    help='Maximum documents per bounded batch; default 48')
    ap.add_argument('--workers', type=int, default=2,
                    help='Parallel document shards while healthy (1 or 2; default 2)')
    ap.add_argument('--max-overloads', type=int, default=12,
                    help='Maximum consecutive provider-outage batches, including across restarts')
    ap.add_argument('--initial-backoff', type=int, default=120,
                    help='First upstream overload cooldown in seconds (default 120)')
    ap.add_argument('--max-backoff', type=int, default=1200,
                    help='Maximum cooldown in seconds (default 1200)')
    ap.add_argument('--batch-pause', type=float, default=2.0,
                    help='Short delay between healthy batches in seconds (default 2)')
    ap.add_argument('--progress-interval', type=float, default=3.0)
    ap.add_argument('--max-batches', type=int, default=10000,
                    help='Safety cutoff for batch submissions')
    ap.add_argument('--expect-maven-run-id', default=EXPECTED_MAVEN_RUN_ID,
                    help='Expected existing CoT/MAVEN run ID; change only deliberately')
    args = ap.parse_args(argv)
    if not 1 <= args.workers <= 2 or not 1 <= args.batch_size <= 1000:
        ap.error('workers must be 1 or 2; batch-size must be 1..1000')
    if (args.initial_backoff < 1 or args.max_backoff < args.initial_backoff or
            args.max_overloads < 1 or args.max_batches < 1 or
            args.batch_pause < 0 or args.progress_interval < 1):
        ap.error('Invalid cooldown, outage, batch, or progress limits')
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    from ragtree.experiments.kbs.engine import (
        OUT, campaign_lock, execute_job, filtered_docs, preflight, run_id,
    )
    from ragtree.experiments.kbs.protocol import job_matrix, load_protocol
    from ragtree.experiments.kbs.runtime import record_path
    from ragtree.experiments.kbs.record_identity import checkpoint_id

    result_root = OUT.resolve()
    db = result_root / 'checkpoints.sqlite'
    if not db.is_file():
        ap.error(f'Existing checkpoint DB not found: {db}')

    jobs = [j for j in job_matrix() if j.phase == PHASE and j.method == METHOD]
    if len(jobs) != 5 or len({j.dataset for j in jobs}) != 5:
        raise RuntimeError(f'Expected five unique CoT {PHASE} jobs, got {len(jobs)}')
    if args.dataset:
        jobs = [j for j in jobs if j.dataset == args.dataset]
        if len(jobs) != 1:
            ap.error(f'Unknown CoT dataset: {args.dataset}')

    protocol = load_protocol()
    states: list[dict] = []
    for j in jobs:
        issues = preflight(j)['issues']
        if issues:
            raise RuntimeError(f'Preflight failed for {j.dataset}: {issues}')
        rid, _ = run_id(j, PROFILE, PROTOCOL_MODE)
        if j.dataset == 'maven_ere' and args.expect_maven_run_id and rid != args.expect_maven_run_id:
            raise RuntimeError('RUN ID CHANGED for CoT/MAVEN-ERE: '\
                               f'expected {args.expect_maven_run_id}, current {rid}. '\
                               'Stop before inference to preserve prior checkpoints.')
        ids = [checkpoint_id(d) for d in filtered_docs(j)]
        expected = int(protocol['expected_records'][j.dataset])
        if len(ids) != expected or len(set(ids)) != len(ids):
            raise RuntimeError(f'{j.dataset}: expected {expected} unique checkpoint IDs; found rows={len(ids)}, unique={len(set(ids))}')
        statuses = db_status(db, rid)
        run_root = result_root / PROFILE / PHASE / METHOD / j.dataset / rid
        # Refuse to trust stale SQLite "done" rows if the actual persisted
        # prediction payload is absent or belongs to another fingerprint.
        for ident in ids:
            if statuses.get(ident) != 'done':
                continue
            path = record_path(run_root, ident)
            if not path.is_file():
                raise RuntimeError(f'{j.dataset}: SQLite done but checkpoint file missing for {ident}')
            payload = json.loads(path.read_text(encoding='utf8'))
            if payload.get('run_id') != rid or payload.get('document_id') != ident:
                raise RuntimeError(f'{j.dataset}: mismatched checkpoint payload for {ident}')
        current_done = sum(statuses.get(i) == 'done' for i in ids)
        states.append({'job': j, 'rid': rid, 'ids': ids, 'expected': expected})
        print(f'PRECHECK {j.dataset}: {current_done}/{expected} done, '
              f'{sum(statuses.get(i)=="failed" for i in ids)} failed, run_id={rid}', flush=True)

    if not args.execute:
        print('DRY RUN — no inference, no changes. Add --execute to resume CoT.')
        return 0
    if not os.environ.get('OPENROUTER_API_KEY'):
        ap.error('OPENROUTER_API_KEY missing; source ~/.config/ragtree/openrouter.env')
    try:
        from tqdm import tqdm  # noqa: F401
    except ImportError:
        ap.error('tqdm missing; run python -m pip install tqdm')

    state_path = result_root / 'reports' / 'cot_resilient_main_gpt_state.json'
    state = read_state(state_path)
    state.setdefault('phase', PHASE)
    state.setdefault('method', METHOD)
    rids = {s['job'].dataset: s['rid'] for s in states}
    expecteds = {s['job'].dataset: s['expected'] for s in states}
    batch_counter = 0
    with campaign_lock(result_root):
        with LiveProgress(db, rids, expecteds, args.progress_interval) as progress:
            for s in states:
                j, rid, ids, expected = s['job'], s['rid'], s['ids'], s['expected']
                root = result_root / PROFILE / PHASE / METHOD / j.dataset / rid
                print(f'\n=== CoT/GPT {j.dataset}: target {expected}, run_id={rid} ===', flush=True)
                while True:
                    status = db_status(db, rid)
                    pending = [i for i in ids if status.get(i) not in ('done', 'failed')]
                    pending += [i for i in ids if status.get(i) == 'failed']
                    if not pending:
                        break
                    if batch_counter >= args.max_batches:
                        print('SAFE STOP: max batch submissions reached. Checkpoints intact.', flush=True)
                        return 1
                    if int(state.get('outage_streak', 0)) >= args.max_overloads:
                        print('SAFE STOP: provider outage limit reached; no more paid retries.', flush=True)
                        return 2
                    remaining_cooldown = float(state.get('next_retry_after', 0)) - time.time()
                    if remaining_cooldown > 0:
                        cooldown(remaining_cooldown, 'persisted DeepInfra outage')
                    probe = bool(state.get('probing', False))
                    subset = pending[:1 if probe else args.batch_size]
                    workers = 1 if probe else min(args.workers, len(subset))
                    session = uuid.uuid4().hex
                    before = sum(status.get(i) == 'done' for i in ids)
                    batch_counter += 1
                    print(f'BATCH {batch_counter} {j.dataset}: '
                          f'{before}/{expected} done; selected={len(subset)}, '
                          f'workers={workers}, recovery_probe={probe}', flush=True)
                    result = execute_job(
                        j, profile_name=PROFILE, mode='parallel',
                        workers=workers, max_errors=0, session_id=session,
                        selected_ids=subset, retry_failed=True,
                        allow_unverified=False, max_shard_seconds=86400,
                        results_root=result_root, protocol_mode=PROTOCOL_MODE,
                    )
                    progress.refresh()
                    failures = session_failures(db, rid, session)
                    kinds = Counter(classify_failure(detail) for _, detail in failures)
                    after_status = db_status(db, rid)
                    after = sum(after_status.get(i) == 'done' for i in ids)
                    selected_done = sum(after_status.get(i) == 'done' for i in subset)
                    print(f'  status={result.get("status")}; now={after}/{expected}; '
                          f'batch_done={selected_done}/{len(subset)}; '
                          f'new_done={after-before}; failures={dict(kinds)}', flush=True)

                    if any(k != 'TRANSIENT_DEEPINFRA_429' for k in kinds):
                        print('SAFE STOP: non-429 failure; diagnose saved session '
                              f'{session}. No blind retry.', flush=True)
                        return 3
                    if result.get('status') in ('blocked', 'orchestration_error'):
                        print('SAFE STOP: job blocked/orchestration failure: '
                              f'{result.get("reason") or result.get("reasons")}', flush=True)
                        return 3
                    if not failures and selected_done < len(subset):
                        print('SAFE STOP: missing checkpoints without recorded 429. '
                              f'Session {session}', flush=True)
                        return 3

                    if failures:
                        outage = int(state.get('outage_streak', 0)) + 1
                        delay = min(args.max_backoff, args.initial_backoff * 2 ** min(outage - 1, 16))
                        state.update({'outage_streak': outage, 'probing': True,
                                      'next_retry_after': time.time() + delay,
                                      'last_session': session, 'last_dataset': j.dataset,
                                      'phase': PHASE, 'method': METHOD})
                        write_state(state_path, state)
                        print(f'  DeepInfra shared-pool 429: circuit OPEN. '
                              f'Next single-document probe after {delay}s.', flush=True)
                        continue

                    if after <= before:
                        print('SAFE STOP: no forward progress despite a clean batch.', flush=True)
                        return 3

                    if probe:
                        print('  Recovery probe succeeded; circuit CLOSED.', flush=True)
                    state.update({'outage_streak': 0, 'probing': False,
                                  'next_retry_after': 0, 'phase': PHASE, 'method': METHOD,
                                  'last_session': session, 'last_dataset': j.dataset})
                    write_state(state_path, state)
                    if args.batch_pause:
                        cooldown(args.batch_pause, 'healthy pacing between batches')

                # Validate exact current population and record-file payloads.
                status = db_status(db, rid)
                if any(status.get(i) != 'done' for i in ids):
                    raise RuntimeError(f'{j.dataset} marked complete without all done statuses')
                for ident in ids:
                    path = record_path(root, ident)
                    if not path.is_file():
                        raise RuntimeError(f'Missing done checkpoint file: {ident}')
                    data = json.loads(path.read_text(encoding='utf8'))
                    if data.get('run_id') != rid or data.get('document_id') != ident:
                        raise RuntimeError(f'Invalid checkpoint payload: {ident}')

                # The selected batch writes a partial "manifest.json". Re-run the
                # original engine against the *entire* population (all records are
                # already committed) to restore selected='all' and original reporting
                # semantics. The runtime skips all done records; no LLM calls.
                print(f'FINALIZE {j.dataset}: full-population no-inference pass', flush=True)
                final_session = uuid.uuid4().hex
                fin = execute_job(
                    j, profile_name=PROFILE, mode='sequential', workers=1,
                    max_errors=0, session_id=final_session,
                    selected_ids=None, retry_failed=False,
                    allow_unverified=False, max_shard_seconds=86400,
                    results_root=result_root, protocol_mode=PROTOCOL_MODE,
                )
                if fin.get('run_id') != rid or fin.get('status') not in ('completed',):
                    raise RuntimeError(f'Full-population finalization failed: {fin}')
                if session_failures(db, rid, final_session):
                    raise RuntimeError('Finalization unexpectedly recorded failures')
                status = db_status(db, rid)
                if sum(status.get(i) == 'done' for i in ids) != expected:
                    raise RuntimeError('Finalized population differs from expected total')
                manifest = json.loads((root / 'manifest.json').read_text(encoding='utf8'))
                if manifest.get('selected') != 'all':
                    raise RuntimeError('Final manifest not full-population')
                progress.refresh()
                print(f'COMPLETE: {j.dataset} {expected}/{expected}, full-population manifest restored.', flush=True)

    print('\nALL REQUESTED GPT CoT DATASETS COMPLETE. No benchmark code changed.', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nInterrupted; engine terminates children; saved checkpoints remain.', file=sys.stderr)
        raise SystemExit(130)
