#!/usr/bin/env python3
"""Resume KBS main_gpt non-CoT jobs safely during transient upstream 429 outages.

This is an orchestration-only tool, not a change to any RAG method, scorer,
protocol or recorded run identity. Defaults to read-only. --execute authorizes
paid inference. It refuses to retry after any unclassified/terminal error.

When healthy it submits a bounded batch of missing/failed document IDs through
RAGTree's engine.execute_job, using the original source script. On *confirmed*
upstream-provider 429 it opens a circuit breaker and probes one document after
an exponential cooldown. It saves outage state across program restarts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
import uuid

PHASE = 'main_gpt'
PROFILE = 'openrouter'
PROTOCOL = 'controlled_revision_v2'
DEFAULT_SOURCE_SESSION = 'd22829fa988f47a0ba66c6b85ad09c76'


def failure_category(detail: str) -> str:
    d = detail.lower()
    if ('429' in d and ('engine_overloaded' in d or
        'upstream_provider_shared_pool' in d or
        'temporarily rate-limited upstream' in d) and not
        any(x in d for x in ('insufficient_quota', 'payment_required',
                              'invalid_api_key', 'credit limit'))):
        return 'UPSTREAM_OVERLOAD_429'
    return 'OTHER_OR_UNKNOWN'


def read_records(db: Path, rid: str) -> dict[str,str]:
    with sqlite3.connect(f'file:{db.as_posix()}?mode=ro', uri=True, timeout=30) as c:
        return {str(doc): str(status) for doc, status in c.execute(
            'SELECT doc_id,status FROM records WHERE run_id=?', (rid,))}


def read_failures(db: Path, session: str, rid: str | None = None):
    with sqlite3.connect(f'file:{db.as_posix()}?mode=ro', uri=True, timeout=30) as c:
        if rid:
            return c.execute('SELECT run_id,doc_id,detail FROM failures WHERE session_id=? AND run_id=?',
                             (session,rid)).fetchall()
        return c.execute('SELECT run_id,doc_id,detail FROM failures WHERE session_id=?',
                         (session,)).fetchall()


def audit_source(db: Path, report_root: Path, session: str):
    p = report_root / f'campaign_{session}.json'
    if not p.is_file():
        raise RuntimeError(f'Cannot verify previous campaign report: {p}')
    report = json.loads(p.read_text(encoding='utf-8'))
    if report.get('session_id') != session:
        raise RuntimeError('Previous campaign session ID mismatch')
    failures = read_failures(db, session)
    counts = Counter(failure_category(x[2]) for x in failures)
    print(f'PREVIOUS SESSION: {session}, jobs={len(report.get("results",[]))}, '
          f'failure_rows={len(failures)}, categories={dict(counts)}', flush=True)
    if not failures or counts != {'UPSTREAM_OVERLOAD_429': len(failures)}:
        raise RuntimeError('Previous session contains non-429 or unverified failures. '
                           'No paid requests will be made.')
    return report


def read_state(path: Path) -> dict:
    if path.exists():
        state = json.loads(path.read_text(encoding='utf-8'))
        if state.get('controller') != 'gpt_rag_resilient_v1':
            raise RuntimeError('Unknown persisted controller state')
        return state
    return {'controller': 'gpt_rag_resilient_v1',
            'outage_streak': 0, 'probing': False, 'next_retry_after': 0}


def save_state(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.'+path.name+'.tmp')
    tmp.write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
    os.replace(tmp,path)


def sleep_seconds(n: float, why: str):
    if n <= 0:
        return
    print(f'COOLDOWN {n:.0f}s ({why}); Ctrl+C cancels cleanly.',flush=True)
    end = time.monotonic()+n
    while time.monotonic()<end:
        time.sleep(min(1,end-time.monotonic()))


class Progress:
    def __init__(self, db, tasks, interval):
        from tqdm import tqdm
        self.tqdm=tqdm
        self.db=db
        self.tasks=tasks
        self.interval=interval
        self.total=sum(x['expected'] for x in tasks)
        self.prev=self.count()
        self.bar=tqdm(total=self.total,initial=self.prev,desc='GPT non-CoT',
                      unit='docs',dynamic_ncols=True,mininterval=1)
        self.stopping=threading.Event()
        self.thread=None

    def count(self):
        with sqlite3.connect(f'file:{self.db.as_posix()}?mode=ro',uri=True,timeout=30) as cx:
            return sum(min(x['expected'], cx.execute(
                "SELECT COUNT(*) FROM records WHERE run_id=? AND status='done'",
                (x['rid'],)).fetchone()[0]) for x in self.tasks)

    def update(self):
        try:
            n=self.count()
            if n>self.prev:
                self.bar.update(n-self.prev)
                self.prev=n
        except sqlite3.Error:
            pass

    def poll(self):
        while not self.stopping.wait(self.interval):
            self.update()

    def __enter__(self):
        self.thread=threading.Thread(target=self.poll,daemon=True)
        self.thread.start()
        return self

    def __exit__(self,*args):
        self.stopping.set()
        self.thread.join(timeout=self.interval+1)
        self.update()
        self.bar.close()


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,default=Path('.'))
    ap.add_argument('--execute',action='store_true',help='Explicit consent to paid inference')
    ap.add_argument('--source-session',default=DEFAULT_SOURCE_SESSION,
                    help='Campaign session whose errors and run fingerprints must be verified')
    ap.add_argument('--method',default=None,help='Restrict to one non-CoT method')
    ap.add_argument('--dataset',default=None,help='Restrict to one dataset')
    ap.add_argument('--batch-size',type=int,default=512,
                    help='Documents selected per RAG process batch (default 512)')
    ap.add_argument('--workers',type=int,default=2,help='Parallel shards within a batch (1 or 2)')
    ap.add_argument('--initial-backoff',type=int,default=180,
                    help='First provider-overload cooldown (seconds)')
    ap.add_argument('--max-backoff',type=int,default=1200,
                    help='Maximum cooldown in seconds')
    ap.add_argument('--max-overloads',type=int,default=12,
                    help='Max consecutive overloaded batches, across restarts')
    ap.add_argument('--healthy-pause',type=float,default=1.0)
    ap.add_argument('--progress-interval',type=float,default=3.0)
    ap.add_argument('--max-batches',type=int,default=10000)
    args=ap.parse_args(argv)
    if not (1<=args.workers<=2 and 1<=args.batch_size<=2000 and
            args.initial_backoff>=1 and args.max_backoff>=args.initial_backoff and
            args.max_overloads>=1 and args.max_batches>=1 and
            args.healthy_pause>=0 and args.progress_interval>=1):
        ap.error('Invalid concurrency, batch, cooldown, or safety limits')
    repo=args.repo.resolve()
    sys.path.insert(0,str(repo))
    from ragtree.experiments.kbs.engine import (
        OUT, campaign_lock, execute_job, filtered_docs, preflight, run_id,
    )
    from ragtree.experiments.kbs.protocol import job_matrix, load_protocol
    from ragtree.experiments.kbs.runtime import record_path

    root=OUT.resolve()
    db=root/'checkpoints.sqlite'
    if not db.is_file():
        raise RuntimeError(f'No checkpoint database: {db}')
    ref=audit_source(db,root/'reports',args.source_session)
    expected_map=load_protocol()['expected_records']
    main=[j for j in job_matrix() if j.phase==PHASE]
    if len(main)!=60:
        raise RuntimeError(f'Expected 60 main GPT configurations, got {len(main)}')
    all_jobs=[j for j in main if j.method!='cot']
    if len(all_jobs)!=55:
        raise RuntimeError(f'Expected 55 GPT non-CoT cells, got {len(all_jobs)}')
    prior={(x.get('phase'),x.get('method'),x.get('dataset')):x
           for x in ref.get('results',[])}
    for j in main:
        prev=prior.get((j.phase,j.method,j.dataset))
        if not prev or not prev.get('run_id'):
            continue
        current,_=run_id(j,PROFILE,PROTOCOL)
        if current!=prev['run_id']:
            raise RuntimeError(f'RUN ID CHANGED for {j.method}/{j.dataset}: '
                               f'{prev["run_id"]} => {current}; refusing paid rerun')
    tasks=[]
    for j in all_jobs:
        if args.method and j.method!=args.method:
            continue
        if args.dataset and j.dataset!=args.dataset:
            continue
        issues=preflight(j)['issues']
        if issues:
            raise RuntimeError(f'Preflight {j.method}/{j.dataset}: {issues}')
        rid,_=run_id(j,PROFILE,PROTOCOL)
        ids=[str(d.get('document_id') or d.get('id')) for d in filtered_docs(j)]
        expected=int(expected_map[j.dataset])
        if len(ids)!=expected or len(set(ids))!=len(ids):
            raise RuntimeError(f'Invalid ID population in {j.method}/{j.dataset}: '
                               f'{len(ids)} vs {expected}')
        status=read_records(db,rid)
        done=sum(status.get(i)=='done' for i in ids)
        failed=sum(status.get(i)=='failed' for i in ids)
        tasks.append({'job':j,'rid':rid,'ids':ids,'expected':expected,
                      'run_root':root/PROFILE/PHASE/j.method/j.dataset/rid})
        print(f'{j.method:21} {j.dataset:18} {done:4}/{expected:4} done '
              f'failed={failed:3} run_id={rid}',flush=True)
    if not tasks:
        raise RuntimeError('No jobs matched your filter')
    print(f'GPT non-CoT: {len(tasks)} selected, '
          f'{sum(t["expected"] for t in tasks)} expected documents.',flush=True)
    if not args.execute:
        print('DRY RUN ONLY. No API calls or writes. Add --execute to resume.',flush=True)
        return 0
    if not os.environ.get('OPENROUTER_API_KEY'):
        raise RuntimeError('OPENROUTER_API_KEY not set. Source the env file.')
    try:
        from tqdm import tqdm  # noqa: F401
    except ImportError as e:
        raise RuntimeError('tqdm is required') from e

    state_path=root/'reports'/'gpt_non_cot_resilient_state.json'
    state=read_state(state_path)
    batches=0
    with campaign_lock(root):
        with Progress(db,tasks,args.progress_interval) as bar:
            for t in tasks:
                j,rid,ids,expected=t['job'],t['rid'],t['ids'],t['expected']
                print(f'\n== {j.method}/{j.dataset} ({expected} documents) ==',flush=True)
                initial=read_records(db,rid)
                manifest_path=t['run_root']/'manifest.json'
                if (sum(initial.get(i)=='done' for i in ids)==expected
                        and manifest_path.is_file()
                        and json.loads(manifest_path.read_text(encoding='utf-8')).get('selected')=='all'):
                    print('SKIP: already complete with full-population manifest; zero work.',flush=True)
                    continue
                while True:
                    status=read_records(db,rid)
                    pending=[i for i in ids if status.get(i) not in ('done','failed')]
                    pending += [i for i in ids if status.get(i)=='failed']
                    if not pending:
                        break
                    if batches>=args.max_batches:
                        raise RuntimeError('Maximum batch count reached; checkpoints preserved')
                    if state.get('outage_streak',0)>=args.max_overloads:
                        raise RuntimeError('Too many consecutive upstream overloads; '
                                           'checkpoints preserved, safe stop')
                    remaining=float(state.get('next_retry_after',0))-time.time()
                    if remaining>0:
                        sleep_seconds(remaining,'persisted upstream outage')
                    probe=bool(state.get('probing',False))
                    subset=pending[:1 if probe else args.batch_size]
                    workers=1 if probe else min(args.workers,len(subset))
                    session=uuid.uuid4().hex
                    before=sum(status.get(i)=='done' for i in ids)
                    batches+=1
                    print(f'Batch {batches}: {j.method}/{j.dataset} '
                          f'pre={before}/{expected}; select={len(subset)}, '
                          f'workers={workers}, probe={probe}',flush=True)
                    result=execute_job(j,profile_name=PROFILE,mode='parallel',
                                       workers=workers,max_errors=0,
                                       session_id=session,selected_ids=subset,
                                       retry_failed=True,allow_unverified=False,
                                       max_shard_seconds=86400,results_root=root,
                                       protocol_mode=PROTOCOL)
                    bar.update()
                    failures=read_failures(db,session,rid)
                    categories=Counter(failure_category(x[2]) for x in failures)
                    current=read_records(db,rid)
                    after=sum(current.get(i)=='done' for i in ids)
                    in_batch=sum(current.get(i)=='done' for i in subset)
                    print(f'  result={result.get("status")}; now={after}/{expected}; '
                          f'batch_done={in_batch}/{len(subset)}, '
                          f'new_done={after-before}, errors={dict(categories)}',flush=True)
                    if result.get('run_id') not in (None,rid):
                        raise RuntimeError('Unexpected run identity; stopping safely')
                    if any(x!='UPSTREAM_OVERLOAD_429' for x in categories):
                        raise RuntimeError('Non-transient/unknown failure; '
                                           f'inspect saved session {session}')
                    if result.get('status') in ('blocked','orchestration_error','failed'):
                        raise RuntimeError(f'Unexpected job status {result}; safe stop')
                    if not failures and in_batch < len(subset):
                        raise RuntimeError('Missing checkpoints without a recorded 429; '
                                           f'inspect saved session {session}')
                    if failures:
                        n=int(state.get('outage_streak',0))+1
                        delay=min(args.max_backoff,
                                  args.initial_backoff*2**min(n-1,16))
                        state.update({'outage_streak':n,'probing':True,
                                      'next_retry_after':time.time()+delay,
                                      'last_session':session,'last_job':j.identity})
                        save_state(state_path,state)
                        print(f'  Provider overload: circuit opened. '
                              f'Next recovery probe after {delay}s.',flush=True)
                        continue
                    if after<=before:
                        raise RuntimeError('No forward progress in a healthy batch')
                    state.update({'outage_streak':0,'probing':False,
                                  'next_retry_after':0,'last_session':session,
                                  'last_job':j.identity})
                    save_state(state_path,state)
                    if args.healthy_pause:
                        sleep_seconds(args.healthy_pause,'healthy pacing')

                # Validate full exact population before marking job complete.
                status=read_records(db,rid)
                if any(status.get(i)!='done' for i in ids):
                    raise RuntimeError(f'Invalid completed population: {j.method}/{j.dataset}')
                for doc in ids:
                    path=record_path(t['run_root'],doc)
                    if not path.is_file():
                        raise RuntimeError(f'SQLite done but file missing: {path}')
                    row=json.loads(path.read_text(encoding='utf-8'))
                    if row.get('run_id')!=rid or row.get('document_id')!=doc:
                        raise RuntimeError(f'Unexpected checkpoint payload: {doc}')
                print(f'FINALIZE {j.method}/{j.dataset}: verify full-population manifest.',flush=True)
                # The batch runner may have left manifest.selected=N; a full
                # sequential pass should skip all done records and write selected=all.
                final_session=uuid.uuid4().hex
                completed=execute_job(j,profile_name=PROFILE,mode='sequential',
                                      workers=1,max_errors=0,session_id=final_session,
                                      selected_ids=None,retry_failed=False,
                                      allow_unverified=False,max_shard_seconds=86400,
                                      results_root=root,protocol_mode=PROTOCOL)
                if completed.get('run_id')!=rid or completed.get('status')!='completed':
                    raise RuntimeError(f'Full-population finalization failed: {completed}')
                if read_failures(db,final_session,rid):
                    raise RuntimeError('Finalization unexpectedly attempted a failing call')
                manifest=json.loads((t['run_root']/'manifest.json').read_text(encoding='utf-8'))
                if manifest.get('selected')!='all':
                    raise RuntimeError('Final manifest does not specify full population')
                bar.update()
                print(f'COMPLETE {j.method}/{j.dataset} {expected}/{expected}',flush=True)
    print('\nALL REQUESTED GPT NON-COT METHODS COMPLETE',flush=True)
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nInterrupted; completed checkpoints remain intact.',file=sys.stderr)
        raise SystemExit(130)
    except Exception as ex:
        print(f'REFUSED/SAFE STOP: {ex}',file=sys.stderr)
        raise SystemExit(2)
