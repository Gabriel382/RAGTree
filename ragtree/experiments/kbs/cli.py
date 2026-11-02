"""KBS Journal reproducible experiment CLI. No remote calls for plan/status/report."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time
import uuid
from .engine import OUT, execute_job, campaign_lock, filtered_docs, preflight, run_id, summarize_campaign
from .protocol import job_matrix, select_jobs, load_protocol
from .runtime import init_db, status_counts, atomic_json
from .estimate import sample_ids, estimate_run, projected_campaign
from .report import report_one, report_campaign


def build_parser():
    parser=argparse.ArgumentParser(prog='kbs_protocol',description='KBS reproducible three-protocol campaign')
    sub=parser.add_subparsers(dest='command',required=True)
    for op in ('plan','smoke','estimate','run','resume','status','report'):
        p=sub.add_parser(op)
        p.add_argument('--profile',choices=['local','openrouter'],default='local')
        p.add_argument('--phase',default=None,help='Filter phase; omit for entire campaign')
        p.add_argument('--method',default=None)
        p.add_argument('--dataset',default=None)
        p.add_argument('--model',default=None)
        p.add_argument('--max-jobs',type=int,default=None)
        p.add_argument('--results-root',type=Path,default=OUT)
        if op in ('smoke','estimate','run','resume'):
            p.add_argument('--mode',choices=['sequential','parallel'],default=None)
            p.add_argument('--workers',type=int,default=None)
            p.add_argument('--max-errors',type=int,default=3)
            p.add_argument('--allow-unverified',action='store_true',help='Run inferred/unverified historical recipes (explicit opt-in)')
            p.add_argument('--retry-failed',action='store_true')
            p.add_argument('--allow-count-mismatch',action='store_true',help='Explicitly permit dataset record totals that differ from journal')
            p.add_argument('--protocol',choices=['legacy_xquality','controlled_revision_v2'],default='legacy_xquality')
            p.add_argument('--max-shard-seconds',type=int,default=86400)
        if op=='smoke':p.add_argument('--documents',type=int,default=1)
        if op=='estimate':
            p.add_argument('--samples-per-cell',type=int,default=6)
            p.add_argument('--sample-cap',type=int,default=30)
            p.add_argument('--target-relative-halfwidth',type=float,default=.20)
            p.add_argument('--no-adaptive',action='store_true')
    return parser


def _jobs(args):
    jobs=job_matrix()
    jobs=select_jobs(jobs,phase=args.phase,method=args.method,dataset=args.dataset,model=args.model,maximum=args.max_jobs)
    if args.command=='smoke' and args.phase is None:
        jobs=[j for j in jobs if j.phase=='main_gpt']
    if args.command=='estimate' and args.phase is None:
        jobs=[j for j in jobs if j.phase=='main_gpt']
    return jobs


def main(argv=None):
    args=build_parser().parse_args(argv)
    root=args.results_root.resolve()
    jobs=_jobs(args)
    if args.command=='plan':
        from collections import Counter
        data={'jobs':len(jobs),'by_phase':dict(Counter(j.phase for j in jobs)),
              'historical_unverified':sum(j.provenance!='reconstructed' for j in jobs),
              'blocked_by_design':sum(j.phase in load_protocol()['blocked_phases'] for j in jobs),
              'items':[{'id':j.identity,'phase':j.phase,'method':j.method,'dataset':j.dataset,
                        'model':j.model,'variant':j.variant,'provenance':j.provenance} for j in jobs]}
        print(json.dumps(data,indent=2))
        return 0
    if args.command=='status':
        db=root/'checkpoints.sqlite'
        if db.is_file():
            counts=status_counts(db)
            with sqlite3.connect(db) as c:
                errs=c.execute('SELECT COUNT(*) FROM failures').fetchone()[0]
        else:counts={};errs=0
        print(json.dumps({'configured_jobs':len(jobs),'record_status':counts,'record_failure_attempts':errs},indent=2))
        return 0
    if args.command=='report':
        rows=[]
        for job in jobs:
            for protocol_mode in ('legacy_xquality','controlled_revision_v2'):
                try: rid,_=run_id(job,args.profile,protocol_mode)
                except (FileNotFoundError,KeyError):rid=None
                base=root/args.profile/job.phase/job.method/job.dataset
                if not base.is_dir():continue
                for rp in sorted(base.iterdir()):
                    if not rp.is_dir() or not (rp/'manifest.json').exists():continue
                    try:
                        manifest=json.loads((rp/'manifest.json').read_text(encoding='utf8'))
                    except (OSError,ValueError):continue
                    if manifest.get('protocol_mode','legacy_xquality')!=protocol_mode:continue
                    if (manifest.get('job') or {}).get('method')!=job.method:continue
                    row=report_one(job,rp)
                    row['protocol_mode']=protocol_mode
                    row['stale_fingerprint']=rp.name!=rid
                    rows.append(row)
        path=report_campaign(rows,root/'reports')
        print(json.dumps({'reported':len(rows),'stale':sum(x['stale_fingerprint'] for x in rows),'directory':str(path)},indent=2))
        return 0
    if args.max_jobs is not None and args.max_jobs<0:raise ValueError('max-jobs must be >=0')
    if args.max_shard_seconds<=0:raise ValueError('max-shard-seconds must be >0')
    if args.command=='smoke' and args.documents<=0:raise ValueError('documents must be >0')
    if args.command=='estimate' and not 1<=args.samples_per_cell<=args.sample_cap<=300:
        raise ValueError('Invalid adaptive pilot sample bounds')
    started=time.monotonic();session=uuid.uuid4().hex
    results=[];estimates=[]
    with campaign_lock(root):
        init_db(root/'checkpoints.sqlite')
        for index,job in enumerate(jobs,1):
            issue=preflight(job)
            if issue['issues'] or (job.provenance!='reconstructed' and not args.allow_unverified):
                reasons=list(issue['issues'])
                if job.provenance!='reconstructed' and not args.allow_unverified:reasons.append('historical recipe requires --allow-unverified')
                item={'job':job.identity,'status':'blocked','reasons':reasons,'method':job.method,'dataset':job.dataset,'phase':job.phase}
                results.append(item)
                print(f'[{index}/{len(jobs)}] BLOCKED {job.method}/{job.dataset}: {reasons}',flush=True)
                continue
            try:
                population=sum(1 for _ in filtered_docs(job))
                if population==0:
                    results.append({'job':job.identity,'status':'blocked','reasons':['no evaluation records']})
                    continue
                expected=load_protocol()['expected_records'][job.dataset]
                if population!=expected and args.command in ('run','resume') and not args.allow_count_mismatch:
                    results.append({'job':job.identity,'status':'blocked','method':job.method,
                                    'dataset':job.dataset,'reasons':[f'evaluation records {population} != journal {expected}; --allow-count-mismatch for exploratory work']})
                    print(f'[{index}/{len(jobs)}] BLOCKED: record mismatch {population} != {expected}',flush=True)
                    continue
                selected=None
                if args.command=='smoke':selected=sample_ids(job,min(args.documents,population))
                elif args.command=='estimate':selected=sample_ids(job,min(args.samples_per_cell,population))
                item=execute_job(job,profile_name=args.profile,mode=args.mode,workers=args.workers,
                    max_errors=args.max_errors,session_id=session,selected_ids=selected,
                    retry_failed=args.retry_failed,allow_unverified=args.allow_unverified,
                    max_shard_seconds=args.max_shard_seconds,results_root=root,
                    protocol_mode=args.protocol)
                results.append(item)
                print(f'[{index}/{len(jobs)}] {job.phase} {job.method}/{job.dataset}: {item["status"]}, {item.get("counts")}',flush=True)
                if item['status'] in ('aborted','failed'):
                    break
                if args.command=='estimate' and item.get('run_root'):
                    current=estimate_run(job,Path(item['run_root']),population)
                    if not args.no_adaptive:
                        n=len(selected)
                        while n<min(args.sample_cap,population):
                            # Conservative one-sided bootstrap, on runtime; dollars may be unavailable locally.
                            d=current['runtime_sec'];avg=d['projected_total'];hi=d['projected_upper_97_5']
                            if avg is not None and avg>0 and hi is not None and (hi/avg-1)<=args.target_relative_halfwidth:
                                break
                            n=min(n+6,args.sample_cap,population)
                            selected=sample_ids(job,n)
                            item=execute_job(job,profile_name=args.profile,mode=args.mode,workers=args.workers,
                                max_errors=args.max_errors,session_id=session,selected_ids=selected,
                                retry_failed=args.retry_failed,allow_unverified=args.allow_unverified,
                                max_shard_seconds=args.max_shard_seconds,results_root=root,
                                protocol_mode=args.protocol)
                            results.append(item)
                            if item['status'] in ('aborted','failed'):break
                            current=estimate_run(job,Path(item['run_root']),population)
                        if item['status'] in ('aborted','failed'):break
                    estimates.append(current)
                    atomic_json(Path(item['run_root'])/'estimate.json',current)
            except Exception as exc:
                results.append({'job':job.identity,'method':job.method,'dataset':job.dataset,
                        'status':'orchestration_error','reason':repr(exc)})
                print(f'[{index}/{len(jobs)}] ORCHESTRATION ERROR: {exc}',file=sys.stderr,flush=True)
                break
            with sqlite3.connect(root/'checkpoints.sqlite',timeout=30) as c:
                count=c.execute('SELECT COUNT(*) FROM failures WHERE session_id=?',(session,)).fetchone()[0]
            if count>args.max_errors:
                print(f'ABORT: terminal error limit exceeded: {count}>{args.max_errors}',flush=True)
                break
        report=summarize_campaign(results,root,session,time.monotonic()-started)
        if estimates:
            atomic_json(root/'reports'/f'estimate_{session}.json',estimates)
            atomic_json(root/'reports'/f'projected_campaign_{session}.json',projected_campaign(estimates))
        print(json.dumps({'report':str(report),'job_count':len(results),
                          'statuses':{x['status']:sum(i['status']==x['status'] for i in results) for x in results}},indent=2))
        return 1 if any(r['status'] in ('aborted','failed','orchestration_error') for r in results) else (2 if any(r['status']=='blocked' for r in results) else 0)

if __name__=='__main__':raise SystemExit(main())
