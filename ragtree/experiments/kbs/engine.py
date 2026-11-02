"""Bounded orchestration of existing xquality CLI scripts (no algorithm rewrites)."""
from __future__ import annotations
import contextlib
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from typing import Any
from .protocol import Job, ROOT, load_protocol, output_config, profile_config
from .runtime import init_db, atomic_json, status_counts

OUT=ROOT/'results/kbs_journal'

SCRIPT_MAP={
 'baseline':'run_single_llm_baseline.py',
 'icl':'run_icl_baseline.py',
 'cot':'run_cot_baseline.py',
 'growlrag':'run_growlrag_relations.py',
 'ograg':'run_ograg_relations.py',
 'chunk_orag':'run_chunk_orag_relations.py',
 'kg_rag':'run_kg_rag_relations.py',
 'community_kgrag':'run_community_kgrag_relations.py',
 'triple_kg_rag':'run_triple_kg_rag_relations.py',
 'agentic_hybrid':'run_agentic_hybrid_relations.py',
 'langgraph_agentic_simple':'run_langgraph_agentic_simple_relations.py',
 'marag':'run_marag_relations.py'
}

@contextlib.contextmanager
def campaign_lock(root:Path):
    root.mkdir(parents=True,exist_ok=True)
    path=root/'.campaign.lock'
    with path.open('a+b') as f:
        try:
            if os.name=='nt':
                import msvcrt
                f.seek(0);f.write(b'0');f.flush();f.seek(0)
                msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as e:
            raise RuntimeError('Another KBS campaign is already running in this results directory') from e
        try:yield
        finally:
            if os.name=='nt':
                f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f,fcntl.LOCK_UN)


def _arg(args,key,default=None):
    flag='--'+key.replace('_','-')
    if flag not in args:return default
    k=list(args).index(flag)
    return args[k+1] if k+1<len(args) else None


def input_path(job:Job) -> Path:
    import yaml
    cfg=yaml.safe_load((ROOT/'configs/default.yaml').read_text(encoding='utf8'))
    dataset_key=_arg(job.arguments,'dataset_key')
    keypath=cfg.get('datasets',{}).get('preprocessed',{}).get(dataset_key)
    if keypath is None:
        keypath=str(Path(cfg['paths']['data_preprocessed'])/(dataset_key+'.jsonl'))
    return (ROOT/keypath).resolve()


def doc_type(job:Job):
    return _arg(job.arguments,'doc_types',_arg(job.arguments,'doc_type_filter',_arg(job.arguments,'doc_type',_arg(job.arguments,'icl_predict_types','all'))))


def filtered_docs(job:Job):
    source=input_path(job)
    types=doc_type(job)
    accepted=None if not types or types=='all' else set(types.split(','))
    seen=set()
    with source.open(encoding='utf8') as f:
        for n,line in enumerate(f,start=1):
            if not line.strip():continue
            obj=json.loads(line)
            identifier=obj.get('document_id') or obj.get('id')
            if identifier is None:raise ValueError(f'{source}:{n} missing document ID')
            identifier=str(identifier)
            if identifier in seen:raise ValueError(f'{source}:{n} duplicate ID {identifier!r}')
            seen.add(identifier)
            if accepted and obj.get('type') not in accepted:continue
            yield obj


def preflight(job:Job):
    issues=[];warnings=[]
    path=input_path(job)
    if not path.is_file():issues.append('missing dataset: '+str(path))
    script=ROOT/'scripts'/SCRIPT_MAP[job.method]
    if not script.is_file():issues.append('missing script: '+str(script))
    resource_flags={'--kg-path','--ontology-links-path','--index-dir','--communitykg-root'}
    args=list(job.arguments)
    for i,v in enumerate(args[:-1]):
        if v not in resource_flags:continue
        p=(ROOT/args[i+1]).resolve()
        if not p.exists():issues.append(f'missing resource {v}: {p}')
    if job.provenance!='reconstructed':warnings.append('historical provenance: '+job.provenance+' ('+job.source+')')
    if job.phase=='budget_control':
        from .context_budget import EVIDENCE_HEADERS
        if job.method not in EVIDENCE_HEADERS:issues.append('unverified evidence prompt boundaries for this method')
    # All method-specific resources are not exhaustively discoverable in CLI args.
    # Runner initialization is the final mandatory preflight for lazy resources.
    return {'job':job.identity,'issues':issues,'warnings':warnings,'input':str(path)}


def _sha_file(path:Path):
    h=sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def run_id(job:Job,profile_name:str, protocol_mode:str="legacy_xquality"):
    """Actual input/source bytes are part of checkpoint identity (not only mtime)."""
    path=input_path(job)
    fingerprint={'job':asdict(job),'profile':profile_name,'protocol_mode':protocol_mode,'api_base':profile_config(profile_name).get('api_base'),
       'protocol':_sha_file(ROOT/'configs/experiments/kbs_journal/protocol.yaml'),
       'source':_sha_file(ROOT/'scripts'/SCRIPT_MAP[job.method]),
       'default_yaml':_sha_file(ROOT/'configs/default.yaml')}
    source_hash=sha256()
    for code in sorted((ROOT/'ragtree').rglob('*.py')):
        if '__pycache__' in str(code):continue
        source_hash.update(str(code.relative_to(ROOT)).encode());source_hash.update(_sha_file(code).encode())
    fingerprint['library_code_sha256']=source_hash.hexdigest()
    if path.is_file():fingerprint['dataset_sha256']=_sha_file(path)
    for index,flag in enumerate(job.arguments[:-1]):
        if flag in ('--kg-path','--ontology-links-path','--index-dir','--communitykg-root'):
            p=ROOT/job.arguments[index+1]
            if p.is_file():fingerprint[flag]=_sha_file(p)
            elif p.is_dir():
                entries=[]
                for q in sorted(p.rglob('*')):
                    if q.is_file():
                        stats=q.stat();entries.append((str(q.relative_to(p)),stats.st_size,stats.st_mtime_ns))
                fingerprint[flag]={'directory_metadata':entries}
    raw=json.dumps(fingerprint,sort_keys=True).encode()
    return sha256(raw).hexdigest()[:24],fingerprint


def _prep_command(job,conf,profile):
    args=list(job.arguments)
    # historical recipe's dataset/resources are preserved; backend explicitly overridden.
    for flag in ('--backend','--model','--config'):
        if flag in args:
            j=args.index(flag);del args[j:j+2]
    args+=['--backend','litellm','--model',profile['models'][job.model],'--config',str(conf)]
    return [sys.executable,str(ROOT/'scripts'/SCRIPT_MAP[job.method]),*args]


def _session_errors(db,session):
    with sqlite3.connect(db,timeout=30) as c:
        return c.execute('SELECT COUNT(*) FROM failures WHERE session_id=?',(session,)).fetchone()[0]


def _process_error(db,session,run_id,identifier,detail):
    with sqlite3.connect(db,timeout=30) as c:
        c.execute('INSERT OR REPLACE INTO failures VALUES (?,?,?,?,?)',(session,run_id,identifier,detail,time.time()))


def _terminate(processes):
    for proc,log,index in processes:
        if proc.poll() is None:proc.terminate()
    deadline=time.monotonic()+5
    for proc,log,index in processes:
        if proc.poll() is None:
            try:proc.wait(timeout=max(.1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=3)
        log.close()


def execute_job(job:Job,profile_name='local',mode=None,workers=None,max_errors=3,
                session_id=None,selected_ids:list[str]|None=None,retry_failed=False,
                allow_unverified=False,max_shard_seconds=86400,results_root=OUT,
                protocol_mode="legacy_xquality"):
    pf=preflight(job)
    if pf['issues']:
        return {'job':job.identity,'status':'blocked','reasons':pf['issues'],'warnings':pf['warnings']}
    if job.provenance in ('unverified','unverified_mapping','inferred') and not allow_unverified:
        return {'job':job.identity,'status':'blocked','reasons':['unverified source recipe; --allow-unverified required'], 'warnings':pf['warnings']}
    if protocol_mode not in ("legacy_xquality","controlled_revision_v2"):
        raise ValueError("unknown protocol mode")
    profile=profile_config(profile_name)
    mode=mode or profile['mode']
    if mode not in ('sequential','parallel'):raise ValueError(mode)
    workers=int(workers if workers is not None else (1 if mode=='sequential' else profile['workers']))
    if mode=='sequential':workers=1
    if not 1<=workers<=64:raise ValueError('workers must be 1..64')
    if not 0<=max_errors<=9999:raise ValueError('max_errors must be nonnegative')
    session_id=session_id or uuid.uuid4().hex
    rid,fingerprint=run_id(job,profile_name,protocol_mode)
    root=Path(results_root)/profile_name/job.phase/job.method/job.dataset/rid
    root.mkdir(parents=True,exist_ok=True)
    db=Path(results_root)/'checkpoints.sqlite'
    init_db(db)
    selected_file=None
    if selected_ids is not None:
        selected_file=root/f'selected_{session_id}.json'
        atomic_json(selected_file,selected_ids)
    manifest={'job':asdict(job),'profile':profile_name,'protocol_mode':protocol_mode,'run_id':rid,'fingerprints':fingerprint,
              'command_shards':[],'selected':len(selected_ids) if selected_ids is not None else 'all',
              'workers':workers,'created':time.time()}
    # Controlled schema is fixed per dataset; separate from per-target gold labels.
    eval_ids=[]
    schema_labels=set()
    if protocol_mode=='controlled_revision_v2':
        import yaml
        all_docs=list(filtered_docs(job))
        eval_ids=[str(d.get('document_id') or d.get('id')) for d in all_docs]
        for d in all_docs:
            if isinstance(d.get('relations'),dict):schema_labels.update(d['relations'])
        if job.dataset=='docred_causal':
            cfg=yaml.safe_load((ROOT/'configs/default.yaml').read_text(encoding='utf8'))
            info_file=ROOT/cfg['datasets']['raw']['DocRED']/'rel_info.json'
            if not info_file.is_file():
                return {'job':job.identity,'status':'blocked','reasons':['DocRED full relation schema missing: '+str(info_file)]}
            info=json.loads(info_file.read_text(encoding='utf8'))
            schema_labels={f'{key} : {value}' for key,value in info.items()}
        if not schema_labels:
            return {'job':job.identity,'status':'blocked','reasons':['Cannot derive nonempty dataset-wide schema']}
        atomic_json(root/'schema.json',sorted(schema_labels))
        atomic_json(root/'eval_ids.json',eval_ids)
    processes=[]
    try:
        for index in range(workers):
            path=output_config(job,root,profile,index)
            command=_prep_command(job,path,profile)
            log_path=root/'logs'/f'shard_{index}.log';log_path.parent.mkdir(parents=True,exist_ok=True)
            log=log_path.open('a',encoding='utf8')
            env=os.environ.copy()
            env.update({'KBS_ACTIVE':'1','KBS_DB':str(db.resolve()),'KBS_RUN_ID':rid,
               'KBS_RUN_ROOT':str(root.resolve()),'KBS_SHARDS':str(workers),'KBS_SHARD':str(index),
               'KBS_SESSION_ID':session_id,'KBS_MAX_ERRORS':str(max_errors),
               'KBS_RETRY_FAILED':str(int(retry_failed)),
               'KBS_PROTOCOL_MODE':protocol_mode,
               'KBS_METHOD':job.method,
               'KBS_API_RETRIES':str(load_protocol()['max_api_retries']),
               'PYTHONPATH':str(ROOT)+os.pathsep+env.get('PYTHONPATH','')})
            if job.phase=='budget_control':env['KBS_EVIDENCE_BUDGET_TOKENS']='2048'
            if selected_file:env['KBS_SELECTED_IDS']=str(selected_file.resolve())
            if protocol_mode=='controlled_revision_v2':
                env['KBS_SCHEMA_FILE']=str((root/'schema.json').resolve())
                env['KBS_EVAL_IDS_FILE']=str((root/'eval_ids.json').resolve())
            if profile.get('api_base'):env['KBS_LITELLM_API_BASE']=profile['api_base']
            api_env=profile.get('api_key_env')
            if api_env and os.environ.get(api_env):env['KBS_LITELLM_API_KEY']=os.environ[api_env]
            elif profile_name=='local':env['KBS_LITELLM_API_KEY']='dummy'
            manifest['command_shards'].append({'index':index,'args':command,'log':str(log_path)})
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            processes.append((proc,log,index))
        atomic_json(root/'manifest.json',manifest)
        start=time.monotonic()
        reason=None
        while any(proc.poll() is None for proc,_,_ in processes):
            errors=_session_errors(db,session_id)
            if errors>max_errors:
                reason=f'terminal errors {errors} > {max_errors}'
                break
            if max_shard_seconds and time.monotonic()-start>max_shard_seconds:
                reason=f'job timeout after {max_shard_seconds}s'
                break
            time.sleep(.25)
        if reason:
            _terminate(processes)
            status='aborted'
        else:
            codes=[]
            for proc,log,index in processes:
                codes.append(proc.wait());log.close()
            if any(code!=0 for code in codes):
                for i,code in enumerate(codes):
                    if code:
                        logtail=(root/'logs'/f'shard_{i}.log').read_text(errors='replace')[-4000:]
                        _process_error(db,session_id,rid,f'__shard_{i}',f'exit={code}: {logtail}')
                status='failed'
                reason='one or more subprocesses exited non-zero'
            else:status='completed'
        counts=status_counts(db,[rid])
        # A previous job's failure must not mark this job as failed, and a
        # previously failed/skipped document must not make smoke report success.
        expected_ids = set(selected_ids) if selected_ids is not None else {
            str(doc.get('document_id') or doc.get('id')) for doc in filtered_docs(job)
        }
        with sqlite3.connect(db, timeout=30) as c:
            state = dict(c.execute('SELECT doc_id,status FROM records WHERE run_id=?',(rid,)))
        selected_done = sum(state.get(identifier)=='done' for identifier in expected_ids)
        selected_failed = sum(state.get(identifier)=='failed' for identifier in expected_ids)
        selected_missing = len(expected_ids)-selected_done-selected_failed
        if status=='completed' and (selected_failed or selected_missing):
            status='completed_with_failures'
        result={'job':job.identity,'run_id':rid,'run_root':str(root),'status':status,
                'counts':counts,'selected_record_status':{
                    'expected':len(expected_ids),'done':selected_done,
                    'failed':selected_failed,'missing':selected_missing},
                'session_errors':_session_errors(db,session_id),'reason':reason,
                'warnings':pf['warnings']}
        atomic_json(root/'last_execution.json',result)
        return result
    except BaseException:
        _terminate(processes)
        raise


def summarize_campaign(results,root:Path,session_id:str,elapsed:float):
    counts={}
    for item in results:counts[item['status']]=counts.get(item['status'],0)+1
    summary={'session_id':session_id,'elapsed_seconds':elapsed,'experiment_jobs':len(results),
             'by_status':counts,'results':results}
    path=root/'reports'/f'campaign_{session_id}.json'
    atomic_json(path,summary)
    import csv
    csv_path=path.with_suffix('.csv')
    with csv_path.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=['job','status','run_id','run_root','reason','session_errors'])
        writer.writeheader()
        for r in results:writer.writerow({k:r.get(k) for k in writer.fieldnames})
    return path
