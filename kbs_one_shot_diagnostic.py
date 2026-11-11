#!/usr/bin/env python3
"""ONE-SHOT, OFFLINE KBS audit of a completed experiment cell.

No imports from RAGTree, no OpenRouter requests, no model inference, no edits to
checkpoint SQLite, original predictions, protocol, source, or run ID.

Usage:
  python kbs_one_shot_diagnostic.py --repo . --run-id bbb952c60885a7dc78722d99

Writes an independently computed scientific audit in a NEW diagnostics folder.
Only existing evidence is used; unobservable stage-level claims are marked unknown.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import random
import sqlite3
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from urllib.parse import quote

DEFAULT_RUN = 'bbb952c60885a7dc78722d99'
METHOD = 'baseline'
DATASET = 'maven_ere'
PHASE = 'main_gpt'
PROFILE = 'openrouter'
MODEL_KEY = 'gpt_oss_20b'
VERSION = '1.0'


def read_json(p, errors=None):
    try:
        with p.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as exc:
        if errors is not None:
            errors.append({'path':str(p),'error':f'{type(exc).__name__}: {exc}'})
        return None


def sha256(p):
    if not p.is_file():return None
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def digest_id(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:24]


def triples(value):
    """Matching RAGTree's relations_to_triples semantics; ignore 'null'."""
    result=set()
    issues=Counter()
    if not isinstance(value,dict):
        issues['not_a_relation_dict']+=1
        return result,issues
    for lab,pairs in value.items():
        if lab=='null':
            issues['null_labels_ignored']+=len(pairs) if isinstance(pairs,list) else 1
            continue
        if not isinstance(pairs,list):
            issues['non_list_relation_value']+=1
            continue
        for pair in pairs:
            if not (isinstance(pair,(list,tuple)) and len(pair)==2 and
                    isinstance(pair[0],str) and isinstance(pair[1],str)):
                issues['invalid_relation_pair']+=1
                continue
            q=(lab,pair[0],pair[1])
            if q in result:issues['duplicate_relation_pairs']+=1
            result.add(q)
    return result,issues


def prf(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0.
    r=tp/(tp+fn) if tp+fn else 0.
    f=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    return {'precision':p,'recall':r,'f1':f}


def pctile(xs,p):
    if not xs:return None
    a=sorted(xs)
    if len(a)==1:return a[0]
    i=(len(a)-1)*p
    lo=math.floor(i); hi=math.ceil(i)
    return a[lo]*(hi-i)+a[hi]*(i-lo) if hi!=lo else a[lo]


def stats(xs):
    if not xs:return {'n':0,'total':None,'mean':None,'median':None,'p90':None,'p99':None,'max':None}
    return {'n':len(xs),'total':sum(xs),'mean':statistics.mean(xs), 'median':statistics.median(xs),
            'p90':pctile(xs,.9),'p99':pctile(xs,.99),'max':max(xs)}


def compare_gold_pred(gold,pred):
    hit=gold & pred
    fp=pred-gold
    fn=gold-pred
    pred_pair={(h,t) for _,h,t in pred}
    gold_pair={(h,t) for _,h,t in gold}
    wrong_label=sum(1 for rel,h,t in fp if (h,t) in gold_pair)
    reversed_pair=sum(1 for rel,h,t in fp if (t,h) in gold_pair)
    wrong_label_or_reversed=sum(1 for rel,h,t in fp if (h,t) in gold_pair or (t,h) in gold_pair)
    fn_pair_recovered=sum(1 for rel,h,t in fn if (h,t) in pred_pair)
    return {
        'tp':len(hit),'fp':len(fp),'fn':len(fn),
        'pred_with_gold_endpoint_pair_wrong_label':wrong_label,
        'pred_with_reversed_gold_endpoint_pair':reversed_pair,
        'pred_with_gold_pair_wrong_label_or_direction':wrong_label_or_reversed,
        'missed_gold_with_predicted_endpoint_pair_other_label':fn_pair_recovered,
        'pair_only_tp':len(pred_pair&gold_pair),
        'pair_only_fp':len(pred_pair-gold_pair),
        'pair_only_fn':len(gold_pair-pred_pair)
    }


def doc_bootstrap_ci(rows, replicates=1000, seed=42):
    """Document-resampling 95% micro-F1 CI; does NOT measure run-to-run variation."""
    if not rows:return None
    rng=random.Random(seed)
    counts=[(r['tp'],r['fp'],r['fn']) for r in rows]
    n=len(counts)
    f1=[]
    for _ in range(replicates):
        tp=fp=fn=0
        for _ in range(n):
            x,y,z=counts[rng.randrange(n)]; tp+=x;fp+=y;fn+=z
        f1.append(prf(tp,fp,fn)['f1'])
    return {'lower':pctile(f1,.025),'upper':pctile(f1,.975),'replicates':replicates,
            'seed':seed,'scope':'Document-sampling uncertainty within one run; NOT LLM stochastic robustness.'}


def endpoint_ids(doc):
    # Not every dataset uses entities, and some event datasets use events.
    for key in ('entities','events'):
        v=doc.get(key)
        if isinstance(v,dict) and v:
            return set(map(str,v.keys())),key
    return None,None


def cross_sentence(gold,doc):
    # Count only gold endpoint pairs for which BOTH mention sentence sets are known.
    entities=doc.get('entities',{})
    if not isinstance(entities,dict):return (0,0)
    def get_sents(v):
        if not isinstance(v,dict):return set()
        ms=v.get('mentions',[])
        if not isinstance(ms,list):return set()
        return {str(m['sent_id']) for m in ms if isinstance(m,dict) and m.get('sent_id') is not None}
    known=between=0
    for _,head,tail in gold:
        hs=get_sents(entities.get(head));ts=get_sents(entities.get(tail))
        if hs and ts:
            known+=1
            if not(hs&ts):between+=1
    return known,between


def csv_write(path,rows,fields):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
        for r in rows:
            w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})


def _number(v):
    try:
        if isinstance(v,bool) or v is None:return None
        x=float(v)
        return x if math.isfinite(x) else None
    except (ValueError,TypeError,OverflowError):return None


def response_contents(resp):
    choices=resp.get('choices') or []
    if not isinstance(choices,list) or not choices:return None,None
    choice=choices[0] if isinstance(choices[0],dict) else {}
    msg=choice.get('message') if isinstance(choice.get('message'),dict) else {}
    content=msg.get('content')
    if isinstance(content,str):return content, choice.get('finish_reason')
    if isinstance(content,list):
        return '\n'.join(item.get('text','') for item in content if isinstance(item,dict) and isinstance(item.get('text'),str)),choice.get('finish_reason')
    return None,choice.get('finish_reason')


def syntax_check(content):
    if content is None or not content.strip():return 'empty'
    t=content.strip()
    if t.startswith('```'):
        t=re.sub(r'^```(?:json)?\s*','',t,flags=re.I)
        t=re.sub(r'\s*```$','',t)
    try:
        q=json.loads(t)
        return 'json_object' if isinstance(q,dict) else 'json_other'
    except (ValueError,TypeError):return 'not_strict_json'


def _extract_request_params(req):
    params=req.get('parameters') or {}
    if not isinstance(params,dict):return {}
    return params


def safe_ratecard(repo):
    try:
        import yaml # optional local dependency, not required
        p=yaml.safe_load((repo/'configs/experiments/kbs_journal/protocol.yaml').read_text(encoding='utf8'))
        return (p or {}).get('price_per_million_usd',{}).get(MODEL_KEY,{})
    except Exception:return {}


def git_commit(repo):
    try:
        p=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,timeout=5,check=True)
        return p.stdout.strip()
    except Exception:return None


def extract_failure_bucket(detail):
    if 'engine_overloaded' in detail or ('429' in detail and 'RateLimit' in detail):return 'upstream_429'
    if 'no usable final-answer' in detail or 'empty content' in detail:return 'empty_llm_final'
    if 'CampaignAbort' in detail or '__shard_' in detail:return 'orchestration_abort'
    if 'Timeout' in detail or 'timed out' in detail:return 'timeout'
    if 'JSONDecodeError' in detail:return 'json_decode_exception'
    return 'other'


def table_md(rows,columns):
    if not rows:return '_No records available._\n'
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join('---' for _ in columns)+' |']
    for row in rows:
        lines.append('| '+' | '.join(str(row.get(x,'')) for x in columns)+' |')
    return '\n'.join(lines)+'\n'


def human(v,d=3):
    return 'unavailable' if v is None else f'{v:,.{d}f}' if isinstance(v,float) else str(v)


def main():
    p=argparse.ArgumentParser(description='Single offline, read-only KBS scientific diagnostic; zero API calls')
    p.add_argument('--repo',type=Path,default=Path('.'))
    p.add_argument('--run-id',default=DEFAULT_RUN)
    p.add_argument('--expected',type=int,default=3516)
    p.add_argument('--out',type=Path,default=None,help='New output directory, not inside the run')
    args=p.parse_args()
    repo=args.repo.expanduser().resolve()
    root=repo/'results/kbs_journal'
    run=root/PROFILE/PHASE/METHOD/DATASET/args.run_id
    if not run.is_dir():p.error(f'Completed run not found: {run}')
    out=(args.out or (root/'diagnostics'/('one_shot_'+args.run_id))).expanduser().resolve()
    if out==run or run in out.parents:p.error('Output cannot be inside experiment run')
    if out.exists() and any(out.iterdir()):p.error(f'Output directory already contains files: {out}; choose --out')
    out.mkdir(parents=True,exist_ok=True)
    errors=[];warnings=[]
    manifest=read_json(run/'manifest.json',errors) or {}
    archived=read_json(run/'metrics.json',errors) or {}
    if manifest.get('run_id')!=args.run_id: warnings.append('Manifest run_id does not match the requested run_id')
    if manifest.get('protocol_mode')!='controlled_revision_v2':warnings.append('Manifest is not controlled_revision_v2')
    if manifest.get('selected')!='all':warnings.append(f'Manifest selected={manifest.get("selected")}; this is not necessarily the full population')
    records={}
    duplicate_files=[]
    for path in sorted((run/'records').glob('*.json')):
        rec=read_json(path,errors)
        if not isinstance(rec,dict):continue
        did=rec.get('document_id')
        if not isinstance(did,str):errors.append({'path':str(path),'error':'missing document_id'});continue
        if rec.get('run_id')!=args.run_id:warnings.append('Record run_id mismatch: '+did)
        if path.stem!=digest_id(did):warnings.append('Record filename/digest mismatch: '+did)
        if did in records:duplicate_files.append(did)
        records[did]=rec
    db_path=root/'checkpoints.sqlite'; db_rows={}; prior_fail=[]
    if db_path.is_file():
        try:
            with sqlite3.connect('file:'+quote(str(db_path))+'?mode=ro',uri=True,timeout=10) as db:
                db_rows={did:status for did,status in db.execute('SELECT doc_id,status FROM records WHERE run_id=?',(args.run_id,))}
                prior_fail=[(sid,did,detail) for sid,did,detail in db.execute('SELECT session_id,doc_id,detail FROM failures WHERE run_id=?',(args.run_id,))]
        except Exception as exc:errors.append({'path':str(db_path),'error':f'{type(exc).__name__}: {exc}'})
    else:warnings.append('Checkpoint database missing')
    finished={did for did,r in records.items() if r.get('status')=='done'}
    status_done={did for did,s in db_rows.items() if s=='done'}
    db_failed={did for did,s in db_rows.items() if s=='failed'}
    integrity={'expected':args.expected,'saved_record_files':len(records),'completed_record_files':len(finished),
               'sqlite_done':len(status_done),'sqlite_failed':len(db_failed),
               'missing_expected_count':max(0,args.expected-len(finished)),
               'record_done_without_sqlite':sorted(finished-status_done)[:40],
               'sqlite_done_without_record':sorted(status_done-finished)[:40],
               'duplicate_record_ids':duplicate_files[:40],
               'record_source_document_missing':sum(not isinstance(rec.get('source_document'),dict) for rec in records.values())}
    # Exact evaluation-population check, not just matching a numerical row count.
    eval_file=run/'eval_ids.json'
    eval_ids=read_json(eval_file,errors) if eval_file.is_file() else None
    if isinstance(eval_ids,list):
        eval_set=set(map(str,eval_ids))
        integrity['eval_ids_file_count']=len(eval_ids)
        integrity['eval_ids_unique_count']=len(eval_set)
        integrity['eval_ids_missing_from_saved_records']=sorted(eval_set-finished)[:40]
        integrity['saved_records_not_in_eval_ids']=sorted(finished-eval_set)[:40]
        if eval_set!=finished:warnings.append('Saved record IDs do not exactly match eval_ids.json')
        if len(eval_set)!=len(eval_ids):warnings.append('eval_ids.json contains duplicate IDs')
    else:
        integrity['eval_ids_file_count']=None
        warnings.append('No usable eval_ids.json; cannot independently check exact evaluation population IDs')
    # Independent micro, per-class and document-level counts.
    all_doc=[]; per_label=defaultdict(lambda:Counter()); per_gold=Counter();per_pred=Counter()
    total=Counter(); gold_issues=Counter(); pred_issues=Counter()
    cross_known=cross_different=cross_correct=0
    entity_counts=[];doc_text_chars=[];gold_counts=[];pred_counts=[]
    support_bins=defaultdict(lambda:Counter())
    zero_gold=zero_pred=both_zero=0
    incomplete_docs=0
    for did,rec in sorted(records.items()):
        doc=rec.get('source_document') or {}
        if not isinstance(doc,dict):doc={};incomplete_docs+=1
        gold,gi=triples(doc.get('relations'))
        pred,pi=triples(rec.get('pred_relations'))
        gold_issues.update(gi);pred_issues.update(pi)
        a=compare_gold_pred(gold,pred)
        n=len(gold);m=len(pred)
        gold_counts.append(n);pred_counts.append(m)
        per_gold.update(lab for lab,_,_ in gold);per_pred.update(lab for lab,_,_ in pred)
        for rel,h,t in gold&pred:per_label[rel]['tp']+=1
        for rel,h,t in pred-gold:per_label[rel]['fp']+=1
        for rel,h,t in gold-pred:per_label[rel]['fn']+=1
        total.update(a)
        if n==0:zero_gold+=1
        if m==0:zero_pred+=1
        if n==m==0:both_zero+=1
        if m==0 and n:total['fn_in_zero_prediction_docs']+=len(gold)
        k,x=cross_sentence(gold,doc);cross_known+=k;cross_different+=x
        # Cross-sentence recall only on gold pairs with accessible mention locations.
        entities=doc.get('entities',{})
        if isinstance(entities,dict):
            for triple in gold:
                _,h,t=triple
                def ss(v):
                    if not isinstance(v,dict):return set()
                    return {str(y['sent_id']) for y in v.get('mentions',[]) if isinstance(y,dict) and y.get('sent_id') is not None} if isinstance(v.get('mentions',[]),list) else set()
                hs=ss(entities.get(h));ts=ss(entities.get(t))
                if hs and ts and not(hs&ts) and triple in pred:cross_correct+=1
        length=str(doc.get('text') or doc.get('sentences') or '')
        doc_text_chars.append(len(length))
        entities,source=endpoint_ids(doc)
        bad_endpoints=None
        if entities is not None:
            entity_counts.append(len(entities))
            bad_endpoints=sum((h not in entities)+(t not in entities) for _,h,t in pred)
        bin_name=('gold=0' if n==0 else 'gold=1' if n==1 else 'gold=2-4' if n<=4 else 'gold>=5')
        for xkey in ('tp','fp','fn'):support_bins[bin_name][xkey]+=a[xkey]
        support_bins[bin_name]['documents']+=1
        all_doc.append({'document_id':did,'gold_relations':n,'pred_relations':m,'tp':a['tp'],'fp':a['fp'],'fn':a['fn'],
                        'precision':prf(a['tp'],a['fp'],a['fn'])['precision'],
                        'recall':prf(a['tp'],a['fp'],a['fn'])['recall'],
                        'f1':prf(a['tp'],a['fp'],a['fn'])['f1'],
                        'missing_prediction':m==0,'text_chars_proxy':len(length),
                        'pred_invalid_pairs':pi.get('invalid_relation_pair',0),
                        'pred_unknown_endpoints_when_registry_available':bad_endpoints,
                        'entity_registry':source or '',
                        'wrong_label_pair_predictions':a['pred_with_gold_endpoint_pair_wrong_label'],
                        'reversed_gold_pair_predictions':a['pred_with_reversed_gold_endpoint_pair']})
    label_rows=[]
    for label in sorted(per_label):
        c=per_label[label]
        label_rows.append({'relation':label,'tp':c['tp'],'fp':c['fp'],'fn':c['fn'],
                           'gold_support':c['tp']+c['fn'],'pred_support':c['tp']+c['fp'],**prf(c['tp'],c['fp'],c['fn'])})
    micro=prf(total['tp'],total['fp'],total['fn'])
    ci=doc_bootstrap_ci(all_doc)
    macro=statistics.mean(r['f1'] for r in label_rows) if label_rows else 0
    pair_micro=prf(total['pair_only_tp'],total['pair_only_fp'],total['pair_only_fn'])
    # All trace calls, including successful documents that needed extra calls and historical failed calls.
    call_stats=Counter();output_tokens=[];input_tokens=[];reason_tokens=[];latencies=[];raw_chars=[]
    call_records=[];cost_tot=0.0;known_cost=0;unknown_cost=0;provider_counts=Counter()
    response_ids=Counter();finish_counts=Counter();json_syntax_counts=Counter()
    last_json_by_doc={}; last_json_ts_by_doc={}
    schema=read_json(run/'schema.json')
    schema_labels=(set(schema) if isinstance(schema,list) else
                   set(schema.keys()) if isinstance(schema,dict) else set())
    unknown_pred_labels=Counter()
    for rec in records.values():
        p=rec.get('pred_relations')
        if isinstance(p,dict) and schema_labels:
            unknown_pred_labels.update({lab:1 for lab in p.keys() if lab not in schema_labels and lab!='null'})
    raw_empty=0;no_response=0;above_cap=0;traces_doc=Counter()
    generation_caps=Counter();routing_checked=0;routing_violations=0;zero_cost_with_tokens=0
    request_roles=Counter();request_message_counts=[];request_prompt_chars=[];temperature_values=Counter()
    for path in sorted((run/'calls').glob('*/*.json')):
        trace=read_json(path,errors)
        if not isinstance(trace,dict):continue
        did=trace.get('document_id')
        if not isinstance(did,str):did='__unknown__'
        traces_doc[did]+=1
        req=trace.get('request') or {};resp=trace.get('response') or {}
        if not isinstance(req,dict):req={}
        if not isinstance(resp,dict):resp={}
        params=_extract_request_params(req)
        messages=req.get('messages')
        if isinstance(messages,list):
            request_message_counts.append(len(messages))
            request_prompt_chars.append(sum(len(m.get('content','')) for m in messages
                                            if isinstance(m,dict) and isinstance(m.get('content'),str)))
            request_roles.update(str(m.get('role','unknown')) for m in messages if isinstance(m,dict))
        if params.get('temperature') is not None:temperature_values[str(params.get('temperature'))]+=1
        usage=resp.get('usage') or {}
        if not isinstance(usage,dict):usage={}
        details=usage.get('completion_tokens_details') or usage.get('output_tokens_details') or {}
        if not isinstance(details,dict):details={}
        iv=_number(usage.get('prompt_tokens'));ov=_number(usage.get('completion_tokens'))
        rv=_number(details.get('reasoning_tokens'))
        if iv is not None:input_tokens.append(iv)
        if ov is not None:output_tokens.append(ov)
        if rv is not None:reason_tokens.append(rv)
        elapsed=_number(trace.get('elapsed_sec'))
        if elapsed is not None:latencies.append(elapsed)
        content,finish=response_contents(resp)
        valid=syntax_check(content)
        json_syntax_counts[valid]+=1
        if content is not None:raw_chars.append(len(content))
        if valid=='json_object' and did in records:
            try:
                raw_content=content.strip()
                if raw_content.startswith('```'):
                    raw_content=re.sub(r'^```(?:json)?\s*','',raw_content,flags=re.I)
                    raw_content=re.sub(r'\s*```$','',raw_content)
                decoded=json.loads(raw_content)
                ts=_number(trace.get('timestamp')) or 0
                if ts>=last_json_ts_by_doc.get(did,-1):
                    last_json_ts_by_doc[did]=ts
                    last_json_by_doc[did]=decoded
            except Exception:pass
        if valid=='empty':raw_empty+=1
        if not resp:no_response+=1
        if finish is not None:finish_counts[str(finish)]+=1
        cap=_number(params.get('max_completion_tokens') or params.get('max_tokens'))
        if ov is not None and cap is not None and ov>cap:above_cap+=1
        if cap is not None:generation_caps[str(int(cap))]+=1
        provider=(resp.get('provider') or '')
        if not provider:
            body=params.get('extra_body') or {}
            if isinstance(body,dict):
                route=body.get('provider') or {}
                if isinstance(route,dict):
                    pinned=route.get('only')
                    if isinstance(pinned,list):provider='route_only:'+','.join(map(str,pinned))
        provider_counts[str(provider or 'not_recorded')]+=1
        body=params.get('extra_body') or {}
        if isinstance(body,dict) and isinstance(body.get('provider'),dict):
            r=body['provider']
            only=r.get('only')
            if isinstance(only,list):
                routing_checked+=1
                if set(str(x).lower() for x in only)!={'deepinfra'} or r.get('allow_fallbacks') is not False:
                    routing_violations+=1
        if resp.get('provider') and str(resp['provider']).lower()!='deepinfra':
            routing_violations+=1
        if resp.get('id'):response_ids[str(resp['id'])]+=1
        # OpenRouter usage.cost is the best available API-reported cost;
        # LiteLLM hidden cost is a fallback, never added to usage.cost.
        cv=_number(usage.get('cost'))
        cost_source='openrouter_usage.cost'
        if cv is None:
            cv=_number(resp.get('kbs_litellm_reported_cost_usd'));cost_source='litellm_response_cost'
        if cv is None:
            detailcost=usage.get('cost_details') or {}
            if isinstance(detailcost,dict):cv=_number(detailcost.get('upstream_inference_cost'));cost_source='upstream_inference_cost'
        if cv is not None:
            cost_tot+=cv;known_cost+=1
            if cv==0. and (iv or 0)+(ov or 0)>0:zero_cost_with_tokens+=1
        else:unknown_cost+=1;cost_source='unknown'
        error=trace.get('error')
        if error:call_stats['error_trace_calls']+=1
        call_stats['traces']+=1
        call_records.append({'document_id':did,'trace':str(path.relative_to(run)),'session':path.name.split('-')[0],
                             'has_response':bool(resp),'response_id':resp.get('id') or '',
                             'finish_reason':finish or '', 'raw_content_json_status':valid,
                             'prompt_tokens':iv,'completion_tokens':ov,'reasoning_tokens':rv,
                             'max_requested_completion_tokens':cap,'latency_sec':elapsed,
                             'cost_usd':cv,'cost_source':cost_source,
                             'provider_or_pinned_route':provider or '',
                             'error_category':extract_failure_bucket(str(error)) if error else '',
                             'has_error':bool(error)})
    parsed_compare=Counter();parsed_examples=[]
    for did,decoded in last_json_by_doc.items():
        rawtr,rawissues=triples(decoded)
        savedtr,_=triples(records[did].get('pred_relations'))
        parsed_compare['docs_with_strict_json_object_call']+=1
        if rawtr==savedtr:
            parsed_compare['raw_and_saved_triples_equal']+=1
        else:
            parsed_compare['raw_and_saved_triples_differ']+=1
            parsed_compare['raw_triples_not_in_saved']+=len(rawtr-savedtr)
            parsed_compare['saved_triples_not_in_raw']+=len(savedtr-rawtr)
            if len(parsed_examples)<20:
                parsed_examples.append({'document_id':did,'raw_triples':len(rawtr),'saved_triples':len(savedtr),
                                        'raw_only':len(rawtr-savedtr),'saved_only':len(savedtr-rawtr)})
    # Optional event sidecars: we do NOT infer retrieval loss if there is no event evidence.
    event_kinds=Counter();event_values=defaultdict(list);event_doc_count=Counter()
    for path in sorted((run/'events').glob('*/*.json')):
        ev=read_json(path,errors)
        if not isinstance(ev,dict):continue
        kind=str(ev.get('event') or 'unknown')
        event_kinds[kind]+=1
        did=ev.get('document_id')
        if isinstance(did,str):event_doc_count[did]+=1
        data=ev.get('data')
        if isinstance(data,dict):
            for key in ('before_tokens','after_tokens','budget_tokens'):
                n=_number(data.get(key))
                if n is not None:event_values[kind+':'+key].append(n)
    failures=Counter();fail_docs=set();fail_sessions=set()
    for sess,did,detail in prior_fail:
        failures[extract_failure_bucket(detail or '')]+=1
        fail_docs.add(did);fail_sessions.add(sess)
    # Campaign wall times from actual successful and aborted attempts; cooldown not counted.
    campaign_rows=[]
    for path in sorted((root/'reports').glob('campaign_*.json')):
        obj=read_json(path) # not a hard error for unrelated campaigns
        if not isinstance(obj,dict):continue
        affected=[r for r in obj.get('results',[]) if isinstance(r,dict) and r.get('run_id')==args.run_id]
        if not affected:continue
        campaign_rows.append({'session':obj.get('session_id'), 'elapsed_seconds':obj.get('elapsed_seconds'),
                              'status':affected[0].get('status'), 'done_at_report':(affected[0].get('counts') or {}).get('done'),
                              'failed_at_report':(affected[0].get('counts') or {}).get('failed')})
    active_wall=sum(_number(r['elapsed_seconds']) or 0 for r in campaign_rows)
    # Validity checks against original existing report and raw logs.
    archived_micro=((archived.get('metrics') or {}).get('micro') or {})
    score_agreement={k: (None if archived_micro.get(k) is None else abs(micro[k]-archived_micro[k])<1e-10) for k in ('precision','recall','f1')}
    if any(v is False for v in score_agreement.values()):warnings.append('Independently computed micro scores DO NOT match saved report; investigate scoring/record selection before publication')
    if len(finished)!=args.expected:warnings.append('Number of completed records does not match expected dataset size')
    if db_failed:warnings.append('SQLite has failed documents in current run')
    if duplicate_files:warnings.append('Duplicate document IDs detected in record filenames')
    if unknown_cost:warnings.append('Some call traces lack recorded costs; observed cost is a lower bound, not a complete bill')
    if zero_cost_with_tokens:warnings.append('Some calls report zero cost despite nonzero token use; verify billed amount')
    if routing_violations:warnings.append('Some saved calls did not satisfy the intended exclusive DeepInfra routing')
    # Boundaries: one configuration cannot answer cross-experiment reviewer questions.
    not_testable=['Cross-backbone performance, robustness or statistical significance between methods',
                  'Equal-context budget comparison among methods',
                  'Retrieval top-k sensitivity or hyperparameter effects',
                  'Causal attribution of missing gold relations to retrieval or prompting without stage-level instrumentation',
                  'Literature selection PRISMA flow, taxonomy rationale, and method-adaptation fidelity',
                  'Actual GPU/indexing/retrieval costs if not independently recorded',
                  'Provider-invoice reconciliation if some calls lack usage.cost']
    # Summary and Markdown.
    doc_lat=[_number(r.get('elapsed_sec')) for r in records.values()]
    doc_lat=[v for v in doc_lat if v is not None]
    output={
       'about':{'version':VERSION,'generated_at':datetime.now(timezone.utc).isoformat(),
                'source':'saved RAGTree run and SQLite only; no network/model/API requests',
                'run_id':args.run_id,'profile':PROFILE,'phase':PHASE,'method':METHOD,'dataset':DATASET,
                'repo_git_commit':git_commit(repo),'manifest_sha256':sha256(run/'manifest.json'),
                'protocol_sha256':sha256(repo/'configs/experiments/kbs_journal/protocol.yaml'),
                'run_manifest_protocol_mode':manifest.get('protocol_mode'),
                'selected_in_latest_manifest':manifest.get('selected'),
                'fingerprint_protocol_sha256':(manifest.get('fingerprints') or {}).get('protocol')},
       'integrity':integrity,
       'metrics':{'micro':micro,'macro_f1':macro,'macro_labels_count':len(label_rows),
                  'total_tp':total['tp'],'total_fp':total['fp'],'total_fn':total['fn'],
                  'saved_report_micro':archived_micro,'independent_score_agreement':score_agreement,
                  'saved_report_status':archived.get('status'),'saved_report_record_count':archived.get('record_count'),
                  'document_bootstrap_95pct_micro_f1':ci,
                  'pair_only_micro':pair_micro},
       'relation_loss_proxies':{
           'gold_total':sum(per_gold.values()),'predicted_total':sum(per_pred.values()),
           'predicted_to_gold_ratio':sum(per_pred.values())/sum(per_gold.values()) if sum(per_gold.values()) else None,
           'documents_with_no_gold_relations':zero_gold,
           'documents_with_zero_predictions':zero_pred,
           'documents_with_zero_predictions_and_nonzero_gold':zero_pred-both_zero,
           'gold_relations_missed_on_zero_prediction_documents':total['fn_in_zero_prediction_docs'],
           'predictions_matching_gold_endpoint_pairs_but_wrong_label':total['pred_with_gold_endpoint_pair_wrong_label'],
           'predictions_on_reversed_gold_endpoint_pairs':total['pred_with_reversed_gold_endpoint_pair'],
           'prediction_wrong_pair_or_label_proxy':total['pred_with_gold_pair_wrong_label_or_direction'],
           'malformed_predicted_format':dict(pred_issues), 'malformed_gold_format':dict(gold_issues),
           'predicted_labels_absent_from_saved_schema':dict(unknown_pred_labels),
           'cross_sentence_comparable_gold':cross_known,
           'cross_sentence_gold_pairs':cross_different,
           'cross_sentence_gold_correct':cross_correct,
           'cross_sentence_gold_recall_if_comparable':cross_correct/cross_different if cross_different else None,
           'cross_sentence_fraction_of_comparable':cross_different/cross_known if cross_known else None,
           'IMPORTANT':'These are loss symptoms/proxies, not causal proof of retrieval, generation, prompt or parser failures.'},
       'api_calls':{
           'saved_trace_count':call_stats['traces'],'calls_per_saved_document_mean':call_stats['traces']/len(records) if records else None,
           'traces_on_saved_documents':sum(v for k,v in traces_doc.items() if k in records),
           'traces_on_missing_documents':sum(v for k,v in traces_doc.items() if k not in records),
           'docs_with_no_call_traces':sum(did not in traces_doc for did in records),
           'docs_with_more_than_one_call_trace':sum(traces_doc[did]>1 for did in records),
           'unique_response_ids':len(response_ids),'duplicate_response_ids':{k:v for k,v in response_ids.items() if v>1},
           'response_missing_calls':no_response, 'trace_error_calls':call_stats['error_trace_calls'],
           'raw_content_json_syntax':dict(json_syntax_counts),
           'raw_content_empty_calls':raw_empty,'finish_reason_counts':dict(finish_counts),
           'raw_vs_saved_prediction_comparison':dict(parsed_compare),
           'raw_vs_saved_discrepancy_examples':parsed_examples,
           'completion_exceeds_requested_cap':above_cap,
           'requested_generation_token_caps':dict(generation_caps),
           'checked_pinned_route_calls':routing_checked,'routing_violations_or_other_response_provider':routing_violations,
           'provider_or_request_route_counts':dict(provider_counts),
           'request_roles':dict(request_roles),
           'request_message_count_per_call':stats(request_message_counts),
           'prompt_content_characters_per_call':stats(request_prompt_chars),
           'temperature_values':dict(temperature_values),
           'prompt_tokens_per_call':stats(input_tokens),'completion_tokens_per_call':stats(output_tokens),
           'reasoning_tokens_per_call':stats(reason_tokens),'latency_seconds_per_call':stats(latencies),
           'raw_visible_chars_per_call':stats(raw_chars)},
       'costs':{
           'observed_trace_cost_usd':cost_tot,'calls_with_recorded_cost':known_cost,
           'calls_missing_recorded_cost':unknown_cost,
           'calls_reporting_zero_cost_despite_token_use':zero_cost_with_tokens,
           'interpretation':'Sum of non-overlapping cost fields from archived call traces; excludes unrecorded charges and cannot replace invoice',
           'cost_fields_precedence':['usage.cost','kbs_litellm_reported_cost_usd','usage.cost_details.upstream_inference_cost'],
           'derived_ratecard_per_million_tokens_usd':safe_ratecard(repo),
           'all_traces_prompt_tokens':sum(input_tokens),'all_traces_completion_tokens':sum(output_tokens),
           'all_traces_reasoning_tokens':sum(reason_tokens)},
       'timing':{'per_document_elapsed_sec':stats(doc_lat),
                 'sum_document_elapsed_sec_not_wall':sum(doc_lat),
                 'campaign_attempts_with_reports':campaign_rows,
                 'sum_campaign_attempt_wall_sec_excluding_controller_cooldowns':active_wall,
                 'caveat':'Parallel document seconds are not wall time; controller cooldowns are not included in campaign elapsed_seconds.'},
       'events':{'event_count':sum(event_kinds.values()),'event_kinds':dict(event_kinds),
                 'documents_with_any_event':len(event_doc_count),
                 'measured_budget_event_token_stats':{k:stats(v) for k,v in event_values.items()},
                 'interpretation':'Absent retrieval/validation events means this stage cannot be diagnosed from final predictions alone.'},
       'failure_history':{'total_failure_rows_all_sessions':len(prior_fail),
                          'distinct_documents_ever_failed':len(fail_docs),
                          'sessions_with_failures':len(fail_sessions),
                          'categories':dict(failures),
                          'still_failed_in_sqlite':len(db_failed)},
       'gold_density_bins':{k:{**dict(v),**prf(v['tp'],v['fp'],v['fn'])} for k,v in support_bins.items()},
       'dataset_characteristics':{'gold_relations_per_document':stats(gold_counts),
                                  'pred_relations_per_document':stats(pred_counts),
                                  'document_text_chars_proxy':stats(doc_text_chars),
                                  'entities_or_events_per_document_if_dict_available':stats(entity_counts)},
       'cannot_conclude_from_one_configuration':not_testable,
       'warnings':warnings,'file_read_errors':errors}
    (out/'summary.json').write_text(json.dumps(output,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf8')
    csv_write(out/'per_relation.csv',label_rows,
              ['relation','gold_support','pred_support','tp','fp','fn','precision','recall','f1'])
    csv_write(out/'per_document.csv',all_doc,
              ['document_id','gold_relations','pred_relations','tp','fp','fn','precision','recall','f1','missing_prediction',
               'text_chars_proxy','pred_invalid_pairs','pred_unknown_endpoints_when_registry_available','entity_registry',
               'wrong_label_pair_predictions','reversed_gold_pair_predictions'])
    csv_write(out/'call_audit.csv',call_records,
              ['document_id','trace','session','has_response','response_id','finish_reason','raw_content_json_status',
               'prompt_tokens','completion_tokens','reasoning_tokens','max_requested_completion_tokens','latency_sec',
               'cost_usd','cost_source','provider_or_pinned_route','error_category','has_error'])
    csv_write(out/'campaign_attempts.csv',campaign_rows,['session','elapsed_seconds','status','done_at_report','failed_at_report'])
    # Explicit fail-proof conclusions; no cherry-picking results.
    coverage_ok=(len(finished)==args.expected and len(status_done)==args.expected and not db_failed
                 and finished==status_done and len(records)==args.expected
                 and (not isinstance(eval_ids,list) or set(map(str,eval_ids))==finished))
    score_ok=(all(v is True for v in score_agreement.values()) and archived_micro and archived.get('status')=='complete')
    headline='PASS' if coverage_ok and score_ok and not errors else 'REVIEW REQUIRED'
    md=[]
    md.append('# ONE-SHOT KBS DIAGNOSTIC — '+headline+'\n')
    md.append(f'**Run:** `{args.run_id}` · **Cell:** {PHASE}/{METHOD}/{DATASET} · **Evidence:** local saved files only.\n')
    md.append(f'**Protocol:** `{manifest.get("protocol_mode")}` · **No inference, network, or checkpoint mutation.**\n')
    md.append('## 1. Evidence and integrity\n')
    md.append(table_md([{'item':'Expected documents','value':args.expected},
                        {'item':'Done record files','value':len(finished)},
                        {'item':'SQLite done','value':len(status_done)},
                        {'item':'SQLite still failed','value':len(db_failed)},
                        {'item':'Score agrees with saved metrics','value':score_agreement},
                        {'item':'Saved report status','value':archived.get('status')},
                        {'item':'Saved exact evaluation population IDs','value':integrity.get('eval_ids_file_count')},
                        {'item':'Missing/unexpected IDs','value':(len(integrity.get('eval_ids_missing_from_saved_records',[])),len(integrity.get('saved_records_not_in_eval_ids',[]))) if isinstance(eval_ids,list) else 'unavailable'}],['item','value']))
    md.append('## 2. Independent relation-level evaluation\n')
    md.append(table_md([{'metric':'TP','value':total['tp']},{'metric':'FP','value':total['fp']},
                        {'metric':'FN','value':total['fn']},{'metric':'Micro P','value':round(micro['precision'],6)},
                        {'metric':'Micro R','value':round(micro['recall'],6)},
                        {'metric':'Micro F1','value':round(micro['f1'],6)},
                        {'metric':'Macro F1','value':round(macro,6)},
                        {'metric':'Document-bootstrap micro F1 95% CI (sampling only)',
                         'value':(round(ci['lower'],6),round(ci['upper'],6)) if ci else 'unavailable'},
                        {'metric':'Pair-only micro F1 (diagnostic, not official)','value':round(pair_micro['f1'],6)}],['metric','value']))
    md.append(f'Per-label details: `per_relation.csv` ({len(label_rows)} labels). Per-document details: `per_document.csv`.\n')
    md.append('## 3. Low-recall audit (symptoms, not causal explanations)\n')
    md.append(table_md([{'indicator':'Total gold relations','value':sum(per_gold.values())},
                        {'indicator':'Total predicted relations','value':sum(per_pred.values())},
                        {'indicator':'Docs with gold but no prediction','value':zero_pred-both_zero},
                        {'indicator':'Missed gold relations on those docs','value':total['fn_in_zero_prediction_docs']},
                        {'indicator':'Predicted pairs with wrong gold label','value':total['pred_with_gold_endpoint_pair_wrong_label']},
                        {'indicator':'Predicted pairs reverse a gold pair','value':total['pred_with_reversed_gold_endpoint_pair']},
                        {'indicator':'Invalid predicted pair structures','value':pred_issues.get('invalid_relation_pair',0)},
                        {'indicator':'Comparable cross-sentence gold pairs','value':cross_known},
                        {'indicator':'Cross-sentence among comparable','value':cross_different}],['indicator','value']))
    md.append('**Scope:** These measurements cannot prove whether missing relations were lost during retrieval, prompt construction, generation, parsing or validation. That requires explicit stage traces; only recorded event kinds are audited below.\n')
    md.append('**Dataset characteristics:** '+json.dumps(output['dataset_characteristics'],ensure_ascii=False)+'\n')
    md.append('## 4. Model output and parsing audit\n')
    md.append(table_md([{'item':'Saved call traces (all retries)','value':call_stats['traces']},
                        {'item':'Call traces with API/adapter errors','value':call_stats['error_trace_calls']},
                        {'item':'Calls without response body','value':no_response},
                        {'item':'Raw JSON syntax classification','value':dict(json_syntax_counts)},
                        {'item':'Finish reasons','value':dict(finish_counts)},
                        {'item':'Calls exceeding requested generation cap','value':above_cap},
                        {'item':'Requested generation caps','value':dict(generation_caps)},
                        {'item':'Pinned route violations','value':routing_violations},
                        {'item':'Provider / pinned route','value':dict(provider_counts)},
                        {'item':'Request temperatures','value':dict(temperature_values)},
                        {'item':'Message roles','value':dict(request_roles)}],['item','value']))
    md.append('**Last strict-JSON response vs saved prediction:** '+json.dumps(dict(parsed_compare),ensure_ascii=False)+'\n')
    md.append('**Schema labels absent from predicted labels:** '+json.dumps(dict(unknown_pred_labels),ensure_ascii=False)+'\n')
    md.append('Strict raw JSON syntax and the last response comparison are audit proxies, **not proof** that a method parser accepted or rejected a reply. All nonempty call traces (including retries) are counted. See `call_audit.csv`.\n')
    md.append('## 5. Tokens, provider cost, and runtime\n')
    md.append(table_md([{'measurement':'Total prompt tokens across traced calls','value':human(sum(input_tokens))},
                        {'measurement':'Total completion tokens across traced calls','value':human(sum(output_tokens))},
                        {'measurement':'Reasoning tokens reported','value':human(sum(reason_tokens))},
                        {'measurement':'Calls with recorded cost','value':known_cost},
                        {'measurement':'Calls missing cost','value':unknown_cost},
                        {'measurement':'Zero-cost calls with token usage','value':zero_cost_with_tokens},
                        {'measurement':'Recorded cost total (USD; not an invoice)','value':human(cost_tot,6)},
                        {'measurement':'Sum document compute time (seconds; parallel)','value':human(sum(doc_lat),2)},
                        {'measurement':'Attempt wall seconds, excludes cooldowns','value':human(active_wall,2)}],['measurement','value']))
    if unknown_cost:md.append(f'**Cost caution:** {unknown_cost} call traces have no API cost field. The recorded total is **not** a verified complete bill.\n')
    md.append('## 6. Stage events and historical errors\n')
    md.append('**Recorded event types:** '+json.dumps(dict(event_kinds),ensure_ascii=False)+'\n\n')
    md.append('**Historic failure categories across retries:** '+json.dumps(dict(failures),ensure_ascii=False)+'\n\n')
    md.append('**Gold-density strata:** '+json.dumps(output['gold_density_bins'],ensure_ascii=False)+'\n\n')
    md.append('## 7. Explicit reviewer-evidence boundaries\n')
    md.append('This single completed cell **cannot** answer the following without other independent evidence:\n')
    for item in not_testable:md.append('- '+item+'\n')
    md.append('\n## 8. Caveats and blockers\n')
    if not warnings and not errors:md.append('No internal-integrity warnings in inspected files.\n')
    for warning in warnings:md.append('- WARNING: '+warning+'\n')
    for e in errors[:15]:md.append('- READ ERROR: '+e['path']+': '+e['error']+'\n')
    md.append('\n**Decision:** '+('The current cell has full checkpoint coverage and independent metrics agree with the saved scoring report.' if headline=='PASS' else 'Resolve data-integrity or scoring discrepancies before citing these results.')+'\n')
    (out/'REPORT.md').write_text('\n'.join(md)+'\n',encoding='utf8')
    print('\nONE-SHOT KBS DIAGNOSTIC:',headline)
    print('Run:',args.run_id,'| docs:',len(finished),'/',args.expected,'| SQLite done:',len(status_done),'| failed:',len(db_failed))
    print('Micro P/R/F1:',*(f'{micro[k]:.6f}' for k in ('precision','recall','f1')),'| Macro F1:',f'{macro:.6f}')
    print('TP/FP/FN:',total['tp'],total['fp'],total['fn'],'| Predicted:',sum(per_pred.values()),'Gold:',sum(per_gold.values()))
    print('Raw call traces:',call_stats['traces'],'| Usage-cost-covered calls:',known_cost,'| Unknown cost:',unknown_cost)
    print('Recorded trace cost USD:',f'{cost_tot:.6f}','| Historic failure categories:',dict(failures))
    print('Score agreement:',score_agreement)
    print('Output:',out)
    print('FILES: REPORT.md, summary.json, per_relation.csv, per_document.csv, call_audit.csv, campaign_attempts.csv')
    print('No API calls, no reruns, no edits to checkpoints.')
    return 0 if headline=='PASS' else 2


if __name__=='__main__':
    sys.exit(main())
