#!/usr/bin/env python3
"""One-off, low-cost *diagnostic*, never an official KBS evaluation run.

FinCausal: replay N archived GPT-baseline prompts with role annotations masked and
entity lines shuffled; compare paired subset with ORIGINAL archived predictions.
CausalBank: offline score existing GPT/Qwen baseline/ICL on the existing gold and
on a *heuristic* cross-connector-clause subset (both gold and predictions filtered).

Default: no API calls. --execute authorizes at most --sample paid calls, pinned to
DeepInfra, with a cumulative usage.cost ceiling. Does not import/modify RAGTree.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, random, re, sys, time
from pathlib import Path
from urllib import request, error
from datetime import datetime, timezone

FINDATA={
    ('main_gpt','baseline','fincausal'):'7ad37dc5f42a0f67dc3ce383',
}
CAUSDATA={
    ('main_gpt','baseline','causalbank'):'de3ba9ce7e87a72acfb82bd4',
    ('main_qwen','baseline','causalbank'):'1e762517400e5879da605f33',
    ('main_gpt','icl','causalbank'):'d973386e0dfad65e3d8783d4',
    ('main_qwen','icl','causalbank'):'837355425cc79e9647b09b1b',
}
BASE='results/kbs_journal/openrouter'
MODEL='openai/gpt-oss-20b'
ENDPOINT='https://openrouter.ai/api/v1/chat/completions'
ALLOW=('temperature','top_p','max_tokens','seed','stop','frequency_penalty','presence_penalty','reasoning_effort','response_format')

def readj(p):return json.loads(p.read_text(encoding='utf-8'))
def triples(obj):
    t=set()
    if isinstance(obj,dict):
        for label,pairs in obj.items():
            if label=='null' or not isinstance(pairs,list):continue
            for pair in pairs:
                if isinstance(pair,(tuple,list)) and len(pair)==2 and all(isinstance(x,str) for x in pair):
                    t.add((label,pair[0],pair[1]))
    return t

def prf(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0.
    r=tp/(tp+fn) if tp+fn else 0.
    f=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    return {'tp':tp,'fp':fp,'fn':fn,'P':p,'R':r,'F1':f}

def score(pairs):
    tp=fp=fn=0
    for gold,pred in pairs:
        tp+=len(gold & pred)
        fp+=len(pred-gold)
        fn+=len(gold-pred)
    return prf(tp,fp,fn)

def load_records(repo,cell,rid,expected):
    phase,method,ds=cell
    path=repo/BASE/phase/method/ds/rid
    if not path.is_dir():raise RuntimeError(f'Missing completed run: {path}')
    manifest=readj(path/'manifest.json')
    if manifest.get('run_id')!=rid or manifest.get('protocol_mode')!='controlled_revision_v2':
        raise RuntimeError(f'Incorrect identity/protocol: {path}')
    records={}
    for f in (path/'records').glob('*.json'):
        r=readj(f)
        if r.get('status')=='done':
            did=r.get('document_id') or (r.get('source_document') or {}).get('document_id')
            if not did or did in records:raise RuntimeError(f'Duplicate/empty document ID in {path}')
            records[did]=r
    if len(records)!=expected:
        raise RuntimeError(f'{cell}: {len(records)} completed vs expected {expected}; refusing unpaired comparison')
    return path,records

def clause_side_sets(doc):
    """Detect entities mentioned strictly to left/right of ORIGINAL connector.
    This is a heuristic restricted subtask, NOT a replacement ground truth.
    """
    title=doc.get('title')
    text=doc.get('text')
    if not isinstance(title,str) or not isinstance(text,str) or ' - ' not in title or ' | ' not in title:return None
    conn, phrases=title.split(' - ',1)
    left, right=phrases.rsplit(' | ',1)
    if not left or not right or text!=f'{left} {conn} {right}':return None
    left_end=len(left.split());right_start=left_end+len(conn.split())
    entities=doc.get('entities')
    if not isinstance(entities,dict):return None
    left_ids,right_ids=set(),set()
    for eid,e in entities.items():
        if not isinstance(e,dict):continue
        positions=[]
        for m in e.get('mentions',[]):
            if isinstance(m,dict) and m.get('sentence_id',m.get('sent_id',0))==0 and isinstance(m.get('token_id'),int):
                positions.append(m['token_id'])
        if not positions:continue
        if all(p<left_end for p in positions):left_ids.add(eid)
        if all(p>=right_start for p in positions):right_ids.add(eid)
    return left_ids,right_ids

def causscore(repo):
    data=[]
    for cell,rid in CAUSDATA.items():
        run,records=load_records(repo,cell,rid,1080)
        original=[];restricted=[];parsed=0;unparsed=0;g_total=0;g_cross=0;p_total=0;p_cross=0
        for rec in records.values():
            doc=rec.get('source_document') or {}
            gold=triples(doc.get('relations'));pred=triples(rec.get('pred_relations'))
            original.append((gold,pred));g_total+=len(gold);p_total+=len(pred)
            sides=clause_side_sets(doc)
            if sides is None:unparsed+=1;continue
            left,right=sides;parsed+=1
            def across(triple):
                _,h,t=triple
                return (h in left and t in right) or (h in right and t in left)
            gg={t for t in gold if across(t)};pp={t for t in pred if across(t)}
            g_cross+=len(gg);p_cross+=len(pp)
            restricted.append((gg,pp))
        data.append({'model':cell[0],'method':cell[1],'docs':len(records),
                     'parsed_docs':parsed,'unparsed_docs':unparsed,
                     'gold_original':g_total,'pred_original':p_total,
                     'gold_cross':g_cross,'pred_cross':p_cross,
                     'original':score(original), 'cross_phrase_subtask':score(restricted)})
    return data

def trace_for_doc(run,did):
    h=hashlib.sha256(did.encode()).hexdigest()[:24]
    files=sorted((run/'calls'/h).glob('*.json'))
    candidates=[]
    for f in files:
        try:
            tr=readj(f)
            if tr.get('document_id')!=did:continue
            if tr.get('response') and (tr.get('request') or {}).get('messages'):
                candidates.append((f,tr))
        except Exception:pass
    if not candidates:raise RuntimeError(f'No successful archived prompt for {did}')
    # If retry occurred prefer most recent successful API trace.
    return candidates[-1]

def mask_messages(messages, seed):
    # Keep same system and user instructions except role cues in entity list.
    if not isinstance(messages,list) or len(messages)!=2:raise RuntimeError('Unexpected messages shape')
    copied=json.loads(json.dumps(messages))
    msg=copied[-1]
    s=msg.get('content')
    if not isinstance(s,str):raise RuntimeError('No user prompt')
    # Match only section between Entities:\n and the final JSON instruction.
    match=re.search(r'(?ms)(^Entities:\s*\n)(.*?)(\n\s*Now output ONLY the JSON object\.)',s)
    if not match:raise RuntimeError('Expected saved baseline Entities block not found')
    block=match.group(2)
    lines=[x for x in block.splitlines() if x.lstrip().startswith('- ')]
    if not lines:raise RuntimeError('No entity rows')
    old_roles=sum(len(re.findall(r'\[(?:CAUSE|EFFECT)\]',x)) for x in lines)
    if old_roles<2:raise RuntimeError('Less than two causal entity-role annotations; cannot test leakage')
    transformed=[]
    for line in lines:
        line=re.sub(r'\[(?:CAUSE|EFFECT)\]', '[ENTITY]',line)
        line=re.sub(r',?\s*mention_id=[^,)\s]+','',line)
        transformed.append(line)
    rng=random.Random(seed)
    rng.shuffle(transformed)
    neutral='\n'.join(transformed)
    msg['content']=s[:match.start(2)]+neutral+s[match.end(2):]
    if msg['content']==s:raise RuntimeError('Prompt was not modified')
    # no relation labels, event IDs, actual sentence, or system instructions modified
    return copied,old_roles

def api_request(payload,key,timeout=120):
    body=json.dumps(payload,ensure_ascii=False).encode()
    req=request.Request(ENDPOINT,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},data=body,method='POST')
    try:
        with request.urlopen(req,timeout=timeout) as resp:return json.load(resp)
    except error.HTTPError as exc:
        msg=exc.read(1000).decode('utf-8','replace')
        raise RuntimeError(f'HTTP_{exc.code}: {msg[:400]}') from exc

def finc_setup(repo,n,seed):
    cell=next(iter(FINDATA));rid=FINDATA[cell]
    run,records=load_records(repo,cell,rid,967)
    rng=random.Random(seed)
    population=[did for did,r in sorted(records.items()) if triples((r.get('source_document') or {}).get('relations'))]
    if len(population)<n:raise RuntimeError('Insufficient gold-positive docs for requested sample')
    selection=rng.sample(population,n)
    out=[]
    for i,did in enumerate(selection):
        f,tr=trace_for_doc(run,did)
        req=tr['request']; params=req.get('parameters') or {}
        messages,num=mask_messages(req['messages'],seed+i)
        if str(params.get('temperature',0)) not in ('0','0.0'):raise RuntimeError('Only fixed temp zero supported')
        if params.get('reasoning_effort') is None:raise RuntimeError('Missing reasoning effort in saved trace')
        raw_params={k:v for k,v in params.items() if k in ALLOW and v is not None}
        payload={'model':MODEL,'messages':messages,'provider':{'only':['deepinfra'],'allow_fallbacks':False,'require_parameters':True},**raw_params}
        if payload.get('max_tokens',0)>4096:payload['max_tokens']=4096
        if payload.get('max_tokens',0)<1:raise RuntimeError('Missing max_tokens')
        rec=records[did]
        out.append({'did':did,'saved_gold':triples((rec.get('source_document') or {}).get('relations')),
                    'saved_pred':triples(rec.get('pred_relations')),'payload':payload,'roles_removed':num,
                    'trace':str(f.relative_to(run))})
    return out

def parse_response(resp,allowed_ids,labels):
    choice=(resp.get('choices') or [{}])[0]
    content=(choice.get('message') or {}).get('content')
    if not isinstance(content,str) or not content.strip():raise RuntimeError('No final-answer text')
    text=content.strip()
    if text.startswith('```'):
        text=re.sub(r'^```(?:json)?\s*','',text,flags=re.I)
        text=re.sub(r'\s*```$','',text)
    value=json.loads(text)
    if not isinstance(value,dict):raise RuntimeError('Response is not JSON object')
    # Match baseline's normalized relation structure without hallucinated entities/labels.
    raw=triples(value)
    valid={(lab,h,t) for lab,h,t in raw if lab in labels and h in allowed_ids and t in allowed_ids}
    return valid,choice.get('finish_reason'),content

def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,default=Path('.'))
    ap.add_argument('--sample',type=int,default=30)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--max-cost-usd',type=float,default=0.05)
    ap.add_argument('--execute',action='store_true',help='Run N paid FinCausal role-neutral calls; CausalBank always offline')
    args=ap.parse_args()
    if not 5<=args.sample<=60:ap.error('--sample must be 5..60')
    if not 0<args.max_cost_usd<=1:ap.error('Cost cap must be positive <=$1')
    repo=args.repo.resolve()
    print('Reading completed FinCausal/CausalBank checkpoints (no writes to official runs)...',flush=True)
    causal=causscore(repo)
    fin=finc_setup(repo,args.sample,args.seed)
    print('CausalBank cross-phrase diagnostic, saved outputs only:',flush=True)
    for r in causal:
        print(f"  {r['model']} {r['method']}: gold {r['gold_original']} -> {r['gold_cross']}; F1 {100*r['original']['F1']:.2f}% -> {100*r['cross_phrase_subtask']['F1']:.2f}%, parsed {r['parsed_docs']}/1080",flush=True)
    origin=score([(x['saved_gold'],x['saved_pred']) for x in fin])
    print(f"FinCausal GPT baseline deterministic sample: {args.sample} gold-positive docs; saved-original F1={100*origin['F1']:.2f}%",flush=True)
    print(f"Role labels removed and entity order shuffled; {args.sample} new requests at most; provider fixed DeepInfra; max recorded cost ${args.max_cost_usd:.2f}",flush=True)
    if not args.execute:
        print('DRY RUN ONLY. To perform SMALL paid role-neutral A/B test add --execute.',flush=True)
        return 0
    key=os.environ.get('OPENROUTER_API_KEY')
    if not key:ap.error('OPENROUTER_API_KEY missing; source ~/.config/ragtree/openrouter.env')
    out=repo/'results/kbs_journal/diagnostics'/('dataset_quick_ab_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True,exist_ok=False)
    results=[]; all_responses=[];cost=0.;failures=0
    for i,row in enumerate(fin,1):
        gold=row['saved_gold'];baseline=row['saved_pred']
        try:
            res=api_request(row['payload'],key)
            usage=res.get('usage') or {};usagecost=usage.get('cost')
            if not isinstance(usagecost,(float,int)):
                raise RuntimeError('Provider did not return numeric usage.cost; cannot enforce cost cap')
            cost+=float(usagecost)
            pred,finish,content=parse_response(res,{x for lab,h,t in gold|baseline for x in (h,t)},set(lab for lab,_,_ in gold))
            status='ok'
            all_responses.append({'document_id':row['did'],'response_id':res.get('id'),'usage':usage,
                                  'provider':res.get('provider'),'finish_reason':finish,'raw_content':content})
        except Exception as exc:
            failures+=1;pred=None;status=f'{type(exc).__name__}: {str(exc)[:240]}'
        results.append({'document_id':row['did'],'gold_count':len(gold),'original_pred_count':len(baseline),
                        'masked_pred_count':len(pred) if pred is not None else '',
                        'original_correct':len(gold&baseline),'masked_correct':len(gold&pred) if pred is not None else '',
                        'status':status})
        print(f"[{i}/{len(fin)}] {'OK' if pred is not None else 'ERROR'} | cumulative known API cost ${cost:.6f} | failures {failures}",flush=True)
        if failures>=3 or cost>=args.max_cost_usd:
            print('Stopping early: maximum failures or known cost ceiling reached.',flush=True);break
        time.sleep(.25)
    (out/'fin_responses.json').write_text(json.dumps(all_responses,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    write_csv(out/'fincausal_sample.csv',results)
    caussimple=[]
    for r in causal:
        t={k:v for k,v in r.items() if k not in ('original','cross_phrase_subtask')}
        t.update({'original_F1':r['original']['F1'],'cross_F1':r['cross_phrase_subtask']['F1']})
        caussimple.append(t)
    write_csv(out/'causalbank_ab.csv',caussimple)
    okids={x['document_id'] for x in results if x['status']=='ok'}
    completed=[x for x in fin if x['did'] in okids]
    # Decode again from saved responses for the paired score; preflight stores ids, not task data.
    recovered={}
    for rr in all_responses:
        item=next((x for x in fin if x['did']==rr['document_id']),None)
        if item is None:continue
        try:
            resp={'choices':[{'message':{'content':rr['raw_content']},'finish_reason':rr['finish_reason']}]}
            recovered[rr['document_id']]=parse_response(resp,{x for lab,h,t in item['saved_gold']|item['saved_pred'] for x in (h,t)},set(lab for lab,_,_ in item['saved_gold']))[0]
        except Exception:pass
    paired=[x for x in fin if x['did'] in recovered]
    origpaired=score([(x['saved_gold'],x['saved_pred']) for x in paired])
    maskedpaired=score([(x['saved_gold'],recovered[x['did']]) for x in paired])
    delta=100*(maskedpaired['F1']-origpaired['F1'])
    summary={'warning':'EXPLORATORY A/B ONLY. Original official run/protocol/checkpoints unchanged.',
             'fincausal':{'requested_sample':args.sample,'completed_paired':len(paired),'failures':failures,
                          'sample_original_F1':origin['F1'],'paired_original':origpaired,'paired_masked':maskedpaired,
                          'delta_F1_percentage_points':delta,'known_api_cost_usd':cost,
                          'interpretation':'Removing explicit entity roles + mention IDs and shuffling entity order; keep original gold, schema and sentence. No official metric replacement.'},
             'causalbank':causal,'unavailable_claims':['Cross-phrase subset is heuristic and not a corrected gold standard',
                    'Small FinCausal sample cannot certify whole-dataset effect',
                    'Any change to official benchmark would require separately versioned experiment and complete controls']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    report=['# Exploratory dataset A/B diagnostic (isolated)',
            'Official RAGTree results, code and protocol were not changed.',
            f'FinCausal GPT Baseline: {len(paired)}/{args.sample} paired successes; original F1 {100*origpaired["F1"]:.2f}%, role-neutral F1 {100*maskedpaired["F1"]:.2f}%, Δ {delta:+.2f} percentage points. Recorded API cost ${cost:.6f}.',
            'CausalBank: no model calls; metrics restricted to gold AND predictions connecting entities mentioned on opposite sides of the connector. Not canonical gold.',
            '', '| Model | Method | Gold original | Gold cross-phrase | F1 original | F1 restricted |', '|---|---|---:|---:|---:|---:|']
    for r in causal:report.append(f"| {r['model']} | {r['method']} | {r['gold_original']} | {r['gold_cross']} | {100*r['original']['F1']:.2f}% | {100*r['cross_phrase_subtask']['F1']:.2f}% |")
    report.extend(['','FinCausal changes also neutralize mention IDs and randomize entity listing to avoid retaining trivial role cues. This is a targeted exploratory task, NOT a controlled full benchmark replacement.',
                   'For CausalBank, no labels were corrected or added; only a restricted scoring subtask was considered.',
                   'Interpret results alongside the original benchmark and limitations; do not silently replace reported metrics.'])
    (out/'REPORT.md').write_text('\n\n'.join(report)+'\n',encoding='utf8')
    print(f'\nFINCAUSAL PAIRED RESULT: original F1={100*origpaired["F1"]:.2f}% masked F1={100*maskedpaired["F1"]:.2f}% delta={delta:+.2f} percentage points',flush=True)
    print(f'CAUSALBANK: offline diagnostic complete. DETAILS: {out}/REPORT.md',flush=True)
    return 0 if len(paired)==args.sample else 2

if __name__=='__main__':sys.exit(main())
