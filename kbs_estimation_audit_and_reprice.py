#!/usr/bin/env python3
"""Read-only KBS campaign diagnosis and price projection. Does not call an LLM.

Run from RAGTree repository:
    python /path/to/kbs_estimation_audit_and_reprice.py --session <id>

Uses captured local token usage, not local inference dollar charges. Secondary
model and ablation variants remain PROXIES until directly calibrated.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys

DEFAULT_SESSION='304d4e7f88f641f6a9f4cd09860f7914'

def parse():
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--repo',type=Path,default=Path('.'))
    p.add_argument('--session',default=DEFAULT_SESSION)
    p.add_argument('--gpt-input',type=float,default=.02,help='USD / 1 million input tokens')
    p.add_argument('--gpt-output',type=float,default=.10)
    p.add_argument('--qwen-input',type=float,default=.04815)
    p.add_argument('--qwen-output',type=float,default=.1931)
    p.add_argument('--contingency',type=float,default=.30)
    p.add_argument('--save',action='store_true',help='Save derived JSON report (never change original experiment files)')
    args=p.parse_args()
    if any(getattr(args,attr)<0 for attr in ('gpt_input','gpt_output','qwen_input','qwen_output','contingency')):
        p.error('Prices and contingency must be >= 0')
    return args

def read(path):
    with path.open(encoding='utf-8') as f:return json.load(f)

def main():
    args=parse(); repo=args.repo.resolve(); root=repo/'results/kbs_journal'; sid=args.session
    camp=root/'reports'/f'campaign_{sid}.json'; estp=root/'reports'/f'estimate_{sid}.json'
    if not camp.is_file() or not estp.is_file():
        sys.exit(f'Missing original campaign or estimate: {camp}, {estp}')
    campaign=read(camp); estimates=read(estp)
    by_run={x.get('run_id'):x for x in campaign.get('results',[]) if x.get('run_id')}
    print('SESSION:',sid)
    print('Execution passes:',len(campaign.get('results',[])), 'Unique method/dataset estimates:',len(estimates))
    print('Execution statuses:',dict(Counter(x.get('status') for x in campaign.get('results',[]))))
    print('\nFAILED RECORDS AND THEIR SAVED LLM RESPONSES (no inference)')
    failures=[]
    db=root/'checkpoints.sqlite'
    if db.is_file():
        with sqlite3.connect(db) as conn:
            failures=conn.execute('SELECT run_id,doc_id,detail FROM failures WHERE session_id=? ORDER BY run_id,doc_id',(sid,)).fetchall()
    print('Distinct failure entries for session:',len(failures))
    faildata=[]
    for run_id,doc_id,detail in failures:
        item=by_run.get(run_id) or {}
        run_root=Path(item['run_root']) if item.get('run_root') else None
        if run_root is None:
            candidates=list(root.glob(f'*/main_gpt/*/*/{run_id}'))
            run_root=candidates[0] if len(candidates)==1 else None
        trace=[]
        if run_root:
            calls=run_root/'calls'/sha256(doc_id.encode()).hexdigest()[:24]
            for path in sorted(calls.glob('*.json')) if calls.is_dir() else []:
                try:
                    datum=read(path)
                    response=datum.get('response') or {}
                    choices=response.get('choices') or []
                    choice=choices[0] if choices else {}
                    msg=choice.get('message') or {}
                    text=msg.get('content')
                    usage=response.get('usage') or {}
                    req=datum.get('request') or {}
                    params=req.get('parameters') or {}
                    trace.append({'file':path.name,'finish_reason':choice.get('finish_reason'),
                        'input_tokens':usage.get('prompt_tokens'),'output_tokens':usage.get('completion_tokens'),
                        'max_tokens':params.get('max_tokens'),'reasoning_effort':params.get('reasoning_effort'),
                        'content_length':len(text) if isinstance(text,str) else 0,
                        'has_reasoning':bool(msg.get('reasoning') or msg.get('reasoning_content')),
                        'error':datum.get('error')})
                except (OSError,ValueError,TypeError):pass
        details={'run_id':run_id,'record':doc_id,'run_path':str(run_root) if run_root else None,
                 'method_dataset': '/'.join(run_root.parts[-3:-1]) if run_root else None,
                 'last_error':detail[-300:],'calls':trace[-6:]}
        faildata.append(details)
        print('\n',details['method_dataset'],doc_id,'run:',run_id)
        print('   Last error:',details['last_error'].splitlines()[-1])
        if not trace:print('   No saved call traces found: verify run_root and call paths')
        for c in trace[-4:]:
            print('   ',c['file'],'finish=',c['finish_reason'],'tokens=',c['input_tokens'],'/',c['output_tokens'],
                  'cap=',c['max_tokens'],'reasoning=',c['reasoning_effort'],'content_len=',c['content_length'],
                  'trace_error=',c['error'])

    print('\nPILOT TOKEN COMPLETENESS')
    incomplete=[]
    for row in estimates:
        n=row.get('sample_size',0)
        ins=row.get('input_tokens') or {}; outs=row.get('output_tokens') or {}
        ni=ins.get('observed_n',0);no=outs.get('observed_n',0)
        if ni!=n or no!=n or n<6:
            incomplete.append({'method':row['method'],'dataset':row['dataset'],'sample_size':n,
                               'input_observed_n':ni,'output_observed_n':no})
    print('Pilot cells missing complete usage:',len(incomplete))
    for r in incomplete[:30]:print(' ',r)
    if len(incomplete)>30:print(' ... and',len(incomplete)-30,'more')

    # Read job_matrix from the actual user's RAGTree checkout, not from a copied formula.
    sys.path.insert(0,str(repo))
    try:
        from ragtree.experiments.kbs.protocol import job_matrix
        jobs=job_matrix()
    except Exception as e:
        sys.exit('Could not import the installed KBS job matrix: '+repr(e))
    pilot={(r['method'],r['dataset'],r['model']):r for r in estimates}
    rates={'gpt_oss_20b':(args.gpt_input,args.gpt_output),
           'qwen3_30b_a3b':(args.qwen_input,args.qwen_output)}
    phase_totals=defaultdict(lambda: {'jobs':0,'dollar_jobs':0,'base_usd':0.,'conservative_usd':0.,'proxy_jobs':0})
    details=[]
    for job in jobs:
        row=pilot.get((job.method,job.dataset,job.model))
        proxy=False
        if row is None:
            row=pilot.get((job.method,job.dataset,'gpt_oss_20b'))
            proxy=bool(row)
        phase=phase_totals[job.phase];phase['jobs']+=1
        item={'phase':job.phase,'method':job.method,'dataset':job.dataset,'model':job.model,
              'variant':job.variant,'proxy':proxy,'estimated_usd':None,'conservative_usd':None}
        if proxy:phase['proxy_jobs']+=1
        if row:
            a=row.get('input_tokens') or {};b=row.get('output_tokens') or {}
            nin=a.get('observed_n',0);nout=b.get('observed_n',0)
            item['sample_size']=row.get('sample_size');item['observed_input_n']=nin;item['observed_output_n']=nout
            if (a.get('mean_per_record') is not None and b.get('mean_per_record') is not None
                and nin==row.get('sample_size') and nout==row.get('sample_size')):
                population=row['population'];rp_in,rp_out=rates[job.model]
                item['estimated_usd']=population*(a['mean_per_record']*rp_in+b['mean_per_record']*rp_out)/1e6
                up_a=a.get('projected_upper_97_5');up_b=b.get('projected_upper_97_5')
                if up_a is not None and up_b is not None:
                    item['conservative_usd']=(up_a*rp_in+up_b*rp_out)/1e6
                phase['dollar_jobs']+=1;phase['base_usd']+=item['estimated_usd']
                if item['conservative_usd'] is not None:phase['conservative_usd']+=item['conservative_usd']
        details.append(item)
    count=len(details);valued=sum(x['estimated_usd'] is not None for x in details)
    print('\nBUDGET PROJECTION — USING SAVED TOKENS ONLY')
    print('Priced jobs:',valued,'/',count,'(unpriced jobs exclude missing token information)')
    for key,v in phase_totals.items():
        print(f"  {key:24} priced {v['dollar_jobs']:3}/{v['jobs']:3}  mean ${v['base_usd']:8.3f}  conservative ${v['conservative_usd']:8.3f}  proxy_jobs={v['proxy_jobs']}")
    allpriced=valued==count
    subtotal=sum(x['estimated_usd'] for x in details if x['estimated_usd'] is not None)
    upper=sum(x['conservative_usd'] for x in details if x['conservative_usd'] is not None)
    print('Priced subset mean:',round(subtotal,3),'USD')
    if allpriced:
        print('Whole-campaign illustrative mean:',round(subtotal,3),'USD')
        print('Whole-campaign illustrative conservative + contingency:',round(upper*(1+args.contingency),3),'USD')
    else:
        print('WHOLE-CAMPAIGN BUDGET: INCOMPLETE — do not present subtotal as total')
    print('LIMITATIONS: 2nd-model token counts are gpt-oss proxies; ablations use native-phase token distributions;')
    print('failed responses, retries, tool fees and purchases/credit fees are not fully priced; check provider rates.')
    if args.save:
        out=root/'reports'/f'kbs_budget_audit_{sid}.json'
        payload={'session':sid,'generated_at':datetime.now(timezone.utc).isoformat(),
                 'ratecard_usd_per_million':{k:dict(zip(('input','output'),v)) for k,v in rates.items()},
                 'phase_summary':dict(phase_totals),'pilot_incomplete':incomplete,'failures':faildata,
                 'jobs':details,'completeness':{'priced_jobs':valued,'total_jobs':count},
                 'illustrative_full_mean_usd':subtotal if allpriced else None,
                 'illustrative_full_upper_with_contingency_usd':upper*(1+args.contingency) if allpriced else None}
        out.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print('Saved derived report:',out)
    return 0

if __name__=='__main__':raise SystemExit(main())
