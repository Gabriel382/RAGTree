"""Stratified pilot sampling and finite-cost projection from successful traces."""
from __future__ import annotations
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics
from .engine import filtered_docs
from .protocol import load_protocol


def _length(doc):
    return len(str(doc.get('text','') or doc.get('sentences','') or ''))

def _gold_density(doc):
    rel=doc.get('relations',{})
    return sum(len(x) for x in rel.values() if isinstance(x,list)) if isinstance(rel,dict) else 0

def sample_ids(job,count=6,seed=42):
    """Deterministic length/density strata; nesting guaranteed for expanding n."""
    docs=list(filtered_docs(job))
    if not docs:return []
    docs=sorted(docs,key=lambda d:(_length(d),_gold_density(d),str(d.get('document_id') or d.get('id'))))
    # Round-robin over length terciles, ordered by density within each.
    groups=[docs[(len(docs)*i)//3:(len(docs)*(i+1))//3] for i in range(3)]
    rng=random.Random(seed)
    for g in groups:g.sort(key=lambda d:(_gold_density(d),_length(d)))
    ordering=[]
    for i,g in enumerate(groups):
        # Sample permutation fixed across expansions
        rng.shuffle(g)
    while any(groups):
        for g in groups:
            if g:ordering.append(g.pop())
    return [str(d.get('document_id') or d.get('id')) for d in ordering[:count]]


def scan_records(run_root:Path):
    for p in (run_root/'records').glob('*.json') if (run_root/'records').is_dir() else ():
        try: yield json.loads(p.read_text(encoding='utf8'))
        except (OSError, ValueError):continue


def calls_for(run_root:Path,doc_id:str):
    import hashlib
    name=hashlib.sha256(doc_id.encode()).hexdigest()[:24]
    for p in sorted((run_root/'calls'/name).glob('*.json')) if (run_root/'calls'/name).is_dir() else ():
        try:yield json.loads(p.read_text(encoding='utf8'))
        except (OSError,ValueError):pass


def cost_per_record(run_root:Path,record:dict,model_key:str,prices:dict):
    calls=list(calls_for(run_root,record['document_id']))
    tokens_in=tokens_out=0
    known_tokens=True
    known_price=True
    billed=0.0
    for call in calls:
        result=call.get('response') or {}
        usage=result.get('usage') or {}
        input_v=usage.get('prompt_tokens')
        output_v=usage.get('completion_tokens')
        if input_v is None or output_v is None:known_tokens=False
        else:tokens_in+=int(input_v);tokens_out+=int(output_v)
        v=result.get('kbs_litellm_reported_cost_usd')
        if v is None:known_price=False
        else:billed+=float(v)
    if not calls:known_tokens=False;known_price=False
    price=prices.get(model_key,{})
    projected=(tokens_in*float(price['input'])+tokens_out*float(price['output']))/1e6 if known_tokens and price.get('input') is not None and price.get('output') is not None else None
    return {'document_id':record['document_id'],'llm_calls':len(calls),
            'input_tokens':tokens_in if known_tokens else None,
            'output_tokens':tokens_out if known_tokens else None,
            'provider_cost_usd':billed if known_price else None,
            'ratecard_cost_usd':projected,'runtime_sec':record.get('elapsed_sec')}


def _bootstrap_upper(samples:list[float],n=400,seed=42):
    if not samples:return None
    rng=random.Random(seed)
    boots=[statistics.mean(rng.choices(samples,k=len(samples))) for _ in range(n)]
    return sorted(boots)[int(.975*(len(boots)-1))]


def estimate_run(job,run_root:Path,population:int):
    proto=load_protocol()
    cost_rows=[cost_per_record(run_root,r,job.model,proto['price_per_million_usd']) for r in scan_records(run_root)]
    result={'method':job.method,'dataset':job.dataset,'model':job.model,'phase':job.phase,
            'population':population,'sample_size':len(cost_rows)}
    for label in ('input_tokens','output_tokens','provider_cost_usd','ratecard_cost_usd','runtime_sec','llm_calls'):
        valid=[float(r[label]) for r in cost_rows if r[label] is not None]
        avg=statistics.mean(valid) if valid else None
        upper=_bootstrap_upper(valid)
        result[label]={'observed_n':len(valid),'mean_per_record':avg,
                        'projected_total':(avg*population if avg is not None else None),
                        'projected_upper_97_5':upper*population if upper is not None else None}
    result['contingency_fraction']=proto.get('contingency_fraction',0.30)
    money=result['provider_cost_usd']['projected_upper_97_5']
    if money is None:money=result['ratecard_cost_usd']['projected_upper_97_5']
    result['recommended_budget_usd']=(money*(1+result['contingency_fraction']) if money is not None else None)
    return result

def projected_campaign(estimates:list[dict]):
    """Expand measured pilot unit costs to all planned phase configurations.

    Models without their own pilot are explicitly proxy extrapolations. Missing
    rate-card prices remain null; no imaginary dollar total is produced.
    """
    from .protocol import job_matrix
    protocol=load_protocol()
    measured={(r['method'],r['dataset'],r['model']):r for r in estimates}
    planned=[]
    for job in job_matrix():
        cell=measured.get((job.method,job.dataset,job.model))
        proxy=False
        if cell is None:
            cell=measured.get((job.method,job.dataset,'gpt_oss_20b'))
            proxy=bool(cell)
        if cell is None:
            planned.append({'phase':job.phase,'method':job.method,'dataset':job.dataset,
                            'model':job.model,'status':'unestimated'})
            continue
        total=cell['population']
        input_m=cell['input_tokens']['mean_per_record']
        output_m=cell['output_tokens']['mean_per_record']
        rates=protocol['price_per_million_usd'].get(job.model,{})
        dollars=(total*(input_m*float(rates['input'])+output_m*float(rates['output']))/1e6
                 if input_m is not None and output_m is not None
                 and rates.get('input') is not None and rates.get('output') is not None else None)
        planned.append({'phase':job.phase,'method':job.method,'dataset':job.dataset,'model':job.model,
                        'status':'secondary_model_proxy' if proxy else 'pilot_extrapolation',
                        'records':total,'input_tokens':total*input_m if input_m is not None else None,
                        'output_tokens':total*output_m if output_m is not None else None,
                        'llm_calls':total*cell['llm_calls']['mean_per_record'] if cell['llm_calls']['mean_per_record'] is not None else None,
                        'estimated_usd':dollars})
    known=[p['estimated_usd'] for p in planned if p.get('estimated_usd') is not None]
    result={'configured_jobs':len(planned),'covered_by_pilot_or_proxy':sum(p['status']!='unestimated' for p in planned),
            'jobs_with_dollar_estimates':len(known),
            'partial_dollar_total_usd':sum(known) if known else None,
            'full_campaign_dollar_total_usd':sum(known) if len(known)==len(planned) else None,
            'contingency_fraction':protocol['contingency_fraction'],'units':'USD excluding credit fees/taxes',
            'jobs':planned,'limitations':[
                'A secondary-model proxy does not measure that model output length or reasoning token consumption.',
                'Phase variants may change token use; base-pilot extrapolation is preliminary.',
                'Only sum costs as a full budget when all cells have measured or explicitly priced estimates.',
                'Provider pricing/credit fees/taxes must be verified at purchase time.']}
    if result['full_campaign_dollar_total_usd'] is not None:
        result['with_contingency_usd']=result['full_campaign_dollar_total_usd']*(1+protocol['contingency_fraction'])
    return result
