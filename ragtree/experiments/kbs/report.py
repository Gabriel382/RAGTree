"""Gold-driven scoring, per-label metrics, cost/trace summary, source comparison."""
from __future__ import annotations
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
from typing import Any
from .engine import filtered_docs, OUT
from .protocol import load_protocol
from .estimate import scan_records, calls_for, cost_per_record
from .runtime import atomic_json
from ragtree.evaluation.relations.metrics import RelationEvalAggregator, relations_to_triples


def _bootstrap_ci(docs:list[tuple[dict,dict]], n=250,seed=42):
    if not docs:return None
    rng=random.Random(seed)
    counts=[]
    for gold,pred in docs:
        g=relations_to_triples(gold,ignore_labels={'null'})
        p=relations_to_triples(pred,ignore_labels={'null'})
        tp=len(g & p);fp=len(p-g);fn=len(g-p)
        counts.append((tp,fp,fn))
    values=[];L=len(counts)
    for _ in range(n):
        tp=fp=fn=0
        for a,b,c in rng.choices(counts,k=L):tp+=a;fp+=b;fn+=c
        values.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    values.sort()
    return [values[int(.025*(n-1))],values[int(.975*(n-1))]]


def evaluate_run(job,run_root:Path):
    cache={r['document_id']:r for r in scan_records(run_root)}
    agg=RelationEvalAggregator(ignore_labels={'null'})
    observations=[]
    missing=[]
    for doc in filtered_docs(job):
        identifier=str(doc.get('document_id') or doc.get('id'))
        rec=cache.get(identifier)
        if rec is None:missing.append(identifier)
        pred=rec.get('pred_relations',{}) if rec else {}
        gold=doc.get('relations',{})
        agg.update(gold_relations=gold,pred_relations=pred)
        observations.append((gold,pred))
    result=agg.compute_metrics()
    labels=result['per_relation']
    result['macro_f1']=sum(x['f1'] for x in labels.values())/len(labels) if labels else 0.0
    result['bootstrap_ci_micro_f1']=_bootstrap_ci(observations)
    result['coverage']={'expected':len(observations),'successful':len(observations)-len(missing),
                        'missing':len(missing),'missing_ids':missing[:200]}
    result['complete']=not missing
    result['run_root']=str(run_root)
    return result


def report_one(job,run_root:Path,with_scoring=True):
    result={'job':job.identity,'phase':job.phase,'method':job.method,'dataset':job.dataset,
            'model':job.model,'variant':job.variant,'status':'no_data'}
    records=list(scan_records(run_root))
    if not records:return result
    result['status']='partial'
    prices=load_protocol()['price_per_million_usd']
    costs=[cost_per_record(run_root,r,job.model,prices) for r in records]
    result['record_count']=len(records)
    result['llm_calls']=sum(v['llm_calls'] for v in costs)
    for metric in ('input_tokens','output_tokens','provider_cost_usd','ratecard_cost_usd','runtime_sec'):
        values=[v[metric] for v in costs if v[metric] is not None]
        result[metric]=sum(values) if len(values)==len(costs) else None
    if with_scoring:
        try:
            scores=evaluate_run(job,run_root)
            result['metrics']=scores
            result['status']='complete' if scores['complete'] else 'partial'
        except (FileNotFoundError, ValueError, KeyError) as exc:
            result['score_error']=repr(exc)
    atomic_json(run_root/'metrics.json',result)
    return result


def dataset_characteristics(job):
    from collections import Counter
    counts=Counter();entity_counts=[];density=[];cross_known=0;cross_true=0
    for doc in filtered_docs(job):
        rels=relations_to_triples(doc.get('relations',{}),ignore_labels={'null'})
        density.append(len(rels))
        ents=doc.get('entities',{})
        if not isinstance(ents,dict):ents={}
        entity_counts.append(len(ents))
        for rel,h,t in rels:
            counts[rel]+=1
            if h in ents and t in ents:
                def sentence_ids(e):
                    return {m['sent_id'] for m in e.get('mentions',[]) if isinstance(m,dict) and m.get('sent_id') is not None}
                try:
                    hids=sentence_ids(ents[h]); tids=sentence_ids(ents[t])
                except (TypeError,AttributeError):continue
                if hids and tids:
                    cross_known+=1
                    if not (hids & tids):cross_true+=1
    return {'dataset':job.dataset,'records':len(density),
            'gold_relations':sum(density),
            'gold_relations_per_record':sum(density)/len(density) if density else None,
            'entities_per_record':sum(entity_counts)/len(entity_counts) if entity_counts else None,
            'relation_types':len(counts),'label_support':dict(sorted(counts.items())),
            'cross_sentence_comparable_gold':cross_known,
            'cross_sentence_fraction_known_only':cross_true/cross_known if cross_known else None,
            'cross_sentence_note':'Requires both gold endpoints to carry sentence IDs; undefined pairs excluded.'}


def report_campaign(rows:list[dict],out:Path):
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'metrics.json',rows)
    csv_fields=['phase','method','dataset','model','variant','status','record_count','llm_calls',
                'input_tokens','output_tokens','provider_cost_usd','ratecard_cost_usd','runtime_sec',
                'micro_f1','macro_f1','coverage']
    with (out/'metrics.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=csv_fields);writer.writeheader()
        for r in rows:
            metrics=r.get('metrics') or {}
            vals={k:r.get(k) for k in csv_fields}
            vals['micro_f1']=(metrics.get('micro') or {}).get('f1')
            vals['macro_f1']=metrics.get('macro_f1')
            vals['coverage']=(metrics.get('coverage') or {}).get('successful')
            writer.writerow(vals)
    with (out/'per_relation.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=['phase','method','dataset','model','relation','tp','fp','fn','precision','recall','f1','support_gold','support_pred'])
        writer.writeheader()
        for r in rows:
            for label,data in (r.get('metrics') or {}).get('per_relation',{}).items():
                writer.writerow({**{k:r.get(k) for k in ('phase','method','dataset','model')},'relation':label,**data})
    baseline={(r['method'],r['dataset']):r for r in rows if r.get('phase')=='main_gpt'}
    from .protocol import recipes
    with (out/'regression.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=['method','dataset','paper_f1_pct','current_f1_pct','delta_pp','complete','note'])
        writer.writeheader()
        for q in recipes():
            item=baseline.get((q['method'],q['dataset']))
            score=((item or {}).get('metrics') or {}).get('micro',{}).get('f1')
            complete=(item or {}).get('status')=='complete'
            writer.writerow({'method':q['method'],'dataset':q['dataset'],'paper_f1_pct':q['paper_f1'],
               'current_f1_pct':round(100*score,4) if complete and score is not None else '',
               'delta_pp':round(100*score-q['paper_f1'],4) if complete and score is not None else '',
               'complete':complete,
               'note':'Comparison requires same document IDs, source resources, model/provider and protocol; differences are not automatically regressions.'})
    props={}
    for r in rows:
        key=r.get('dataset')
        if key in props or r.get('status')=='no_data':continue
        # Dataset properties loaded only when gold inputs exist.
        from .protocol import job_matrix
        job=next((j for j in job_matrix() if j.method==r.get('method') and j.dataset==key),None)
        if job:
            try:props[key]=dataset_characteristics(job)
            except (OSError,ValueError,KeyError):pass
    atomic_json(out/'dataset_properties.json',props)
    (out/'reviewer_evidence.md').write_text(
        '# Experiment evidence to support response to reviewers\n\n'
        '- Additional backbone: `metrics.csv`, `metrics.json` (both `main_*` phases)\n'
        '- Robustness: `prompt_robustness` and `execution_repeats` phase records\n'
        '- Context budget: `budget_control` request/event traces, verify evidence limits and measured counts\n'
        '- Retrieval depth: `retrieval_sensitivity` parameter variations\n'
        '- Failure diagnoses: per-call `calls/`, `events/`, error traces; full stage attribution needs method-specific retrieval candidate instrumentation\n'
        '- Per-relation/macro statistics: `per_relation.csv` and run `metrics.json`\n'
        '- Dataset properties: `dataset_properties.json` (cross-sentence partial when IDs missing)\n'
        '- Costs: run token summaries and local estimation output; local runtime cannot replace OpenRouter billed expense\n'
        '- Historical comparison: `regression.csv`, only valid on matched protocols/providers\n'
        '- Survey search, selection and method-adaptation fidelity require separate manuscript documentation.\n',
        encoding='utf8')
    # Only complete evaluations are exported to submission-ready tables.
    def tex_escape(v):
        return str(v).replace('\\','\\textbackslash{}').replace('_','\\_').replace('&','\\&').replace('%','\\%')
    line_end = ' ' + '\\\\' + '\n'
    with (out/'metrics_table.tex').open('w',encoding='utf8') as f:
        f.write('% Auto-generated by KBS RAGTree; completed cells only\n')
        f.write('\\begin{tabular}{llllrr}\n\\hline\n')
        f.write('Phase & Dataset & Method & Model & Micro-F1 (\\%) & Macro-F1 (\\%)'+line_end+'\\hline\n')
        for r in rows:
            if r.get('status')!='complete':continue
            metric=r['metrics']
            f.write(' & '.join(tex_escape(r.get(k,'')) for k in ('phase','dataset','method','model'))+
                    f" & {100*metric['micro']['f1']:.2f} & {100*metric['macro_f1']:.2f}"+line_end)
        f.write('\\hline\n\\end{tabular}\n')
    incomplete=sum(r['status']!='complete' for r in rows)
    summary=(f'# KBS experiment evidence report\n\n'
             f'- Configurations reported: {len(rows)}\n'
             f'- Incomplete/blocked/no data: {incomplete}\n'
             f'- A complete result requires all expected evaluation IDs.\n'
             f'- Compare paper scores in configs/experiments/kbs_journal/recipes.json.\n'
             f'- Do not present partial metrics as complete journal evidence.\n\n'
             f'Files: `metrics.json`, `metrics.csv`; each run has `metrics.json` and `calls/`.\n')
    (out/'README.md').write_text(summary,encoding='utf8')
    return out
