"""Frozen KBS experiment recipes and phase expansion, without GPU side effects."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any
import yaml

ROOT=Path(__file__).resolve().parents[3]
CFG_DIR=ROOT/'configs/experiments/kbs_journal'

@dataclass(frozen=True)
class Job:
    phase: str
    method: str
    dataset: str
    model: str
    variant: str
    arguments: tuple[str,...]
    source: str
    provenance: str

    @property
    def identity(self):
        blob=json.dumps(asdict(self),sort_keys=True).encode()
        return sha256(blob).hexdigest()[:20]


def load_protocol():
    p=CFG_DIR/'protocol.yaml'
    result=yaml.safe_load(p.read_text(encoding='utf8'))
    if not result or result['schema_version'] != 1:
        raise ValueError('Invalid KBS protocol schema')
    return result


def recipes():
    return json.loads((CFG_DIR/'recipes.json').read_text(encoding='utf8'))


def _override_args(base: list[str], keys: dict[str,str]):
    args=list(base)
    for key,val in keys.items():
        flag='--'+key.replace('_','-')
        if flag in args:
            i=args.index(flag)
            if i+1 >=len(args) or args[i+1].startswith('--'):
                raise ValueError('Bad historical flag '+flag)
            args[i+1]=str(val)
        else:
            args.extend([flag,str(val)])
    return args


def job_matrix(profile: str | None=None):
    protocol=load_protocol()
    originals=recipes()
    by_key={(r['method'],r['dataset']):r for r in originals}
    jobs=[]
    def add(phase, methods, datasets, model, variant='native', variations=None):
        for method in methods:
            for dataset in datasets:
                r=by_key[(method,dataset)]
                args=_override_args(r['args'],variations.get(method,{}) if variations else {})
                jobs.append(Job(phase,method,dataset,model,variant,tuple(args),r['source'],r['provenance']))
    all_methods=protocol['methods']
    all_datasets=protocol['datasets']
    primary='gpt_oss_20b'; secondary='qwen3_30b_a3b'
    add('main_gpt',all_methods,all_datasets,primary)
    add('main_qwen',all_methods,all_datasets,secondary)
    for i in (1,2):
        add('prompt_robustness',protocol['robust_methods'],protocol['robust_datasets'],primary,'prompt_v'+str(i))
    for model in (primary,secondary):
        add('budget_control',protocol['budget_methods'],protocol['budget_datasets'],model,'context_budget_2048')
    for index in (3,10):
        changes={
          'chunk_orag': {'top_k':index},
          'community_kgrag': {'top_communities':index},
          'triple_kg_rag': {'kg_max_triples':index*24},
        }
        add('retrieval_sensitivity',protocol['sensitivity_methods'],protocol['sensitivity_datasets'],primary,'depth_'+str(index),changes)
    for i in (1,2):
        add('execution_repeats',protocol['repeat_methods'],protocol['repeat_datasets'],primary,'repeat_'+str(i))
    keys=[j.identity for j in jobs]
    if len(keys)!=len(set(keys)):
        raise AssertionError('Duplicate experimental identities')
    return jobs


def profile_config(profile_name: str):
    data=load_protocol()['profiles']
    if profile_name not in data: raise KeyError(f'Unknown profile {profile_name}. Allowed: {list(data)}')
    return data[profile_name]


def select_jobs(jobs,phase=None,method=None,dataset=None,model=None,maximum=None):
    out=[j for j in jobs if (not phase or j.phase==phase) and (not method or j.method==method) and (not dataset or j.dataset==dataset) and (not model or j.model==model)]
    return out[:maximum] if maximum is not None else out


def output_config(job:Job,run_root:Path,profile:dict,shard:int=0):
    """Write a derived YAML, never alter xquality's original configs/default.yaml."""
    cfg=yaml.safe_load((ROOT/'configs/default.yaml').read_text(encoding='utf8'))
    cfg['paths']['data_processed']=str((run_root/'legacy_shards'/f'{shard}').resolve())
    cfg['paths']['results']=str((run_root/'legacy_shards'/f'{shard}'/'results').resolve())
    model_id=profile['models'][job.model]
    for llm_section,value in cfg.get('llm',{}).items():
        if not isinstance(value,dict) or 'backends' not in value: continue
        default=value['backends'].get('vllm',{})
        value['backends']['litellm']={
           'model':model_id,
           'temperature':default.get('temperature',0.0),
           'max_tokens':default.get('max_tokens',1024)}
    if job.variant.startswith('prompt_v'):
        suffix=('\nReturn only a machine-readable relation object.' if job.variant=='prompt_v1'
                 else '\nCheck each proposed relation against the document before producing the final JSON.')
        for section in cfg.get('prompts',{}).values():
            if not isinstance(section,dict):continue
            for k,v in section.items():
                if isinstance(v,str):section[k]=v+suffix
    # A prompt ceiling is NOT equivalent to budget-matched external evidence.
    # That experimental phase is explicitly gated until adapters can enforce it.
    path=run_root/f'overlay_shard_{shard}.yaml'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf8')
    return path
