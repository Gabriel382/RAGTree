from __future__ import annotations
import ast
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3
import pytest
from ragtree.experiments.kbs.protocol import ROOT, job_matrix, recipes, select_jobs
from ragtree.experiments.kbs.engine import SCRIPT_MAP
from ragtree.experiments.kbs import runtime
from ragtree.experiments.kbs.context_budget import enforce_external_evidence_budget
from ragtree.experiments.kbs.estimate import sample_ids


def test_exact_matrix():
    jobs=job_matrix()
    assert len(jobs)==192
    assert len({j.identity for j in jobs})==192
    assert Counter(j.phase for j in jobs)=={'main_gpt':60,'main_qwen':60,
       'prompt_robustness':24,'budget_control':18,'retrieval_sensitivity':18,'execution_repeats':12}
    assert len(recipes())==60
    assert len({(j.method,j.dataset) for j in jobs if j.phase=='main_gpt'})==60


def test_all_legacy_recipe_flags_exist():
    flags={}
    for method,script in SCRIPT_MAP.items():
        tree=ast.parse((ROOT/'scripts'/script).read_text())
        allowed=set()
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument':
                allowed.update(x.value for x in node.args if isinstance(x,ast.Constant) and isinstance(x.value,str))
        flags[method]=allowed
    for recipe in recipes():
        for a in recipe['args']:
            if a.startswith('--'):assert a in flags[recipe['method']],(recipe['method'],a)


@pytest.fixture
def checkpoint_env(tmp_path,monkeypatch):
    db=tmp_path/'state.sqlite'
    runtime.init_db(db)
    values={'KBS_ACTIVE':'1','KBS_DB':str(db),'KBS_RUN_ROOT':str(tmp_path),
        'KBS_RUN_ID':'trial123','KBS_SESSION_ID':'session123','KBS_SHARDS':'1',
        'KBS_SHARD':'0','KBS_MAX_ERRORS':'3'}
    for k,v in values.items():monkeypatch.setenv(k,v)
    return tmp_path,db


def test_checkpoint_commit_resume(checkpoint_env):
    root,db=checkpoint_env
    doc={'document_id':'one','relations':{'CAUSE':[['E1','E2']]}}
    assert runtime.should_process(doc)
    result=runtime.invoke(lambda d,relation_types=None:{'CAUSE':[['E1','E2']]},doc,relation_types=['CAUSE'])
    assert result['CAUSE']==[['E1','E2']]
    assert not runtime.should_process(doc)
    assert runtime.status_counts(db)=={'done':1}
    data=json.loads(runtime.record_path(root,'one').read_text())
    assert data['source_document']['relations']['CAUSE']==[['E1','E2']]


def test_failed_records_skipped_and_abort_after_fourth(checkpoint_env,monkeypatch):
    _,db=checkpoint_env
    def fail(_):raise RuntimeError('forced failure')
    for n in range(3):
        doc={'document_id':f'bad{n}'}
        assert runtime.invoke(fail,doc)=={}
        assert not runtime.should_process(doc)
    with pytest.raises(runtime.CampaignAbort):runtime.invoke(fail,{'document_id':'bad3'})
    assert runtime.status_counts(db)=={'failed':4}
    monkeypatch.setenv('KBS_RETRY_FAILED','1')
    assert runtime.should_process({'document_id':'bad0'})


def test_deterministic_disjoint_sharding(checkpoint_env,monkeypatch):
    ids=[{'document_id':f'd-{n}'} for n in range(100)]
    seen=[]
    monkeypatch.setenv('KBS_SHARDS','5')
    for shard in range(5):
        monkeypatch.setenv('KBS_SHARD',str(shard))
        selection={d['document_id'] for d in ids if runtime.should_process(d)}
        seen.append(selection)
    assert set.union(*seen)=={d['document_id'] for d in ids}
    assert all(not (seen[i]&seen[j]) for i in range(5) for j in range(i+1,5))


def test_controlled_protocol_strips_gold(checkpoint_env,monkeypatch):
    root,_=checkpoint_env
    monkeypatch.setenv('KBS_PROTOCOL_MODE','controlled_revision_v2')
    (root/'schema.json').write_text('["CAUSE","PRECONDITION"]')
    (root/'eval_ids.json').write_text('["target","other"]')
    monkeypatch.setenv('KBS_SCHEMA_FILE',str(root/'schema.json'))
    monkeypatch.setenv('KBS_EVAL_IDS_FILE',str(root/'eval_ids.json'))
    doc={'document_id':'target','relations':{'CAUSE':[['E1','E2']]},'title':'A'}
    def predict(d,relation_types,few_shots):
        assert 'relations' not in d
        assert relation_types==['CAUSE','PRECONDITION']
        assert [x['document_id'] for x in few_shots]==['train']
        return {'CAUSE':[]}
    assert runtime.invoke(predict,doc,relation_types=['CAUSE'],few_shots=[{'document_id':'other'},{'document_id':'train'}])=={'CAUSE':[]}
    saved=json.loads(runtime.record_path(root,'target').read_text())
    assert 'relations' in saved['source_document']


@pytest.mark.parametrize('method,text',[
 ('growlrag','head\n## Ontology-guided retrieval context\n'+('a'*1000)+'\n## Output format\nTAIL'),
 ('community_kgrag','head\n## Tool: CommunityKG-RAG evidence sentences (retrieved from corpus via KG communities)\n'+('a'*1000)+'\n## Output format\nTAIL'),
 ('langgraph_agentic_simple','head\n## Tool: Web context\n'+('a'*1000)+'\n## Tool: Wikidata context\n'+('b'*1000)+'\n## Output format\nTAIL'),
])
def test_budget_preserves_non_evidence(method,text):
    messages,stats=enforce_external_evidence_budget([{'role':'user','content':text}],method,100,lambda t:len(t)//4)
    assert stats['after_tokens']<=100 and stats['before_tokens']>stats['after_tokens']
    assert messages[0]['content'].startswith('head\n') and messages[0]['content'].endswith('## Output format\nTAIL')


def test_budget_refuses_unrecognized_boundaries():
    with pytest.raises(ValueError):enforce_external_evidence_budget([{'role':'user','content':'hello'}],'growlrag',200,lambda t:len(t)//4)


def test_sample_nested(monkeypatch):
    from ragtree.experiments.kbs import estimate
    fake=[{'document_id':f'{i:03}','text':'x'*i,'relations':{'A':[['1','2']]* (i%3)}} for i in range(90)]
    monkeypatch.setattr(estimate,'filtered_docs',lambda job:iter(fake))
    a=sample_ids(None,6);b=sample_ids(None,18)
    assert b[:6]==a
    assert len(set(b))==18


def test_duplicate_checkpoint_identity_keeps_original_ids_and_annotations(tmp_path):
    from ragtree.experiments.kbs.record_identity import build_input_cache, checkpoint_id, UID_KEY
    source=tmp_path/'mixed.jsonl'
    records=[
        {'document_id':'duplicate','title':'A','type':'test','relations':{'CAUSE':[['e1','e2']]}},
        {'document_id':'unique','title':'U','type':'test','relations':{'CAUSE':[]}},
        {'document_id':'duplicate','title':'B','type':'test','relations':{'CAUSE':[['e4','e5']]}},
    ]
    source.write_text(''.join(json.dumps(x)+'\n' for x in records),encoding='utf8')
    cache=build_input_cache(source,tmp_path/'cache')
    assert cache!=source
    derived=[json.loads(s) for s in cache.read_text().splitlines()]
    assert [x['document_id'] for x in derived]==['duplicate','unique','duplicate']
    assert len({checkpoint_id(x) for x in derived})==3
    assert UID_KEY not in derived[1]
    assert [{k:v for k,v in x.items() if k!=UID_KEY} for x in derived]==records
    assert build_input_cache(source,tmp_path/'cache')==cache
    assert source.read_text().count('duplicate')==2
    source.write_text(source.read_text()+json.dumps({'document_id':'next'})+'\n')
    assert build_input_cache(source,tmp_path/'cache')!=cache


def test_checkpoint_uid_never_reaches_model(checkpoint_env,monkeypatch):
    from ragtree.experiments.kbs.record_identity import UID_KEY
    root,db=checkpoint_env
    monkeypatch.setenv('KBS_PROTOCOL_MODE','controlled_revision_v2')
    (root/'schema.json').write_text('["CAUSE"]')
    monkeypatch.setenv('KBS_SCHEMA_FILE',str(root/'schema.json'))
    doc={'document_id':'shared',UID_KEY:'checkpoint-2','text':'original','relations':{'CAUSE':[]}}
    def predictor(safe_doc,relation_types):
        assert safe_doc == {'document_id':'shared','text':'original'}
        assert relation_types==['CAUSE']
        return {'CAUSE':[]}
    runtime.invoke(predictor,doc,relation_types=['CAUSE'])
    saved=json.loads(runtime.record_path(root,'checkpoint-2').read_text())
    assert saved['document_id']=='checkpoint-2'
    assert saved['source_document']=={'document_id':'shared','text':'original','relations':{'CAUSE':[]}}
    assert not runtime.should_process(doc)
