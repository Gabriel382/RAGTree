"""One bounded real subprocess integration test (no inference network)."""
import os
from pathlib import Path
import subprocess
import sys
import json
import pytest
from ragtree.experiments.kbs.protocol import ROOT

@pytest.mark.parametrize('mode,workers', [('sequential',1),('parallel',5)])
def test_baseline_shards_and_resume(tmp_path,mode,workers):
    data=ROOT/'data/preprocessed/maven_ere.jsonl'
    if data.exists():pytest.skip('preserve user dataset; test only on a checkout without real records')
    data.parent.mkdir(parents=True,exist_ok=True)
    try:
        data.write_text(''.join(json.dumps({'document_id':f'synthetic-{i}', 'type':'test',
           'sentences':['A affects B'],'title':'A','entities':{},'relations':{'CAUSE':[]}})+'\n' for i in range(6)),encoding='utf8')
        env=os.environ.copy()
        env['PYTHONPATH']=str(ROOT/'tests/kbs/fixtures/fake_litellm')+os.pathsep+str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
        cmd=[sys.executable,str(ROOT/'scripts/kbs_protocol.py'),'run','--phase','main_gpt','--method','baseline',
             '--dataset','maven_ere','--profile','local','--results-root',str(tmp_path),
             '--mode',mode,'--workers',str(workers),'--max-shard-seconds','30','--allow-count-mismatch']
        for i in range(2):
            r=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=40)
            assert r.returncode==0,(r.stdout,r.stderr)
        import sqlite3
        with sqlite3.connect(tmp_path/'checkpoints.sqlite') as c:
            done=c.execute('SELECT COUNT(*) FROM records WHERE status="done"').fetchone()[0]
        assert done==6
        calls=list(tmp_path.rglob('calls/*/*.json'))
        assert len(calls)==6, 'successful docs must be skipped, never billed again on resume'
    finally:
        data.unlink(missing_ok=True)

def test_threshold_abort(tmp_path):
    data=ROOT/'data/preprocessed/maven_ere.jsonl'
    if data.exists():pytest.skip('preserve existing dataset')
    data.parent.mkdir(parents=True,exist_ok=True)
    try:
        data.write_text(''.join(json.dumps({'document_id':f'forced-fail-{i}','type':'test','title':'',
             'sentences':['a'],'entities':{},'relations':{'CAUSE':[]}})+'\n' for i in range(6)),encoding='utf8')
        env=os.environ.copy();env['KBS_TEST_FORCE_ERROR']='1'
        env['PYTHONPATH']=str(ROOT/'tests/kbs/fixtures/fake_litellm')+os.pathsep+str(ROOT)
        cmd=[sys.executable,str(ROOT/'scripts/kbs_protocol.py'),'run','--phase','main_gpt',
            '--method','baseline','--dataset','maven_ere','--profile','local','--results-root',str(tmp_path),
            '--max-errors','3','--allow-count-mismatch','--max-shard-seconds','30']
        run=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=40)
        assert run.returncode!=0,(run.stdout,run.stderr)
        with __import__('sqlite3').connect(tmp_path/'checkpoints.sqlite') as c:
            errors=c.execute("SELECT COUNT(*) FROM failures WHERE doc_id NOT LIKE '__shard_%'").fetchone()[0]
            completed=c.execute('SELECT COUNT(*) FROM records WHERE status="done"').fetchone()[0]
        assert errors==4 and completed==0
    finally:
        data.unlink(missing_ok=True)


def test_duplicates_end_to_end_controlled_without_merging(tmp_path):
    """Use fake LiteLLM and six duplicate-ID rows; no external inference."""
    from ragtree.experiments.kbs.record_identity import UID_KEY
    data=ROOT/'data/preprocessed/maven_ere.jsonl'
    if data.exists():
        pytest.skip('preserve user dataset; integration tests require synthetic data')
    data.parent.mkdir(parents=True, exist_ok=True)
    original=[
        {'document_id':'same','type':'test','title':f'row-{i}',
         'sentences':['a causes b'],'entities':{},'relations':{'CAUSE':[]}}
        for i in range(6)]
    try:
        data.write_text(''.join(json.dumps(x)+'\n' for x in original),encoding='utf8')
        env=os.environ.copy()
        env['PYTHONPATH']=str(ROOT/'tests/kbs/fixtures/fake_litellm')+os.pathsep+str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
        cmd=[sys.executable,str(ROOT/'scripts/kbs_protocol.py'),'run','--phase','main_gpt','--method','baseline',
             '--dataset','maven_ere','--profile','local','--results-root',str(tmp_path),
             '--protocol','controlled_revision_v2','--mode','parallel','--workers','5',
             '--max-shard-seconds','30','--allow-count-mismatch']
        for _ in range(2):
            r=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=40)
            assert r.returncode==0,(r.stdout,r.stderr)
        import sqlite3
        with sqlite3.connect(tmp_path/'checkpoints.sqlite') as c:
            assert c.execute('SELECT COUNT(DISTINCT doc_id) FROM records WHERE status="done"').fetchone()[0]==6
        records=list(tmp_path.rglob('records/*.json'))
        assert len(records)==6
        assert {json.loads(p.read_text())['source_document']['title'] for p in records} == {f'row-{i}' for i in range(6)}
        assert all(UID_KEY not in json.loads(p.read_text())['source_document'] for p in records)
        assert len(list(tmp_path.rglob('calls/*/*.json')))==6
    finally:
        data.unlink(missing_ok=True)
