"""Optional KBS per-document sidecar. Inactive for legacy RAGTree runs.

Each document is assigned to a deterministic process shard; committed predictions
are immutable checkpoints. An exception is recorded and the next document proceeds
until the campaign-wide terminal-error threshold is exceeded.
"""
from __future__ import annotations
import contextvars
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import traceback
from typing import Any, Callable

ACTIVE_DOC: contextvars.ContextVar[str | None] = contextvars.ContextVar('kbs_active_doc', default=None)

def active() -> bool:
    return os.environ.get('KBS_ACTIVE') == '1'

def doc_id(doc: dict) -> str:
    value = doc.get('document_id') or doc.get('id')
    if value is None:
        raise ValueError('KBS requires a stable document_id or id; refusing to infer an unstable ID')
    return str(value)

def _digest(value: str) -> str:
    return hashlib.sha256(value.encode('utf8')).hexdigest()[:24]

def _root() -> Path:
    return Path(os.environ['KBS_RUN_ROOT'])

def _connect():
    c = sqlite3.connect(os.environ['KBS_DB'], timeout=30)
    c.execute('PRAGMA busy_timeout=30000')
    c.execute('PRAGMA journal_mode=WAL')
    return c

def init_db(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=30) as c:
        c.execute('PRAGMA journal_mode=WAL')
        c.execute('''CREATE TABLE IF NOT EXISTS records (
          run_id TEXT NOT NULL, doc_id TEXT NOT NULL, status TEXT NOT NULL,
          elapsed REAL, detail TEXT, updated REAL NOT NULL,
          PRIMARY KEY(run_id,doc_id))''')
        c.execute('''CREATE TABLE IF NOT EXISTS failures (
          session_id TEXT NOT NULL, run_id TEXT NOT NULL, doc_id TEXT NOT NULL,
          detail TEXT NOT NULL, updated REAL NOT NULL,
          PRIMARY KEY(session_id,run_id,doc_id))''')

def atomic_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf8', dir=path.parent, prefix='.tmp-', suffix='.json', delete=False) as f:
        tmp = Path(f.name)
        json.dump(obj, f, ensure_ascii=False, default=str)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def record_path(run_root: Path, identifier: str) -> Path:
    return run_root / 'records' / f'{_digest(identifier)}.json'

def call_path(identifier: str, number: int) -> Path:
    return _root() / 'calls' / _digest(identifier) / f'{os.environ.get("KBS_SESSION_ID","session")}-{os.getpid()}-{number:05}.json'

_CALL_COUNTER = {}
_EVENT_COUNTER = {}
def trace_event(kind: str, data: Any):
    if not active():return
    identifier=ACTIVE_DOC.get()
    if identifier is None:return
    n=_EVENT_COUNTER.get(identifier,0)+1
    _EVENT_COUNTER[identifier]=n
    path=_root()/'events'/_digest(identifier)/f'{os.environ.get("KBS_SESSION_ID","session")}-{os.getpid()}-{n:05}.json'
    atomic_json(path,{'document_id':identifier,'event':kind,'data':data,'timestamp':time.time()})

def trace_call(request: dict, response: dict | None, elapsed: float, error: str | None = None):
    if not active():
        return
    identifier = ACTIVE_DOC.get()
    if identifier is None:
        return
    n = _CALL_COUNTER.get(identifier, 0) + 1
    _CALL_COUNTER[identifier] = n
    atomic_json(call_path(identifier, n), {'document_id': identifier, 'request': request,
                'response': response, 'elapsed_sec': elapsed, 'error': error,
                'timestamp': time.time()})

def should_process(doc: dict) -> bool:
    if not active():
        return True
    identifier = doc_id(doc)
    shard_count = int(os.environ.get('KBS_SHARDS', '1'))
    shard_index = int(os.environ.get('KBS_SHARD', '0'))
    if int(_digest(identifier), 16) % shard_count != shard_index:
        return False
    select_path = os.environ.get('KBS_SELECTED_IDS')
    if select_path:
        selected = _load_selected(select_path)
        if identifier not in selected:
            return False
    with _connect() as c:
        row = c.execute('SELECT status FROM records WHERE run_id=? AND doc_id=?',
                        (os.environ['KBS_RUN_ID'], identifier)).fetchone()
    if row and row[0] == 'done':
        file = record_path(_root(), identifier)
        if file.exists():
            try:
                data = json.loads(file.read_text(encoding='utf8'))
                if data.get('document_id') == identifier and data.get('run_id') == os.environ['KBS_RUN_ID']:
                    return False
            except (ValueError, OSError):
                pass
    if row and row[0] == 'failed' and os.environ.get('KBS_RETRY_FAILED') != '1':
        return False
    return True

_SELECTED_CACHE = {}
def _load_selected(path: str):
    if path not in _SELECTED_CACHE:
        _SELECTED_CACHE[path] = set(json.loads(Path(path).read_text(encoding='utf8')))
    return _SELECTED_CACHE[path]

class CampaignAbort(RuntimeError):
    pass

def invoke(fn: Callable, doc: dict, *args, **kwargs):
    """Run a legacy strategy, preserve its original prediction return value.
    Failures return {} for legacy script continuation, but are excluded from
    success checkpoints and counted as terminal errors in the DB.
    """
    if not active():
        return fn(doc, *args, **kwargs)
    identifier = doc_id(doc)
    # Controlled protocol: no evaluation gold passed into a predictor. Keep it
    # in the saved source_document for scoring. Fixed relation schema is external.
    if os.environ.get('KBS_PROTOCOL_MODE') == 'controlled_revision_v2':
        safe_doc = dict(doc)
        safe_doc.pop('relations', None)
        doc = doc  # original remains the immutable output/evaluation reference
        kwargs = dict(kwargs)
        args = tuple(args)
        schema = os.environ.get('KBS_SCHEMA_FILE')
        if not schema:
            raise RuntimeError('controlled mode requires KBS_SCHEMA_FILE')
        kwargs['relation_types'] = json.loads(Path(schema).read_text(encoding='utf8'))
        shots=kwargs.get('few_shots')
        eval_ids_file=os.environ.get('KBS_EVAL_IDS_FILE')
        if isinstance(shots,list) and eval_ids_file:
            eval_ids=_load_selected(eval_ids_file)
            kwargs['few_shots']=[shot for shot in shots if str(shot.get('document_id') or shot.get('id')) not in eval_ids]
    token = ACTIVE_DOC.set(identifier)
    start = time.monotonic()
    run_id = os.environ['KBS_RUN_ID']
    try:
        result = fn(safe_doc if os.environ.get('KBS_PROTOCOL_MODE') == 'controlled_revision_v2' else doc, *args, **kwargs)
        if not isinstance(result, dict):
            raise TypeError(f'expected dict of relations, got {type(result).__name__}')
        elapsed = time.monotonic() - start
        trace_event('final_prediction',{'relation_counts':{k:len(v) if isinstance(v,list) else None for k,v in result.items()}})
        record = {'run_id': run_id, 'document_id': identifier, 'elapsed_sec': elapsed,
                  'pred_relations': result, 'source_document': doc, 'status': 'done',
                  'timestamp': time.time()}
        atomic_json(record_path(_root(), identifier), record)
        with _connect() as c:
            c.execute('INSERT OR REPLACE INTO records VALUES (?,?,?,?,?,?)',
                      (run_id, identifier, 'done', elapsed, None, time.time()))
        return result
    except Exception:
        elapsed = time.monotonic() - start
        detail = traceback.format_exc()[-12000:]
        with _connect() as c:
            c.execute('INSERT OR REPLACE INTO records VALUES (?,?,?,?,?,?)',
                      (run_id, identifier, 'failed', elapsed, detail, time.time()))
            c.execute('INSERT OR REPLACE INTO failures VALUES (?,?,?,?,?)',
                      (os.environ['KBS_SESSION_ID'], run_id, identifier, detail, time.time()))
            errors = c.execute('SELECT COUNT(*) FROM failures WHERE session_id=?',
                               (os.environ['KBS_SESSION_ID'],)).fetchone()[0]
        atomic_json(_root()/'errors'/f'{_digest(identifier)}.json',
                    {'run_id':run_id,'document_id': identifier, 'elapsed_sec':elapsed,'exception':detail})
        if errors > int(os.environ.get('KBS_MAX_ERRORS','3')):
            raise CampaignAbort(f'KBS terminal errors {errors} exceed limit') from None
        return {}
    finally:
        ACTIVE_DOC.reset(token)


def status_counts(db: Path, run_ids: list[str] | None = None) -> dict:
    with sqlite3.connect(db) as c:
        query = 'SELECT status, COUNT(*) FROM records'
        params: list = []
        if run_ids is not None:
            if not run_ids:
                return {}
            query += f" WHERE run_id IN ({','.join('?' for _ in run_ids)})"
            params = run_ids
        query += ' GROUP BY status'
        return dict(c.execute(query,params).fetchall())
