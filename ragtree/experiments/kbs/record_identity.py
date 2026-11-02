"""Non-destructive, deterministic KBS record identities for JSONL datasets.

This module exists exclusively for KBS journal checkpointing. It leaves all
original ``document_id`` values intact so KG and ontology lookups are unchanged.
When duplicate IDs exist, a derived *input cache* annotates only colliding rows
with ``__kbs_record_uid``. That private field must be removed before LLM calls.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile

UID_KEY = '__kbs_record_uid'


def checkpoint_id(document: dict) -> str:
    original = document.get('document_id') or document.get('id')
    if original is None:
        raise ValueError('KBS needs document_id or id for every record')
    return str(document.get(UID_KEY) or original)


def build_input_cache(source: Path, cache_root: Path) -> Path:
    """Copy the JSONL only if IDs collide, preserving all rows and original IDs.

    The content-derived cache filename makes concurrent launch safe and ensures
    changed input bytes cannot silently reuse old checkpoint identities.
    Construction uses an atomic rename; no originals are ever modified.
    """
    source = Path(source)
    cache_root = Path(cache_root)
    hasher = hashlib.sha256()
    identifiers = []
    rows = 0
    with source.open('rb') as f:
        for number, line in enumerate(f, 1):
            hasher.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = row.get('document_id') or row.get('id')
            if identifier is None:
                raise ValueError(f'{source}:{number}: missing document ID')
            if UID_KEY in row:
                raise ValueError(f'{source}:{number}: {UID_KEY} is reserved by KBS')
            identifiers.append(str(identifier))
            rows += 1
    counts = Counter(identifiers)
    collision_rows = sum(v for v in counts.values() if v > 1)
    if not collision_rows:
        return source

    # Filename depends on file bytes AND source location, not only basename.
    full_sha = hasher.hexdigest()
    path_sha = hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:10]
    target = cache_root / (f'{source.stem}-{path_sha}-{full_sha[:24]}.kbs.jsonl')
    if target.is_file():
        return target

    cache_root.mkdir(parents=True, exist_ok=True)
    seen = set()
    actual = 0
    tmp = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf8', dir=cache_root,
                                         prefix='.kbs-', suffix='.jsonl', delete=False) as output:
            tmp = Path(output.name)
            with source.open(encoding='utf8') as original:
                for number, line in enumerate(original, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    identifier = str(row.get('document_id') or row.get('id'))
                    if counts[identifier] > 1:
                        # Original ID remains unchanged; use an unambiguous row key
                        # based on source line. It is stable while source bytes are.
                        row[UID_KEY] = f'kbs-dup:{full_sha[:12]}:{number}'
                    uid = checkpoint_id(row)
                    if uid in seen:
                        raise ValueError(f'{source}:{number}: non-unique generated checkpoint ID {uid}')
                    seen.add(uid)
                    actual += 1
                    output.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')
            output.flush()
            os.fsync(output.fileno())
        if actual != rows or len(seen) != rows:
            raise AssertionError(f'KBS cache count mismatch ({actual} != {rows})')
        # Rename after fully verifying. Another process may create the same
        # byte-equivalent cache; the first/last complete file is still valid.
        os.replace(tmp, target)
        return target
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()
