#!/usr/bin/env python3
"""Read-only: diagnose KBS smoke failures and source ID collisions."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default='.')
    parser.add_argument('--dataset', default='data/preprocessed/fincausal.jsonl')
    parser.add_argument('--max-groups', type=int, default=8)
    args = parser.parse_args()
    root = Path(args.repo).resolve()
    path = root / args.dataset
    if not path.is_file():
        print('Dataset not found:', path)
        return 2
    docs = defaultdict(list)
    total = 0
    with path.open(encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            total += 1
            key = str(obj.get('document_id') or obj.get('id'))
            signature = hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
            docs[key].append((lineno, obj, signature))
    collisions = [(k, v) for k, v in docs.items() if len(v) > 1]
    print(f'Dataset: {path}\nRows: {total}; unique IDs: {len(docs)}; duplicate-ID groups: {len(collisions)}')
    for key, group in collisions[:args.max_groups]:
        print('\nDuplicate ID:', repr(key))
        for line, obj, signature in group:
            rel = obj.get('relations') or {}
            rel_counts = {k: len(v) for k, v in rel.items() if isinstance(v, list)} if isinstance(rel, dict) else '?'
            print(f'  line={line} type={obj.get("type")} title={str(obj.get("title"))[:40]!r} '
                  f'content_hash={signature} relation_counts={rel_counts} text={str(obj.get("text", ""))[:100]!r}')
        print('  All source records identical:', len({signature for _, _, signature in group}) == 1)
    if len(collisions) > args.max_groups:
        print(f'... {len(collisions) - args.max_groups} additional duplicate-ID groups omitted')
    db = root / 'results/kbs_journal/checkpoints.sqlite'
    print('\n=== Most recent failed record checkpoints ===')
    if db.is_file():
        with sqlite3.connect(db) as connection:
            rows = connection.execute('''SELECT run_id, doc_id, updated, detail
              FROM records WHERE status='failed' ORDER BY updated DESC LIMIT 5''').fetchall()
        if not rows:
            print('No failed document checkpoints')
        for rid, doc_id, when, detail in rows:
            print(f'\nrun={rid}, document={doc_id}, timestamp={when}')
            print((detail or '[no traceback]')[-2500:])
    else:
        print('No checkpoint DB found:', db)
    print('\n=== Recent baseline/MAVEN shard logs ===')
    p = root / 'results/kbs_journal/local/main_gpt/baseline/maven_ere'
    logs = sorted(p.glob('*/logs/shard_*.log'), key=lambda x:x.stat().st_mtime, reverse=True)
    for log in logs[:2]:
        print('\n', log)
        print(log.read_text(errors='replace')[-2500:])
    if not logs: print('No matching logs found')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
