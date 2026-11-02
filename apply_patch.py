#!/usr/bin/env python3
"""Apply the KBS patch after checking original files and backups. Python stdlib only."""
import argparse,hashlib,json,os,shutil,sys,tempfile
from pathlib import Path
from datetime import datetime,timezone

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None

def main():
    ap=argparse.ArgumentParser(description='Apply RAGTree KBS journal patch')
    ap.add_argument('repo',nargs='?',default='.',help='checked-out RAGTree repository root')
    ap.add_argument('--force',action='store_true',help='allow upstream-different files, save existing as backups')
    ap.add_argument('--check',action='store_true',help='verify already installed files; do not write')
    args=ap.parse_args()
    root=Path(args.repo).resolve()
    pkg=Path(__file__).resolve().parent
    if not (root/'ragtree').is_dir() or not (root/'scripts').is_dir():
        ap.error('Target must be RAGTree repository root (missing ragtree/ or scripts/)')
    manifest=json.loads((pkg/'manifest.json').read_text())
    changes=[]; conflicts=[]
    for item in manifest['files']:
        rel=Path(item['path'])
        if rel.is_absolute() or '..' in rel.parts:
            ap.error(f'unsafe archive path {rel}')
        dest=(root/rel).resolve(); source=(pkg/'files'/rel).resolve()
        if not dest.is_relative_to(root) or not source.is_relative_to((pkg/'files').resolve()):
            ap.error(f'unsafe resolved path {rel}')
        if digest(source)!=item['patched_sha256']:
            ap.error(f'patch corrupted: {rel}')
        current=digest(dest)
        if current==item['patched_sha256']:continue
        if args.check:
            conflicts.append(f'{rel}: missing or modified')
            continue
        if current!=item['original_sha256'] and not args.force:
            conflicts.append(f'{rel}: upstream content differs, existing={str(current)[:12]}, expected={str(item["original_sha256"])[:12]}')
        changes.append((source,dest,current,rel))
    if conflicts:
        print('Preflight failed:');print('\n'.join('  '+x for x in conflicts))
        print('Patch was not applied. Ensure branch is xquality, or review differences and rerun with --force.')
        return 2
    if args.check:
        print(f'All {len(manifest["files"])} patch files match installed contents.')
        return 0
    if not changes:
        print(f'Already installed ({len(manifest["files"])} files matched); no changes.')
        return 0
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backups=root/'.kbs_patch_backups'/stamp
    written=[]
    try:
        for src,dst,old_hash,rel in changes:
            if dst.exists():
                backup=backups/rel
                backup.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(dst,backup)
            dst.parent.mkdir(parents=True,exist_ok=True)
            fd,tmp=tempfile.mkstemp(prefix='.kbs_',dir=dst.parent)
            try:
                with os.fdopen(fd,'wb') as fout,src.open('rb') as fin:
                    shutil.copyfileobj(fin,fout)
                    fout.flush();os.fsync(fout.fileno())
                os.replace(tmp,dst)
            finally:
                if os.path.exists(tmp):os.unlink(tmp)
            written.append(rel)
        print(f'Patched {len(written)} files in {root}. Previous versions backed up in {backups} (if any).')
        return 0
    except Exception as exc:
        print(f'Patch failed after writing {len(written)} files: {exc}; backups: {backups}',file=sys.stderr)
        return 1

if __name__=='__main__':sys.exit(main())
