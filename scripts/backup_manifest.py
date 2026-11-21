"""Prepare a stable log snapshot and verify a complete downloaded backup."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def prepare(root, logs, destination):
    root, logs, destination = Path(root), Path(logs), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    log_copy = destination / ('logs_' + str(time.time_ns()))
    shutil.copytree(logs, log_copy, ignore=shutil.ignore_patterns('*.lock', '*.pid'))
    folders = {name: root / 'FADE' / name for name in
               ('output', 'input', 'inversion', 'mask', 'saig_out', 'masks')}
    folders['Annotations'] = root / 'DAVIS/Annotations/480p'
    folders['logs'] = log_copy
    entries = []
    for label, directory in folders.items():
        files = sorted(p for p in directory.rglob('*') if p.is_file())
        if not files:
            raise ValueError(f'Backup source empty or missing: {directory}')
        for file in files:
            if file.is_symlink():
                raise ValueError(f'Unexpected symlink: {file}')
            stat = file.stat()
            checksum = digest(file)
            after = file.stat()
            if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f'File changed during backup preparation: {file}')
            entries.append(dict(path=(PurePosixPath(label) / file.relative_to(directory).as_posix()).as_posix(),
                                size=stat.st_size, sha256=checksum))
    manifest = dict(version=1, folders={k: str(v) for k, v in folders.items()}, files=entries)
    path = destination / 'manifest.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)
    print('BACKUP_MANIFEST_READY', path, len(entries))


def verify(manifest_path, destination):
    destination = Path(destination).resolve()
    receipt = destination / 'BACKUP_VERIFIED.json'
    receipt.unlink(missing_ok=True)
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8-sig'))
    if manifest.get('version') != 1 or not manifest.get('files'):
        raise ValueError('Empty or unsupported backup manifest')
    seen = set()
    for entry in manifest['files']:
        rel = PurePosixPath(entry['path'])
        if rel.is_absolute() or '..' in rel.parts or ':' in str(rel) or '\\' in str(rel) or str(rel) in seen:
            raise ValueError('Invalid or duplicate backup path')
        seen.add(str(rel))
        path = (destination / str(rel)).resolve()
        if not path.is_relative_to(destination) or path.stat().st_size != entry['size'] or digest(path) != entry['sha256']:
            raise ValueError(f'Backup integrity mismatch: {rel}')
    receipt.write_text(json.dumps(dict(files=len(seen), manifest_sha256=digest(manifest_path),
                                      verified_at=time.time()), indent=2), encoding='utf-8')
    print('BACKUP_VERIFIED', len(seen))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--root', required=True)
    p.add_argument('--logs', required=True)
    p.add_argument('--destination', required=True)
    v = sub.add_parser('verify')
    v.add_argument('--manifest', required=True)
    v.add_argument('--destination', required=True)
    a = ap.parse_args()
    if a.command == 'prepare':
        prepare(a.root, a.logs, a.destination)
    else:
        verify(a.manifest, a.destination)
