"""Validate saved stages and metrics before advancing the experiment."""
import argparse
import csv
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile


def ledger_success(path, seq, stage):
    with open(path, encoding='utf-8-sig') as stream:
        matches = [r for r in csv.DictReader(stream) if r['seq'] == seq and r['stage'] == stage]
    if not matches or matches[-1]['rc'] != '0':
        raise ValueError(f'{seq}/{stage}: latest ledger attempt is not successful')


def validate_tensor(path, kind):
    import torch
    value = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(value, (list, tuple)):
        value = torch.stack(value)
    expected = (51, 1, 13, 16, 60, 90) if kind == 'inversion' else (49, 480, 720)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
        raise ValueError(f'{path}: expected tensor shape {expected}')
    for frame in value:
        if not torch.isfinite(frame).all().item():
            raise ValueError(f'{path}: nonfinite tensor')
    if kind == 'mask' and not ((value == 0) | (value == 1)).all().item():
        raise ValueError(f'{path}: expected binary FADE mask')


def validate_video(path):
    data = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
        '-show_entries', 'stream=width,height,nb_read_frames,avg_frame_rate', '-of', 'json', str(path)]))
    stream = data['streams'][0]
    if (stream['width'], stream['height'], int(stream['nb_read_frames']), Fraction(stream['avg_frame_rate'])) != (720, 480, 49, 4):
        raise ValueError(f'{path}: unexpected video format {stream}')
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path), '-f', 'null', '-'], check=True)


def merge_metrics(paths, seqs, out):
    from metrics_compare import PROMPTS
    rows = []
    for path, seq in zip(paths, seqs):
        with open(path, encoding='utf-8-sig') as stream:
            part = list(csv.DictReader(stream))
        if len(part) != 3 or {r['video'] for r in part} != {'b0', 'gate', 'center'}:
            raise ValueError(f'{seq}: require three unique arms')
        source, target = PROMPTS[seq]
        for row in part:
            if row['seq'] != seq or row['init_prompt'] != 'a ' + source or row['target_prompt'] != 'a ' + target:
                raise ValueError(f'{seq}: wrong metric prompts or sequence')
            if int(row['n']) != 13 or not all(math.isfinite(float(row[k])) for k in ('ID_S', 'Sem_T_target', 'Sem_T_init')):
                raise ValueError(f'{seq}: incomplete or nonfinite metric')
        rows.extend(part)
    dest = Path(out)
    fd, tmp = tempfile.mkstemp(prefix=dest.name, suffix='.tmp', dir=dest.parent)
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    print(f'METRICS_VERIFIED rows={len(rows)}')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    stage = sub.add_parser('stage')
    stage.add_argument('--ledger', required=True)
    stage.add_argument('--seq', required=True)
    stage.add_argument('--stage', required=True)
    stage.add_argument('--path', required=True)
    stage.add_argument('--kind', choices=['inversion', 'mask', 'video'], required=True)
    tensor = sub.add_parser('tensor')
    tensor.add_argument('--path', required=True)
    tensor.add_argument('--kind', choices=['inversion', 'mask'], required=True)
    video = sub.add_parser('video')
    video.add_argument('--path', required=True)
    metrics = sub.add_parser('metrics')
    metrics.add_argument('--directory', required=True)
    metrics.add_argument('--seqs', nargs='+', required=True)
    metrics.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.cmd == 'metrics':
        merge_metrics([Path(args.directory) / f'metrics_{s}.csv' for s in args.seqs], args.seqs, args.out)
        return
    if args.cmd == 'stage':
        ledger_success(args.ledger, args.seq, args.stage)
    if args.cmd == 'video' or args.kind == 'video':
        validate_video(args.path)
    else:
        validate_tensor(args.path, args.kind)
    print('ASSET_VERIFIED', args.path)


if __name__ == '__main__':
    main()
