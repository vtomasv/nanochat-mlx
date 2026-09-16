"""Read-only FP32 dictionary lower-bound census; no training or codec benchmark.

This diagnostic rejects only the local per-block exact-value-dictionary format.
An unrejected block is not evidence that a grammar can compress it.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import struct
import subprocess

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def census(path, rows_per_block):
    with path.open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        if size > (1 << 24):
            raise ValueError('Unexpected safetensors header size')
        header = json.loads(stream.read(size))
    records = []
    skipped = []
    for name, meta in sorted(header.items()):
        if name == '__metadata__':
            continue
        shape = meta['shape']
        if meta['dtype'] != 'F32' or len(shape) != 2:
            skipped.append({'tensor': name, 'dtype': meta['dtype'], 'shape': shape})
            continue
        start, end = meta['data_offsets']
        if end - start != math.prod(shape) * 4 or 8 + size + end > path.stat().st_size:
            raise ValueError('Invalid tensor extent')
        bits = np.memmap(path, dtype='<u4', mode='r', offset=8 + size + start, shape=tuple(shape))
        for row in range(0, shape[0], rows_per_block):
            block = np.asarray(bits[row:row + rows_per_block])
            special = int(np.count_nonzero((block == 0x80000000) | ((block & 0x7f800000) == 0x7f800000)))
            nz = block[block != 0]
            distinct = len(np.unique(nz))
            # Header plus exact FP32 dictionary; intentionally omits all rules,
            # column IDs, row delimiters, alignment, and container overhead.
            lower = 64 + 4 * distinct
            records.append({
                'file': path.name, 'tensor': name, 'row_start': row,
                'rows': block.shape[0], 'cols': block.shape[1],
                'cells': block.size, 'positive_zero_cells': block.size - nz.size,
                'nonzero_bit_patterns': distinct, 'special_cells': special,
                'dense_bytes': block.nbytes, 'dictionary_header_lower_bound_bytes': lower,
                'rejected_by_dictionary_bound': lower * 100 > block.nbytes * 95,
            })
        del bits
    return records, skipped


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rows-per-block', type=int, default=128)
    args = parser.parse_args()
    if args.rows_per_block < 1:
        parser.error('rows-per-block must be positive')
    expected = json.loads((args.checkpoint / 'hashes.json').read_text())
    hashes = {}
    for name in ('model.safetensors', 'optimizer.safetensors'):
        path = args.checkpoint / name
        hashes[name] = digest(path)
        matches = [h for p, h in expected.items() if Path(p).name == name]
        if matches != [hashes[name]]:
            raise ValueError(f'Historical hash mismatch: {name}')
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {
        'kind': 'exploratory_exact_dictionary_bound_census',
        'checkpoint': str(args.checkpoint), 'checkpoint_hashes': hashes,
        'repo_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'script_sha256': digest(Path(__file__)), 'python': platform.python_version(),
        'numpy': np.__version__, 'rows_per_block': args.rows_per_block,
        'threshold_ratio': 0.95, 'files': {},
        'limitations': [
            'Only FP32 matrices and local dictionaries of the stated block size.',
            'No grammar construction, latency measurement, or global impossibility claim.',
            'An unrejected block remains undecided; special IEEE values require RAW.',
            'One existing checkpoint, no independent training replicas.',
        ],
    }
    for name in hashes:
        records, skipped = census(args.checkpoint / name, args.rows_per_block)
        with (args.output / (name + '.csv')).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        rejected = [r for r in records if r['rejected_by_dictionary_bound']]
        summary['files'][name] = {
            'matrices': len({r['tensor'] for r in records}), 'blocks': len(records),
            'rejected_blocks': len(rejected),
            'dense_bytes': sum(r['dense_bytes'] for r in records),
            'dense_bytes_in_rejected_blocks': sum(r['dense_bytes'] for r in rejected),
            'special_cells': sum(r['special_cells'] for r in records), 'skipped': skipped,
        }
    summary['csv_sha256'] = {p.name: digest(p) for p in sorted(args.output.glob('*.csv'))}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary['files'], indent=2))


if __name__ == '__main__':
    main()
