"""Bounded CPU codec diagnostic against externally built author executables.

Does not modify the v1 experiment, train models, or benchmark inference latency.
Author sources/executables stay under ignored sources/, not in this repository.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time

import numpy as np
from nanochat_mlx.experiments.grammar_matrix import encode, decode_exact, resident_bytes

ROOT = Path(__file__).resolve().parents[1]
REV = 'edd85fa3193dd1e6e60e1d4886f1ad8ded2a138f'


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def block(path, name, start=0):
    # Read only the first 128 rows of a saved FP32 tensor, without loading MLX.
    with path.open('rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))[name]
    assert header['dtype'] == 'F32' and len(header['shape']) == 2
    view = np.memmap(path, dtype='<f4', mode='r', offset=8+n+header['data_offsets'][0], shape=tuple(header['shape']))
    return np.array(view[start:start+128], copy=True)


def expand_author(base, shape):
    vals = np.fromfile(str(base)+'.fval', dtype='<f4')
    raw = np.fromfile(str(base)+'.vc.R', dtype='<i4')
    alpha = int(raw[0]); rules = raw[1:].reshape(-1, 2)
    seq = np.fromfile(str(base)+'.vc.C', dtype='<i4')
    pieces = []
    for i, pair in enumerate(rules):
        assert all(0 < s < alpha+i for s in pair)
        pieces.append(np.concatenate([np.array([s], np.int64) if s < alpha else pieces[s-alpha] for s in pair]))
    if len(rules):
        seq = np.concatenate([np.array([s], np.int64) if s < alpha else pieces[s-alpha] for s in seq])
    ends = seq == 0
    assert ends.sum() == shape[0] and ends[-1]
    row = np.cumsum(ends)-ends
    codes = seq[~ends].astype(np.int64)-1
    out = np.zeros(shape, np.float32)
    out[row[~ends], codes % shape[1]] = vals[codes // shape[1]]
    return out, {'values': len(vals), 'rules': len(rules), 'symbols': (Path(str(base)+'.vc.C').stat().st_size//4), 'alpha': alpha}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--run-id', default='mm-repair-edd85fa3-probe')
    p.add_argument('--adam-only', action='store_true')
    a = p.parse_args(); upstream = a.upstream.resolve()
    work = ROOT/'experiments/navarro/artifacts'/a.run_id
    results = ROOT/'experiments/navarro/verification'/a.run_id
    work.mkdir(parents=True, exist_ok=False); results.mkdir(parents=True, exist_ok=False)
    reference = ROOT/'experiments/navarro/artifacts/primary/checkpoints/primary-native-reference-s17'
    cases = []
    for step in (100, 500, 1000):
        path = reference/f'step-{step:04d}/optimizer.safetensors'
        for layer in ('c_fc', 'c_proj'):
            name = f'muon.blocks.4.mlp.{layer}.weight'
            cases.append((f'muon-{layer}-{step}', block(path, name), {'checkpoint': str(path.relative_to(ROOT)), 'tensor': name, 'row_slice': [0,128]}))
    path = reference/'step-1000/model.safetensors'
    for layer in ('c_fc', 'c_proj'):
        name = f'blocks.4.mlp.{layer}.weight'
        cases.append((f'weight-{layer}-1000', block(path, name), {'checkpoint': str(path.relative_to(ROOT)), 'tensor': name, 'row_slice': [0,128]}))
    if a.adam_only:
        cases=[]
        path=reference/'step-1000/optimizer.safetensors'
        for name in ('adam.lm_head.weight.v','adam.wte.weight.m'):
            for start in (0,16384,32640):
                cases.append((f'{name}-{start}',block(path,name,start),{'checkpoint':str(path.relative_to(ROOT)),'tensor':name,'row_slice':[start,start+128]}))
    else:
        rng = np.random.default_rng(20260914)
        cases += [('synthetic-repeated', np.tile(rng.integers(0,4,size=(1,512)).astype(np.float32), (128,1)), {'synthetic': True}),
                  ('synthetic-random', rng.normal(size=(128,512)).astype(np.float32), {'synthetic': True})]
    record = {'upstream_commit': REV, 'repo_sha': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'scope': 'Explicit 128-row slices recorded per case; 3 sequential construction repetitions; no training/inference benchmark.',
              'timing_caveat': 'Author path includes two CLI process launches and intermediate file IO; our path is an in-memory library call. Diagnostic ratios are not integrated speedups.',
              'upstream_sources_sha256': {str(p.relative_to(upstream)): digest(p) for p in upstream.rglob('*') if p.is_file() and p.suffix in ('.c','.h','.cpp','.hpp','.md')},
              'executables_sha256': {x:digest(upstream/x) for x in ('bin2csrvf','brepair/irepair0')}, 'cases': []}
    (results/'probe_script.py').write_bytes(Path(__file__).read_bytes())
    record['probe_script_sha256']=digest(results/'probe_script.py')
    for label, matrix, origin in cases:
        assert np.isfinite(matrix).all() and not np.any(matrix.view(np.uint32)==0x80000000)
        base = work/(label+'.f32'); matrix.tofile(base)
        commands = [[str(upstream/'bin2csrvf'), str(base), *map(str,matrix.shape)],
                    [str(upstream/'brepair/irepair0'), str(base)+'.vc', '256']]
        author_times=[]; own_times=[]
        for rep in range(3):
            # Alternate execution order; three runs remain a small diagnostic.
            for who in (('author','own') if rep%2==0 else ('own','author')):
                tick = time.perf_counter()
                if who == 'author':
                    for ci, command in enumerate(commands):
                        with (results/f'{label}-{rep}-{ci}.log').open('w') as log:
                            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
                    author_times.append(time.perf_counter()-tick)
                else:
                    own = encode(matrix, 're_32', 128)
                    own_times.append(time.perf_counter()-tick)
        restored, stats = expand_author(base, matrix.shape)
        assert restored.tobytes()==matrix.tobytes() and decode_exact(own).tobytes()==matrix.tobytes()
        iv = encode(matrix, 're_iv', 128)
        author_bytes=sum(Path(str(base)+ext).stat().st_size for ext in ('.fval','.vc.R','.vc.C'))
        # With no rules, dictionary + distinct terminals at the C stream's width
        # is a conservative bound; excludes repeated symbols, separators, headers.
        vc = np.fromfile(str(base)+'.vc', dtype='<u4')
        terminals=np.unique(vc[vc!=0]); min_width=int(terminals[-1]).bit_length() if len(terminals) else 1
        lower=stats['values']*4+(len(terminals)*min_width+7)//8
        row={'case':label,'origin':origin,'shape':matrix.shape,'input_sha256':digest(base),'dense_bytes':matrix.nbytes,
             'author_re32_payload_bytes':author_bytes,'author_re32_payload_ratio':author_bytes/matrix.nbytes,
             'author_stats':stats,'author_construction_seconds':author_times,'own_construction_seconds':own_times,
             'own_re32':resident_bytes(own),'own_reiv':resident_bytes(iv),'bit_exact_both':True,
             'author_fixed_width_payload_lower_bound_ratio':lower/matrix.nbytes if stats['rules']==0 else None,
             'commands':commands}
        record['cases'].append(row)
        (results/'probe.json').write_text(json.dumps(record,indent=2)+'\n')
        print(f'{label}: author re32={author_bytes/matrix.nbytes:.3f}x, own reiv={resident_bytes(iv)["payload_bytes"]/matrix.nbytes:.3f}x; construction author={np.median(author_times):.4f}s own={np.median(own_times):.4f}s; exact',flush=True)
    print(results/'probe.json')


if __name__=='__main__':main()
