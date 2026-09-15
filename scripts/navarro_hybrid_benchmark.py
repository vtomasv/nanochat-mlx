"""Paired codec diagnostic and separate-process training windows for hybrid RePair.

The baseline is our own committed v1 implementation, not upstream MM RePair.
No scientific v1 result is overwritten. Timings are engineering measurements.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

import numpy as np

from nanochat_mlx.experiments import grammar_matrix as gm
from nanochat_mlx.experiments.config import ROOT, dump, load_config, sha256
from scripts.navarro_mm_repair_probe import block

BASELINE = '7787ae2'
CODEC_SOURCE = 'nanochat_mlx/experiments/native.cpp'


def libraries(work):
    """Load both independent implementations with identical compiler options."""
    current = gm.library()
    old_source = subprocess.check_output(['git', 'show', f'{BASELINE}:{CODEC_SOURCE}'])
    path = work / 'baseline.cpp'
    path.write_bytes(old_source)
    binary = work / 'baseline.dylib'
    subprocess.run(['clang++', '-std=c++17', '-O3', '-ffp-contract=off',
                    '-dynamiclib', str(path), '-o', str(binary)], check=True)
    old = C.CDLL(str(binary))
    for name in ('ng_encode', 'ng_data', 'ng_free', 'ng_decode', 'ng_matvec'):
        getattr(old, name).argtypes = getattr(current, name).argtypes
        getattr(old, name).restype = getattr(current, name).restype
    return {'baseline': old, 'hybrid': current}


def provenance(results, name):
    paths = [p for folder in ('nanochat_mlx', 'scripts', 'configs/navarro')
             for p in (ROOT / folder).rglob('*') if p.suffix in ('.py', '.cpp', '.json')]
    with zipfile.ZipFile(results / f'{name}-sources.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, p.relative_to(ROOT))
        z.writestr('baseline/native.cpp', subprocess.check_output(['git', 'show', f'{BASELINE}:{CODEC_SOURCE}']))
    return {'baseline_commit': subprocess.check_output(['git', 'rev-parse', BASELINE], text=True).strip(),
            'repo_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'hybrid_source_sha256': sha256(ROOT / CODEC_SOURCE),
            'script_sha256': sha256(Path(__file__)),
            'source_snapshot_sha256': sha256(results / f'{name}-sources.zip'),
            'command': [sys.executable, '-m', 'scripts.navarro_hybrid_benchmark', *sys.argv[1:]],
            'platform': platform.platform(), 'numpy': np.__version__,
            'compiler': subprocess.check_output(['clang++', '--version'], text=True).strip()}


def codec(args, work, results):
    record = provenance(results, 'codec')
    libs = libraries(work)
    reference = ROOT / 'experiments/navarro/artifacts/primary/checkpoints/primary-native-reference-s17'
    cases = []
    for step in (100, 500, 1000):
        path = reference / f'step-{step:04d}/optimizer.safetensors'
        for layer in ('c_fc', 'c_proj'):
            name = f'muon.blocks.4.mlp.{layer}.weight'
            cases.append((f'{name}-{step}', block(path, name)))
    for name in ('adam.lm_head.weight.v', 'adam.wte.weight.m'):
        for start in (0, 16384, 32640):
            cases.append((f'{name}-{start}', block(reference / 'step-1000/optimizer.safetensors', name, start)))
    for layer in ('c_fc', 'c_proj'):
        name = f'blocks.4.mlp.{layer}.weight'
        cases.append((f'{name}-1000', block(reference / 'step-1000/model.safetensors', name)))
    rng = np.random.default_rng(20260914)
    cases += [('synthetic-repeated', np.tile(rng.integers(0, 4, (1, 512)).astype(np.float32), (128, 1))),
              ('synthetic-random', rng.normal(size=(128, 512)).astype(np.float32))]
    record.update(scope='14 trained 128-row blocks + 2 synthetic fixtures; five paired in-memory repetitions after warmup; no confidence interval', cases=[])
    for label, matrix in cases:
        for variant in ('adaptive', 're_iv'):
            times = {k: [] for k in libs}
            handles = {}
            for repeat in range(6):
                order = ('baseline', 'hybrid') if repeat % 2 == 0 else ('hybrid', 'baseline')
                for name in order:
                    gm._LIB = libs[name]
                    tick = time.perf_counter()
                    handles[name] = gm.encode(matrix, variant)
                    elapsed = time.perf_counter() - tick
                    if repeat:
                        times[name].append(elapsed)
            assert handles['baseline'].blocks == handles['hybrid'].blocks, (label, variant)
            products = {}
            for name in libs:
                gm._LIB = libs[name]
                assert gm.decode_exact(handles[name]).tobytes() == matrix.tobytes()
                products[name] = [gm.matvec(handles[name], np.ones(matrix.shape[1], np.float32)),
                                  gm.matvec(handles[name], np.ones(matrix.shape[0], np.float32), True)]
            assert all(x.tobytes() == y.tobytes() for x, y in zip(products['baseline'], products['hybrid']))
            row = {'case': label, 'variant': variant, 'shape': list(matrix.shape),
                   'input_sha256': hashlib.sha256(matrix.tobytes()).hexdigest(),
                   'seconds': times, 'payload_and_products_identical': True,
                   'hybrid_accounting': gm.resident_bytes(handles['hybrid']),
                   'baseline_over_hybrid_median': float(np.median(times['baseline']) / np.median(times['hybrid']))}
            record['cases'].append(row)
            print(f'{label} {variant}: {row["baseline_over_hybrid_median"]:.2f}x; identical', flush=True)
    dump(results / 'codec.json', record)


def train(args, work, results):
    name = f'train-{args.arm}'
    record = provenance(results, name)
    if args.arm in ('baseline', 'hybrid'):
        gm._LIB = libraries(work)[args.arm]
    c = load_config('configs/navarro/primary.json')
    os.environ['NANOCHAT_BASE_DIR'] = c['data_dir']
    import mlx.core as mx
    from nanochat_mlx.experiments.measure import setup_device, memory, RSSSampler, environment
    from nanochat_mlx.experiments.replay import load_checkpoint, save_checkpoint, verify_files
    from nanochat_mlx.experiments.runner import build, update, evaluate
    setup_device()
    tape = json.loads((Path(c['artifact_dir']) / 'tape.json').read_text())
    verify_files(tape['files'])
    assert tape['config_sha256'] == c['config_sha256']
    reference = Path(c['artifact_dir']) / 'checkpoints/primary-native-reference-s17/step-0100'
    arm = {'baseline': 'staged_grammar', 'hybrid': 'staged_grammar', 'raw': 'staged_raw', 'native': 'native'}[args.arm]
    x = np.load(Path(c['artifact_dir']) / 'train_x.npy', mmap_mode='r')
    y = np.load(Path(c['artifact_dir']) / 'train_y.npy', mmap_mode='r')

    def batches(step):
        for i in range(step * c['accumulation'], (step + 1) * c['accumulation']):
            yield mx.array(x[i]), mx.array(y[i])

    model, opt = build(c, 17, arm)
    state = load_checkpoint(reference, model, opt, c)
    start = state['step']
    tick = time.perf_counter()
    update(model, opt, batches(start), start, c, arm)
    record['discarded_warmup_seconds'] = time.perf_counter() - tick
    del model, opt
    model, opt = build(c, 17, arm)
    load_checkpoint(reference, model, opt, c)
    mx.reset_peak_memory()
    rows = []
    with RSSSampler() as rss:
        for step in range(start, start + args.steps):
            tick = time.perf_counter()
            loss = update(model, opt, batches(step), step, c, arm)
            seconds = time.perf_counter() - tick
            if rss.swap_violation:
                raise MemoryError('Sustained swap increase exceeds 1 GiB')
            rows.append({'step': step, 'loss': loss, 'seconds': seconds, **memory()})
            print(f'{args.arm} step={step+1} loss={loss:.7f} seconds={seconds:.4f}', flush=True)
    # Save weights and optimizer handles for cross-process comparisons and resume.
    dest = work / f'final-{args.arm}'
    save_checkpoint(dest, model, opt, start + args.steps, 17, c)
    record.update(scope='Engineering replay window, primary model; one fresh process per arm, no confirmatory CI',
                  config=c, environment=environment(), arm=args.arm, source_checkpoint=str(reference),
                  source_checkpoint_hashes={p.name: sha256(p) for p in reference.iterdir() if p.is_file()},
                  steps=rows, mean_step_seconds=float(np.mean([r['seconds'] for r in rows])),
                  quality=evaluate(model, c), final_checkpoint=str(dest),
                  final_checkpoint_hashes={p.name: sha256(p) for p in dest.iterdir() if p.is_file()},
                  last_accounting=getattr(opt, 'last_accounting', []),
                  process_tree_rss_peak_bytes=rss.peak, **memory())
    dump(results / f'{name}.json', record)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('command', choices=['codec', 'train', 'inference', 'report', 'reproducibility'])
    p.add_argument('--run-id', default='hybrid-repair-v2')
    p.add_argument('--arm', choices=['baseline', 'hybrid', 'raw', 'native'], default='hybrid')
    p.add_argument('--steps', type=int, default=10)
    args = p.parse_args()
    if args.command == 'reproducibility':
        from scripts.navarro_hybrid_report import reproducibility
        reproducibility(args.run_id)
        return
    if args.command == 'inference':
        from scripts.navarro_hybrid_inference import main as inference
        inference(args.run_id)
        return
    if args.command == 'report':
        from scripts.navarro_hybrid_report import main as report
        report(args.run_id)
        return
    if not 1 <= args.steps <= 100:
        p.error('Use a bounded engineering window of 1–100 steps')
    work = ROOT / 'experiments/navarro/artifacts' / args.run_id
    results = ROOT / 'experiments/navarro/verification' / args.run_id
    work.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (codec if args.command == 'codec' else train)(args, work, results)


if __name__ == '__main__':
    main()
