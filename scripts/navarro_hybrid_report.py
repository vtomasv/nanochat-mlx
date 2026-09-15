"""Validate saved hybrid evidence and compare final states without retraining."""
import json
from pathlib import Path
import struct

import numpy as np

from nanochat_mlx.experiments.config import ROOT, dump, sha256
from nanochat_mlx.experiments.grammar_matrix import deserialize, decode_exact
from nanochat_mlx.experiments.metrics import errors


def safetensors(path):
    with path.open('rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        meta = json.loads(f.read(n))
    return {k: np.memmap(path, mode='r', dtype={'F32': '<f4', 'I32': '<i4'}[v['dtype']],
                         shape=tuple(v['shape']), offset=8+n+v['data_offsets'][0])
            for k, v in meta.items() if k != '__metadata__'}


def optimizer(root):
    if (root / 'optimizer.safetensors').exists():
        yield from safetensors(root / 'optimizer.safetensors').items()
        return
    manifest = json.loads((root / 'handles/handles.json').read_text())
    for path, state in manifest['states'].items():
        for key, entry in state.items():
            name = f'muon.{path}' if key == 'buf' else f'adam.{path}.{key}'
            if 'integer' in entry:
                value = np.array(entry['integer'], np.int32)
            elif entry['kind'] == 'grammar':
                value = decode_exact(deserialize(root / 'handles' / entry['file']))
            else:
                value = np.load(root / 'handles' / entry['file'], allow_pickle=False)
            yield name, value


def compare(left, right):
    result = {}
    for group in ('parameters', 'optimizer'):
        a = safetensors(left / 'model.safetensors') if group == 'parameters' else dict(optimizer(left))
        b = safetensors(right / 'model.safetensors') if group == 'parameters' else dict(optimizer(right))
        assert a.keys() == b.keys()
        rows = {}
        for k in a:
            assert a[k].shape == b[k].shape and a[k].dtype == b[k].dtype
            rows[k] = {'bit_exact': a[k].tobytes() == b[k].tobytes(),
                       **errors(a[k], b[k], 1e-5, 1e-3)}
        result[group] = {'tensor_count': len(rows), 'bit_exact_all': all(r['bit_exact'] for r in rows.values()),
                         'tolerance_pass_all': all(r['pass'] for r in rows.values()), 'per_tensor': rows}
        del a, b
    return result


def reproducibility(run_id='hybrid-repair-v2'):
    root = ROOT / 'experiments/navarro/verification' / run_id
    first = json.loads((root / 'train-native.json').read_text())
    second = json.loads((root.parent / (run_id+'-native-repeat') / 'train-native.json').read_text())
    assert first['source_checkpoint_hashes'] == second['source_checkpoint_hashes']
    assert first['config'] == second['config']
    path = root / 'training-reproducibility.json'
    if path.exists():
        raise FileExistsError(path)
    result = compare(Path(first['final_checkpoint']), Path(second['final_checkpoint']))
    result['nll_delta'] = second['quality']['nll'] - first['quality']['nll']
    result['scope'] = 'Two native processes, identical initial checkpoint and ten replay updates; diagnostic only'
    result['evidence'] = [str(root / 'train-native.json'), str(root.parent / (run_id+'-native-repeat') / 'train-native.json')]
    dump(path, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('parameters', 'optimizer')}, indent=2))
    for group in ('parameters', 'optimizer'):
        print(group, result[group]['bit_exact_all'], result[group]['tolerance_pass_all'])


def main(run_id='hybrid-repair-v2'):
    root = ROOT / 'experiments/navarro/verification' / run_id
    if (root / 'comparison.json').exists():
        raise FileExistsError('Comparison already exists; preserve the original diagnostic')
    codec = json.loads((root / 'codec.json').read_text())
    inference = json.loads((root / 'inference.json').read_text())
    paths = {arm: root / f'train-{arm}.json' for arm in ('baseline', 'hybrid', 'raw', 'native')}
    # Reuse the measured baseline process: its loaded codec is fixed at 7787ae2,
    # independent of which candidate source was also present during compilation.
    if not paths['baseline'].exists():
        paths['baseline'] = root.parent / 'hybrid-repair-v1/train-baseline.json'
    runs = {arm: json.loads(path.read_text()) for arm, path in paths.items()}
    assert len(codec['cases']) == 32 and all(c['payload_and_products_identical'] for c in codec['cases'])
    assert inference['pass'] and len(inference['prompts']) == 8
    for record in (codec, inference, runs['hybrid']):
        assert record['hybrid_source_sha256'] == codec['hybrid_source_sha256']
    for arm, run in runs.items():
        path = Path(run['final_checkpoint'])
        for name, digest in run['final_checkpoint_hashes'].items():
            assert sha256(path / name) == digest
        for name, digest in json.loads((path / 'hashes.json').read_text()).items():
            assert sha256(name) == digest
        assert run['source_checkpoint_hashes'] == runs['baseline']['source_checkpoint_hashes']
        assert [s['step'] for s in run['steps']] == list(range(100, 110))
    comparisons = {}
    for arm in ('baseline', 'raw', 'native'):
        key = f'hybrid_vs_{arm}'
        comparisons[key] = compare(Path(runs['hybrid']['final_checkpoint']), Path(runs[arm]['final_checkpoint']))
        comparisons[key]['nll_delta'] = runs['hybrid']['quality']['nll'] - runs[arm]['quality']['nll']
        print(key, {k: v['bit_exact_all'] for k, v in comparisons[key].items() if isinstance(v, dict)}, flush=True)
    timings = {arm: r['mean_step_seconds'] for arm, r in runs.items()}
    direct = {arm: float(np.mean([t for p in inference['prompts'] for t in p['timings'][arm]['decode_seconds']]))
              for arm in ('baseline', 'hybrid', 'native')}
    result = {'scope': 'Engineering comparison; no confirmatory success or quality noninferiority claim',
              'training_evidence': {arm: str(path.relative_to(ROOT)) for arm, path in paths.items()},
              'training_mean_step_seconds': timings,
              'training_baseline_over_hybrid': timings['baseline'] / timings['hybrid'],
              'training_hybrid_over_native': timings['hybrid'] / timings['native'],
              'inference_mean_decode_seconds': direct,
              'checkpoint_comparisons': comparisons,
              'codec_cases_identical': 32, 'inference_logits_identical_prompts': 8}
    dump(root / 'comparison.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'checkpoint_comparisons'}, indent=2))


if __name__ == '__main__':
    main()
