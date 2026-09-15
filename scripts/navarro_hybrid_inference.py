"""Primary E3 regression: original codec vs hybrid, with native MLX control."""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from scripts.navarro_hybrid_benchmark import libraries, provenance
from nanochat_mlx.experiments.config import ROOT, load_config, dump
from nanochat_mlx.experiments import grammar_matrix as gm


def main(run_id='hybrid-repair-v2'):
    work = ROOT / 'experiments/navarro/artifacts' / run_id
    results = ROOT / 'experiments/navarro/verification' / run_id
    work.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    record = provenance(results, 'inference')
    libs = libraries(work)
    c = load_config('configs/navarro/primary.json')
    os.environ['NANOCHAT_BASE_DIR'] = c['data_dir']
    import mlx.core as mx
    from nanochat_mlx.engine import KVCache
    from nanochat_mlx.experiments.measure import setup_device, environment
    from nanochat_mlx.experiments.runner import build
    from nanochat_mlx.experiments.replay import load_inference_checkpoint
    from nanochat_mlx.experiments.grammar_linear import GrammarLinear
    from nanochat_mlx.experiments.metrics import errors
    setup_device()
    model, opt = build(c, 17)
    del opt
    checkpoint = Path(c['artifact_dir']) / 'checkpoints/primary-native-reference-s17/step-1000'
    state = load_inference_checkpoint(checkpoint, model, c)
    assert state['step'] == 1000 and state['seed'] == 17
    mlp = model.blocks[4].mlp
    layers = {'native': mlp.c_fc}
    conversion = {}
    for name, lib in libs.items():
        gm._LIB = lib
        tick = time.perf_counter()
        layers[name] = GrammarLinear(layers['native'].weight, variant='re_iv')
        conversion[name] = time.perf_counter() - tick
    assert layers['baseline']._handle.blocks == layers['hybrid']._handle.blocks
    x = np.load(Path(c['artifact_dir']) / 'val_x.npy', mmap_mode='r')
    for name in layers:
        gm._LIB = libs.get(name, libs['hybrid'])
        mlp.c_fc = layers[name]
        mx.eval(model(mx.array(x[32:33, :1]), kv_cache=KVCache(c['depth'], model.window_sizes)))
    record.update(environment=environment(), source_checkpoint=str(checkpoint),
                  checkpoint_hashes=json.loads((checkpoint / 'hashes.json').read_text()),
                  scope='Eight fixed 512-token prompts, 128 teacher-forced decode tokens each; sequential arms in one process; regression and descriptive timing, no confidence interval',
                  conversion_seconds=conversion, payload_identical=True, prompts=[])
    for p in range(8):
        seq = x[32+p]
        outputs = {}
        timings = {}
        for name in (('native', 'baseline', 'hybrid') if p % 2 == 0 else ('hybrid', 'baseline', 'native')):
            gm._LIB = libs.get(name, libs['hybrid'])
            mlp.c_fc = layers[name]
            cache = KVCache(c['depth'], model.window_sizes)
            values = []
            seconds = []
            for k in range(129):
                ids = seq[None, :512] if k == 0 else np.array([[seq[511+k]]])
                tick = time.perf_counter()
                logits = model(mx.array(ids), kv_cache=cache)
                mx.eval(logits)
                mx.synchronize()
                seconds.append(time.perf_counter() - tick)
                values.append(np.array(logits[0, -1], copy=True))
            outputs[name] = np.stack(values)
            timings[name] = {'prefill_seconds': seconds[0], 'decode_seconds': seconds[1:]}
            assert cache.offset == 640
        identical = outputs['baseline'].tobytes() == outputs['hybrid'].tobytes()
        err = errors(outputs['hybrid'], outputs['native'], 1e-4, 1e-3)
        assert identical and err['pass'], (p, identical, err)
        record['prompts'].append({'prompt': p, 'prompt_sha256': hashlib.sha256(seq[:512].tobytes()).hexdigest(),
                                  'timings': timings, 'baseline_hybrid_logits_bit_exact': identical,
                                  'native_errors': err,
                                  'output_sha256': {n: hashlib.sha256(a.tobytes()).hexdigest() for n, a in outputs.items()}})
        print(f'prompt {p+1}/8: baseline/hybrid logits exact; native tolerance passes', flush=True)
    record['pass'] = True
    dump(results / 'inference.json', record)


if __name__ == '__main__':
    main()
