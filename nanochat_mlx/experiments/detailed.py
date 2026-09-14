"""Explicit barriers for phase diagnostics; never used as principal timing."""
import time
from pathlib import Path
import numpy as np
from .runner import build
from .replay import load_checkpoint,checkpoint_path
from .reporting import Run
from .measure import setup_device,memory,RSSSampler
from .config import dump

def profile(c,arm):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map
    from nanochat_mlx.gpt import loss_fn
    from nanochat_mlx.optim import get_lr_multiplier,get_muon_momentum,get_weight_decay
    from .activation_tape import loss_and_grad
    setup_device();checkpoint=checkpoint_path(c,'primary-native-reference-s17',100)
    experiment='E2' if arm.startswith('staged_') else 'E1'
    run=Run(experiment,c,arm,'detailed',17,f'primary-detailed-{arm}')
    model,opt=build(c,17,arm);load_tick=time.perf_counter();load_checkpoint(checkpoint,model,opt,c)
    run.event('load_and_conversion',seconds=time.perf_counter()-load_tick,**memory())
    tx=np.load(Path(c['artifact_dir'])/'train_x.npy',mmap_mode='r');ty=np.load(Path(c['artifact_dir'])/'train_y.npy',mmap_mode='r')
    phases={};acc=None;losses=[]
    if arm.startswith('staged_'):
        original_encode=opt._encode;original_decode=opt._decode
        def encode(path,array):
            t=time.perf_counter();h=original_encode(path,array);phases['nested_encode_seconds']=phases.get('nested_encode_seconds',0)+time.perf_counter()-t;return h
        def decode(handle):
            t=time.perf_counter();a=original_decode(handle);mx.eval(a);mx.synchronize();phases['nested_decode_seconds']=phases.get('nested_decode_seconds',0)+time.perf_counter()-t;return a
        opt._encode=encode;opt._decode=decode
    mx.reset_peak_memory();started=time.perf_counter()
    with RSSSampler() as rss:
        for micro in range(c['accumulation']):
            index=100*c['accumulation']+micro;x=mx.array(tx[index]);y=mx.array(ty[index])
            if arm.startswith('tape_'):
                loss,g=loss_and_grad(model,x,y,arm,c['rank_sample_bits'],profile=True)
                for key,value in model._navarro_profile.items():phases[key]=phases.get(key,0)+value
            else:
                t=time.perf_counter();loss,g=nn.value_and_grad(model,loss_fn)(model,x,y);mx.eval(loss);mx.synchronize();phases['forward_and_graph_construction_seconds']=phases.get('forward_and_graph_construction_seconds',0)+time.perf_counter()-t
                t=time.perf_counter();mx.eval(g);mx.synchronize();phases['backward_materialization_seconds']=phases.get('backward_materialization_seconds',0)+time.perf_counter()-t
            acc=g if acc is None else tree_map(lambda a,b:a+b,acc,g);mx.eval(acc);losses.append(loss.item())
        acc=tree_map(lambda g:g*(1./c['accumulation']),acc);mx.eval(acc);mx.synchronize()
        opt.set_lr_multiplier(get_lr_multiplier(100,1000,0,.5,0));opt.set_muon_momentum(get_muon_momentum(100));opt.set_muon_weight_decay(get_weight_decay(100,1000,opt.config.weight_decay))
        t=time.perf_counter();opt.update(model,acc);mx.eval(model.parameters(),*opt.state);mx.synchronize();phases['update_including_state_codec_seconds']=time.perf_counter()-t
    phases['total_seconds']=time.perf_counter()-started
    run.event('explicit_barrier_phases',**phases)
    run.row('steps.csv',dict(step=100,loss=float(np.mean(losses)),seconds=phases['total_seconds'],tokens=c['total_batch_size'],**memory()))
    run.finish(status='measured',decision='INCONCLUSIVE',intervention={'phases':phases,'process_tree_rss_peak_bytes':rss.peak,**memory()},limitations=['Detailed cold profile with explicit barriers; excluded from principal timing ratios.','Nested pack/unpack/decode/encode times overlap their parent phase and must not be added again.','Additional attention/FC forward counts are summed across both microbatches.'])
    return run.root
