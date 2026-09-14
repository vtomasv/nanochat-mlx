"""Training orchestration. Every timing arm is a separate CLI process."""
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from .config import dump
from .measure import setup_device,memory,RSSSampler
from .replay import seed_all,prepare,save_checkpoint,load_checkpoint,verify_files
from .reporting import Run

def build(c,seed,arm='native'):
    import mlx.core as mx
    from nanochat_mlx.train import build_model
    from nanochat_mlx.gpt import GPT,GPTConfig
    from nanochat_mlx.optim import MultiOptimizer,OptimizerConfig
    os.environ['NANOCHAT_BASE_DIR']=c['data_dir']
    from nanochat_mlx.tokenizer import get_tokenizer
    seed_all(seed)
    model=build_model(c['depth'],get_tokenizer().get_vocab_size(),c['aspect_ratio'],c['head_dim'],c['sequence_len'],c['window_pattern'])
    count=model.num_scaling_params()
    # Match train.py's depth-12 reference and exact order of scalar arithmetic.
    ref=GPT(GPTConfig(sequence_len=c['sequence_len'],vocab_size=model.config.vocab_size,n_layer=12,n_head=6,n_kv_head=6,n_embd=768))
    rc=ref.num_scaling_params(); del ref
    target=int(12*(count['transformer_matrices']+count['lm_head']))
    dref=12*(rc['transformer_matrices']+rc['lm_head'])
    scale=(c['total_batch_size']/2**19)**.5
    wd=.2*math.sqrt(c['total_batch_size']/2**19)*(dref/max(target,1))
    optc=OptimizerConfig(model.config.n_embd,unembedding_lr=.004*scale,embedding_lr=.3*scale,matrix_lr=.02*scale,weight_decay=wd,scalar_lr=.5*scale)
    cls=MultiOptimizer
    if arm.startswith('staged_'):
        from .staged_optimizer import StagedOptimizer
        opt=StagedOptimizer(model,optc,compress=arm=='staged_grammar',rows_per_block=c['rows_per_block'])
    else: opt=cls(model,optc)
    return model,opt

def update(model,opt,batches,step,c,arm):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map
    from nanochat_mlx.gpt import loss_fn
    from nanochat_mlx.optim import get_lr_multiplier,get_muon_momentum,get_weight_decay
    fn=nn.value_and_grad(model,loss_fn)
    if arm.startswith('tape_'):
        from .activation_tape import loss_and_grad
        fn=lambda m,x,y:loss_and_grad(m,x,y,arm,c['rank_sample_bits'])
    accum=None; loss_sum=0.
    for x,y in batches:
        loss,grads=fn(model,x,y); lv=loss.item()
        if not math.isfinite(lv): raise FloatingPointError(f'Nonfinite loss at {step}')
        accum=grads if accum is None else tree_map(lambda a,b:a+b,accum,grads)
        mx.eval(loss,accum); loss_sum+=lv
    if c['accumulation']>1: accum=tree_map(lambda g:g*(1./c['accumulation']),accum)
    opt.set_lr_multiplier(get_lr_multiplier(step,1000,0,.5,0))
    opt.set_muon_momentum(get_muon_momentum(step))
    opt.set_muon_weight_decay(get_weight_decay(step,1000,opt.config.weight_decay))
    opt.update(model,accum); mx.eval(model.parameters(),*opt.state); mx.synchronize()
    return loss_sum/c['accumulation']

def evaluate(model,c):
    import mlx.core as mx
    x=np.load(Path(c['artifact_dir'])/'val_x.npy',mmap_mode='r'); y=np.load(Path(c['artifact_dir'])/'val_y.npy',mmap_mode='r')
    losses=[]
    for i in range(32,128):
        loss=model(mx.array(x[i:i+1]),targets=mx.array(y[i:i+1])); mx.eval(loss); losses.append(loss.item())
    return {'nll':float(np.mean(losses)),'valid_tokens':96*c['sequence_len'],'per_sequence_nll':losses,'bpb':None}

def train_run(c,arm,stage,seed,steps,checkpoint=None,experiment='baseline',run_id=None):
    import mlx.core as mx
    setup_device()
    tape=json.loads((Path(c['artifact_dir'])/'tape.json').read_text()); verify_files(tape['files'])
    if tape['config_sha256']!=c['config_sha256']: raise ValueError('Tape profile mismatch')
    run=Run(experiment,c,arm,stage,seed,run_id)
    run.event('start',arm=arm,seed=seed,steps=steps)
    try:
        build_tick=time.perf_counter();model,opt=build(c,seed,arm)
        run.event('model_optimizer_construction',seconds=time.perf_counter()-build_tick)
        initial=Path(c['artifact_dir'])/f'initial-s{seed}'
        if not initial.exists():
            if arm!='native': raise ValueError('Native initial checkpoint must exist first')
            save_checkpoint(initial,model,opt,0,seed,c)
        load_tick=time.perf_counter();state=load_checkpoint(checkpoint or initial,model,opt,c); start=state['step']
        run.event('checkpoint_load_and_state_conversion',seconds=time.perf_counter()-load_tick,path=str(checkpoint or initial))
        run.event('loaded_state',start_step=start,target_step=start+steps,cursor=state['cursor'])
        if start+steps>1000: raise ValueError('Cannot extend original schedule')
        x=np.load(Path(c['artifact_dir'])/'train_x.npy',mmap_mode='r'); y=np.load(Path(c['artifact_dir'])/'train_y.npy',mmap_mode='r')
        pipeline=c.get('measurement_mode','replay')=='pipeline'
        def make_loader():
            from nanochat_mlx.tokenizer import get_tokenizer
            from nanochat_mlx.dataloader import dataloader_bos_bestfit
            return dataloader_bos_bestfit(get_tokenizer(),c['device_batch_size'],c['sequence_len'],'train')
        loader=make_loader() if pipeline else None
        if pipeline and start: raise ValueError('Pipeline resume is not exact; use replay checkpoint windows')
        def batches(step):
            if pipeline:
                for _ in range(c['accumulation']):
                    a,b,_=next(loader);yield a,b
                return
            for i in range(step*c['accumulation'],(step+1)*c['accumulation']):
                if c.get('replay_layout','contiguous')=='original':
                    row=mx.array(np.concatenate((x[i],y[i,:,-1:]),axis=-1))
                    yield row[:,:-1],row[:,1:]
                else:yield mx.array(x[i]),mx.array(y[i])
        # Warm disposable values, then reload weights, optimizer and RNG before measuring.
        cold=time.perf_counter(); update(model,opt,batches(start),start,c,arm)
        run.event('warmup_discarded',seconds=time.perf_counter()-cold)
        del model,opt
        model,opt=build(c,seed,arm); load_checkpoint(checkpoint or initial,model,opt,c)
        if pipeline: loader=make_loader()
        mx.reset_peak_memory()
        times=[]; wall=time.perf_counter()
        with RSSSampler() as rss:
            for step in range(start,start+steps):
                tick=time.perf_counter(); loss=update(model,opt,batches(step),step,c,arm); elapsed=time.perf_counter()-tick
                times.append(elapsed)
                if rss.swap_violation: raise MemoryError('Sustained swap increase exceeds 1 GiB')
                run.row('steps.csv',dict(step=step,loss=loss,seconds=elapsed,tokens=c['total_batch_size'],**memory()))
                if hasattr(opt,'last_accounting'):
                    for row in opt.last_accounting: run.row('tensors.csv',dict(step=step,**row))
                if hasattr(model,'_navarro_accounting'):
                    for row in model._navarro_accounting: run.row('tensors.csv',dict(step=step,**row))
                if (step+1)%10==0: print(f'{run.root.name} step={step+1} loss={loss:.6f} seconds={elapsed:.3f}',flush=True)
                if step+1 in (10,100,400,500,900,1000) or step+1==start+steps:
                    dest=Path(c['artifact_dir'])/'checkpoints'/experiment/run.root.name/f'step-{step+1:04d}'
                    tick=time.perf_counter(); save_checkpoint(dest,model,opt,step+1,seed,c)
                    run.event('checkpoint',step=step+1,path=str(dest),seconds=time.perf_counter()-tick)
        total=time.perf_counter()-wall
        run.event('swap_samples',samples=rss.swap_samples)
        tick=time.perf_counter(); quality=evaluate(model,c); run.event('evaluation',seconds=time.perf_counter()-tick,**quality)
        stats=dict(median_step_seconds=float(np.median(times)),mean_step_seconds=float(np.mean(times)),p95_step_seconds=float(np.quantile(times,.95)),tokens_per_second=c['total_batch_size']/float(np.mean(times)),parameters=model.num_scaling_params()['total'],steps=steps,tokens=steps*c['total_batch_size'],wall_seconds=total,process_tree_rss_peak_bytes=rss.peak,process_tree_rss_error=rss.error,**quality,**memory())
        run.finish(status='measured',intervention=stats,decision='INCONCLUSIVE',limitations=['Single engineering process; no confirmatory confidence interval.','RSS and Metal are reported separately, never summed.','Per-phase profiling and pipeline timing are separate requirements.'])
        print(str(run.root),flush=True)
        return run.root
    except BaseException as e:
        run.event('error',error=repr(e)); run.finish(status='INCONCLUSIVE',decision='INCONCLUSIVE',failure_reason=repr(e)); raise
