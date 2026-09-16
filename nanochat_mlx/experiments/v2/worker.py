"""Fixed evaluator/training worker. Search evaluates calibration only.

Candidate source is archived and loaded explicitly; trials run in new processes.
Numerical gates compare every trainable tensor before collecting timing data.
"""
import functools
import hashlib
import shutil
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from .records import digest,now,write_json,verify_sources


def protect_reserved(data_root):
    """Catch accidental Python-level reads of held-out files during discovery.

    This is an experimental guard, not a security sandbox for hostile native code.
    """
    data_root=Path(data_root).resolve()
    def audit(event,args):
        if event not in ('open','sqlite3.connect') or not args or not isinstance(args[0],(str,bytes)):
            return
        path=Path(os.fsdecode(args[0])).resolve()
        if path.parent==data_root and (path.name.startswith('reserved_') or path.name=='documents.sqlite'):
            raise PermissionError('Discovery cannot read reserved evaluation or document database')
    import os
    sys.addaudithook(audit)


def candidate_packer(path,scenario):
    spec=importlib.util.spec_from_file_location('navarro_trial_candidate',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return functools.partial(module.pack_positive,strategy=scenario.get('strategy','rank'))


def gradient_fn(model,scenario,packer=None):
    import mlx.core as mx
    import mlx.nn as nn
    from nanochat_mlx.gpt import loss_fn,norm
    from nanochat_mlx.experiments.activation_tape import loss_and_grad
    arm=scenario['arm']
    if arm=='bitmap_gpu':
        return lambda m,x,y:loss_and_grad(m,x,y,'tape_bitmap',scenario.get('sample_bits',256),packer=packer)
    if arm.startswith('tape_'):
        return lambda m,x,y:loss_and_grad(m,x,y,arm,scenario.get('sample_bits',256))
    if arm=='checkpoint_native':
        from mlx.nn.utils import checkpoint
        def loss(m,ids,targets):
            masks=m._get_masks(ids.shape[1]);x=norm(m.wte(ids));x0=x
            for i,block in enumerate(m.blocks):
                x=m.resid_lambdas[i]*x+m.x0_lambdas[i]*x0
                ve=m.value_embeds[str(i)](ids) if str(i) in m.value_embeds else None
                x=checkpoint(block)(x,ve=ve,mask=masks[i])
            logits=m.lm_head(norm(x))[...,:m.config.vocab_size].astype(mx.float32)
            logits=15*mx.tanh(logits/15);mask=targets!=-1
            safe=mx.where(mask,targets,mx.zeros_like(targets))
            return mx.sum(nn.losses.cross_entropy(logits,safe,reduction='none')*mask)/mx.maximum(mx.sum(mask),1)
        return nn.value_and_grad(model,loss)
    if arm!='native':raise ValueError('Unknown immutable-harness arm: '+arm)
    return nn.value_and_grad(model,loss_fn)


def step(model,opt,batches,step_id,config,fn):
    import mlx.core as mx
    from mlx.utils import tree_map
    from nanochat_mlx.optim import get_lr_multiplier,get_muon_momentum,get_weight_decay
    accum=None;loss_total=0.
    for x,y in batches:
        loss,grads=fn(model,x,y);mx.eval(loss,grads)
        value=loss.item()
        if not math.isfinite(value):raise FloatingPointError('Nonfinite training loss')
        accum=grads if accum is None else tree_map(lambda a,b:a+b,accum,grads)
        mx.eval(accum);loss_total+=value
    if config['accumulation']>1:accum=tree_map(lambda g:g/config['accumulation'],accum)
    opt.set_lr_multiplier(get_lr_multiplier(step_id,1000,0,.5,0))
    opt.set_muon_momentum(get_muon_momentum(step_id))
    opt.set_muon_weight_decay(get_weight_decay(step_id,1000,opt.config.weight_decay))
    opt.update(model,accum);mx.eval(model.parameters(),*opt.state);mx.synchronize()
    return loss_total/config['accumulation']


def evaluate(model,data_root,split='calibration'):
    import mlx.core as mx
    if split not in ('calibration','reserved'):raise ValueError('Unknown evaluation partition')
    x=np.load(data_root/f'{split}_x.npy',mmap_mode='r');y=np.load(data_root/f'{split}_y.npy',mmap_mode='r')
    provenance=json.loads((data_root/f'{split}_provenance.json').read_text())
    rows=[];total=0.;count=0
    for i in range(len(x)):
        n=int(np.count_nonzero(y[i]!=-1))
        loss=model(mx.array(x[i:i+1]),targets=mx.array(y[i:i+1]));mx.eval(loss)
        value=loss.item()
        if not math.isfinite(value):raise FloatingPointError('Nonfinite evaluation')
        total+=value*n;count+=n
        rows.append({'document_id':provenance[i]['document_id'],'nll_sum':value*n,'valid_tokens':n})
    return {'split':split,'nll':total/count,'valid_tokens':count,'documents':len(rows),'rows':rows}


def numerical_gate(model,x,y,scenario,packer):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from nanochat_mlx.gpt import loss_fn
    from nanochat_mlx.experiments.metrics import errors
    codec_gate=[]
    if packer is not None:
        for n in (0,33,257,4096):
            rng=np.random.default_rng(821+n);p=np.maximum(rng.normal(size=n).astype(np.float32),0)
            if n:p[0]=np.nextafter(np.float32(0),np.float32(1))
            handle=packer(mx.array(p),scenario.get('sample_bits',256))
            actual=np.asarray(handle.unpack('mlx'))
            ok=actual.dtype==p.dtype and actual.shape==p.shape and actual.tobytes()==p.tobytes()
            codec_gate.append({'n':n,'pass':ok})
        if not all(r['pass'] for r in codec_gate):return {'pass':False,'codec':codec_gate,'reason':'Candidate failed exact roundtrip'}
    ref,gr=nn.value_and_grad(model,loss_fn)(model,x,y);mx.eval(ref,gr)
    value,ga=gradient_fn(model,scenario,packer)(model,x,y);mx.eval(value,ga)
    a=dict(tree_flatten(ga));b=dict(tree_flatten(gr))
    if a.keys()!=b.keys():raise ValueError('Missing/extra gradient parameter')
    records={k:errors(np.asarray(a[k]),np.asarray(b[k]),atol=1e-5,rtol=1e-3) for k in a}
    good=abs(value.item()-ref.item())<=1e-4 and all(r['pass'] and r['relative_l2']<=1e-3 for r in records.values())
    return {'pass':good,'codec':codec_gate,'loss_delta':value.item()-ref.item(),'tensors':records,
        'reference_nll':ref.item(),'candidate_nll':value.item()}


def run(campaign,trial_id):
    import csv
    import mlx.core as mx
    from nanochat_mlx.experiments.runner import build
    from nanochat_mlx.experiments.replay import save_checkpoint,load_checkpoint
    from nanochat_mlx.experiments.measure import setup_device,memory,RSSSampler,environment
    from .data import verify
    trial=campaign/'trials'/trial_id;request=json.loads((trial/'request.json').read_text())
    verify_sources(request['sources']['files'])
    if digest(trial/'candidate.py')!=request['candidate_sha256']:raise ValueError('Candidate source changed')
    c=request['config'];scenario=request['scenario'];data_root=Path(request['data_root'])
    quality_split=request.get('quality_split','calibration')
    if quality_split=='reserved':
        plan=campaign/'sealed-plan.json'
        if not plan.exists() or request.get('sealed_plan_sha256')!=digest(plan):raise ValueError('Reserved evaluation requires sealed confirmatory plan')
    else:protect_reserved(data_root)
    if request['data_manifest_sha256']!=digest(data_root/'manifest.json'):raise ValueError('Data manifest changed')
    verify(data_root,include_reserved=quality_split=='reserved')
    started=time.monotonic();write_json(trial/'environment.json',environment())
    setup_device();mx.set_cache_limit(1024**3)
    checkpoints=Path(request['artifact_root'])/trial_id
    def save_checked(destination,model,opt,step_id):
        from mlx.utils import tree_flatten
        live=sum(a.nbytes for _,a in tree_flatten(model.parameters()))+sum(a.nbytes for _,a in tree_flatten(opt.state) if hasattr(a,'nbytes'))
        if shutil.disk_usage(Path(request['artifact_root'])).free<live*1.2+c.get('disk_reserve_gib',3)*1024**3:
            raise OSError('Checkpoint would violate registered disk reserve')
        save_checkpoint(destination,model,opt,step_id,c['seed'],c)
    initial=Path(request['initial_checkpoint'])
    packer=candidate_packer(trial/'candidate.py',scenario) if scenario['arm']=='bitmap_gpu' else None
    model,opt=build(c,c['seed'])
    if not initial.exists():
        if request['kind']!='reference':raise ValueError('Missing native initial checkpoint')
        save_checked(initial,model,opt,0)
    input_checkpoint=Path(request.get('checkpoint') or initial)
    state=load_checkpoint(input_checkpoint,model,opt,c)
    x=np.load(data_root/'train_x.npy',mmap_mode='r');y=np.load(data_root/'train_y.npy',mmap_mode='r')
    if x.shape!=y.shape or x.shape[1:]!=(c['device_batch_size'],c['sequence_len']):raise ValueError('Replay/config shape mismatch')
    if len(x)<(state['step']+request['steps'])*c['accumulation']:raise ValueError('Replay too short')
    def batches(k):
        for i in range(k*c['accumulation'],(k+1)*c['accumulation']):
            yield mx.array(x[i]),mx.array(y[i])
    # Same single-batch gate on trained weights; rows unchanged across arms.
    gx,gy=next(batches(state['step']))
    gate=numerical_gate(model,gx,gy,scenario,packer)
    write_json(trial/'correctness.json',gate)
    del gx,gy
    if not gate['pass']:
        write_json(trial/'summary.json',{'status':'FAILURE_CORRECTNESS','correctness_pass':False,'scenario':scenario,'quality':None,'metrics':None});return
    if request['kind']=='gate':
        write_json(trial/'summary.json',{'status':'GATE_PASS','correctness_pass':True,'scenario':scenario});return
    fn=gradient_fn(model,scenario,packer)
    warm=time.monotonic();step(model,opt,batches(state['step']),state['step'],c,fn)
    warm_seconds=time.monotonic()-warm
    del fn,model,opt;mx.synchronize()
    model,opt=build(c,c['seed']);state=load_checkpoint(input_checkpoint,model,opt,c);fn=gradient_fn(model,scenario,packer)
    # All live baseline weights, states and graph temporaries were restored.
    mx.reset_peak_memory();times=[];losses=[];payload=[]
    start_step=state['step'];stop_step=start_step+request['steps']
    if stop_step>1000:raise ValueError('Schedule would extend beyond 1000')
    checkpoint_events=[];measured_started=time.monotonic()
    with (trial/'steps.csv').open('x',newline='') as output,(trial/'tensors.jsonl').open('x') as tensors,RSSSampler() as rss:
        writer=csv.DictWriter(output,fieldnames=['step','loss','seconds','tokens','mlx_active_peak_bytes','rss_peak_bytes']);writer.writeheader()
        for k in range(start_step,stop_step):
            tick=time.monotonic();loss=step(model,opt,batches(k),k,c,fn);elapsed=time.monotonic()-tick
            times.append(elapsed);losses.append(loss);mem=memory()
            writer.writerow({'step':k,'loss':loss,'seconds':elapsed,'tokens':c['total_batch_size'],
                'mlx_active_peak_bytes':mem['mlx_active_peak_bytes'],'rss_peak_bytes':mem['rss_peak_bytes']});output.flush()
            account=getattr(model,'_navarro_accounting',[])
            if account:
                payload.append(sum(r['resident_bytes'] for r in account)/sum(r['raw_bytes'] for r in account))
                tensors.write(json.dumps({'step':k,'microbatch':'last_of_accumulation','tensors':account})+'\n');tensors.flush()
            if rss.swap_violation:raise MemoryError('Sustained swap increase exceeds 1 GiB')
            if (k+1)%10==0:print(trial_id,'step',k+1,'loss',round(loss,6),'seconds',round(elapsed,4),flush=True)
            save_steps=c.get('reference_checkpoint_steps',[10,100,400,500,900,1000]) if quality_split=='calibration' else ([400,900] if c['seed']==211 else [])
            if request['kind']=='reference' and k+1 in save_steps:
                dest=checkpoints/f'step-{k+1:04d}';tick=time.monotonic();save_checked(dest,model,opt,k+1)
                checkpoint_events.append({'step':k+1,'path':str(dest),'seconds':time.monotonic()-tick})
        # Freeze training peaks BEFORE checkpoint/evaluation allocations.
        training_memory=memory();rss_peak=rss.peak;rss_error=rss.error;swap_samples=list(rss.swap_samples)
    measured_wall=time.monotonic()-measured_started
    # Final fingerprints are outside measurement. Pilots are replayable from the
    # archived source, input checkpoint and token tape, not resumable checkpoints.
    from mlx.utils import tree_flatten
    fingerprints={}
    for group,tree in [('model',model.parameters()),('optimizer',opt.state)]:
        fingerprints[group]={}
        for name,array in tree_flatten(tree):
            if hasattr(array,'shape'):
                data=np.asarray(array)
                fingerprints[group][name]={'shape':list(data.shape),'dtype':str(data.dtype),
                    'sha256':hashlib.sha256(data.tobytes()).hexdigest()}
    write_json(trial/'final-state-fingerprints.json',{'step':stop_step,'tensors':fingerprints,
        'checkpoint_policy':'Fingerprints only at final state; full native checkpoints listed in summary.'})
    tick=time.monotonic();quality=evaluate(model,data_root,quality_split);evaluation_seconds=time.monotonic()-tick
    write_json(trial/'quality.json',quality)
    write_json(trial/'summary.json',{'status':'MEASURED_EXPLORATORY','correctness_pass':True,'scenario':scenario,
        'started_at':request['created_at'],'completed_at':now(),'steps':len(times),'start_step':start_step,
        'tokens':len(times)*c['total_batch_size'],'parameters':model.num_scaling_params()['total'],
        'quality':{k:v for k,v in quality.items() if k!='rows'},
        'metrics':{'mean_step_seconds':float(np.mean(times)),'median_step_seconds':float(np.median(times)),
            'p95_step_seconds':float(np.quantile(times,.95)),'payload_ratio':float(np.mean(payload)) if payload else None,
            **training_memory,'process_tree_rss_peak_bytes':rss_peak,'rss_error':rss_error},
        'warmup_discarded_seconds':warm_seconds,'training_wall_including_checkpoints_seconds':measured_wall,
        'evaluation_seconds':evaluation_seconds,'total_seconds':time.monotonic()-started,
        'checkpoints':checkpoint_events,'swap_samples':swap_samples,
        'limitations':['One engineering process; no confirmatory inference.',
            'Calibration only; reserved evaluation was not opened.' if quality_split=='calibration' else 'Reserved evaluation opened under sealed confirmatory plan.',
            'RSS and MLX overlap and must not be summed.',
            'Activation accounting records last microbatch, not a whole-step live-buffer census.']})
    verify_sources(request['sources']['files'])
