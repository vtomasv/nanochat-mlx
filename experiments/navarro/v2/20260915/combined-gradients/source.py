"""A separately registered pilot of exact embedding-gradient row storage.

Only accumulator residence between microbatches changes. Every gradient is
materialized before optimizer.update, with the original elementwise sum/order.
All unvisited rows must contain positive-zero bits; otherwise abort explicitly.
No test-reserved data, no optimizer/architecture changes, no promotion by p-value.
"""
import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten,tree_unflatten,tree_map

from nanochat_mlx.experiments.v2.records import digest,write_json,now,verify_sources
from nanochat_mlx.experiments.v2.worker import gradient_fn,step,evaluate,numerical_gate,protect_reserved,candidate_packer
from nanochat_mlx.experiments.runner import build
from nanochat_mlx.experiments.replay import load_checkpoint
from nanochat_mlx.experiments.measure import setup_device,environment,memory,RSSSampler


@dataclass
class Rows:
    shape: tuple
    indices: object
    bits: object

    @property
    def bytes(self):return self.indices.nbytes+self.bits.nbytes+64

    def restore(self,union=None):
        rows=self.shape[0] if union is None else len(union)
        indices=self.indices if union is None else np.searchsorted(union,self.indices).astype(np.uint32)
        out=mx.zeros((rows,self.shape[1]),mx.uint32)
        out[mx.array(indices)]=self.bits
        return out.view(mx.float32)


def pack_gradients(grads,ids):
    unique=np.unique(np.asarray(ids)).astype(np.uint32);handles={};checks=[]
    for name,g in tree_flatten(grads):
        if name=='wte.weight' or name.startswith('value_embeds.') and name.endswith('.weight'):
            if g.ndim!=2 or g.dtype!=mx.float32 or len(unique) and int(unique[-1])>=g.shape[0]:raise ValueError('Embedding-gradient schema mismatch')
            row_mask=mx.zeros((g.shape[0],),mx.bool_);row_mask[mx.array(unique)]=True
            checks.append(mx.all(mx.where(row_mask[:,None],mx.array(True),g.view(mx.uint32)==0)))
            h=Rows(g.shape,unique,g[mx.array(unique)].view(mx.uint32));handles[name]=h
        else:handles[name]=g
    if not checks:raise ValueError('No embedding gradients found')
    good=mx.all(mx.stack(checks));mx.eval(good,[h.bits if isinstance(h,Rows) else h for h in handles.values()])
    if not good.item():raise ValueError('Unvisited embedding row has nonzero bits; cannot discard it')
    return handles


def accumulate(a,b):
    if a.keys()!=b.keys():raise ValueError('Gradient tree changed')
    result={}
    for name,left in a.items():
        right=b[name]
        if isinstance(left,Rows):
            if not isinstance(right,Rows) or left.shape!=right.shape:raise ValueError('Row storage mismatch')
            union=np.union1d(left.indices,right.indices).astype(np.uint32)
            value=left.restore(union)+right.restore(union)
            result[name]=Rows(left.shape,union,value.view(mx.uint32))
        else:result[name]=left+right
    mx.eval([h.bits if isinstance(h,Rows) else h for h in result.values()]);return result


def unpack_gradients(handles,divisor):
    return tree_unflatten([(name,(h.restore() if isinstance(h,Rows) else h)/divisor) for name,h in handles.items()])


def row_step(model,opt,batches,step_id,c,fn):
    from nanochat_mlx.optim import get_lr_multiplier,get_muon_momentum,get_weight_decay
    accumulator=None;loss_total=0.;account=[]
    for x,y in batches:
        loss,grads=fn(model,x,y);mx.eval(loss,grads);value=loss.item()
        if not np.isfinite(value):raise FloatingPointError('Nonfinite training loss')
        current=pack_gradients(grads,x);del grads
        accumulator=current if accumulator is None else accumulate(accumulator,current)
        del current
        raw=sum(np.prod(h.shape)*4 if isinstance(h,Rows) else h.nbytes for h in accumulator.values())
        stored=sum(h.bytes if isinstance(h,Rows) else h.nbytes for h in accumulator.values())
        account.append({'raw_bytes':int(raw),'stored_bytes':int(stored),
            'embedding_rows':{name:len(h.indices) for name,h in accumulator.items() if isinstance(h,Rows)}})
        loss_total+=value
    grads=unpack_gradients(accumulator,c['accumulation']);mx.eval(grads);del accumulator
    opt.set_lr_multiplier(get_lr_multiplier(step_id,1000,0,.5,0))
    opt.set_muon_momentum(get_muon_momentum(step_id))
    opt.set_muon_weight_decay(get_weight_decay(step_id,1000,opt.config.weight_decay))
    opt.update(model,grads);mx.eval(model.parameters(),*opt.state);mx.synchronize()
    model._row_gradient_accounting=account
    return loss_total/c['accumulation']


def selftest(candidate):
    import mlx.nn as nn
    from nanochat_mlx.gpt import GPT,GPTConfig,loss_fn
    from nanochat_mlx.optim import MultiOptimizer,OptimizerConfig
    # Bit-level roundtrip and accumulation include negative/subnormal values.
    first=np.zeros((7,4),np.float32);second=first.copy()
    first[1]=[1.,-1.,np.nextafter(np.float32(0),np.float32(1)),-0.]
    second[3]=[-2.,4.,0.,-3.]
    a=pack_gradients({'wte':{'weight':mx.array(first)}},mx.array([[1,1]]))
    b=pack_gradients({'wte':{'weight':mx.array(second)}},mx.array([[3]]))
    np.testing.assert_array_equal(np.asarray(a['wte.weight'].restore()).view(np.uint32),first.view(np.uint32))
    actual=unpack_gradients(accumulate(a,b),2)['wte']['weight']
    expected=(mx.array(first)+mx.array(second))/2;mx.eval(actual,expected)
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint32),np.asarray(expected).view(np.uint32))
    invalid=first.copy();invalid[6,0]=1
    try:pack_gradients({'wte':{'weight':mx.array(invalid)}},mx.array([[1]]))
    except ValueError:pass
    else:raise AssertionError('Discarded a nonzero unvisited gradient')
    mx.random.seed(907);cfg=GPTConfig(sequence_len=16,vocab_size=67,n_layer=4,n_head=2,n_kv_head=2,n_embd=32,window_pattern='SSSL')
    native=GPT(cfg);native.init_weights()
    for block in native.blocks:
        block.attn.c_proj.weight=mx.random.normal(block.attn.c_proj.weight.shape)*.03
        block.mlp.c_proj.weight=mx.random.normal(block.mlp.c_proj.weight.shape)*.03
    model=GPT(cfg);model.update(native.parameters());mx.eval(native.parameters(),model.parameters())
    oc=OptimizerConfig(32,matrix_lr=.0001,embedding_lr=.0001,unembedding_lr=.0001,scalar_lr=.0001)
    oa=MultiOptimizer(native,oc);ob=MultiOptimizer(model,oc)
    batches=[(mx.array([[1,1,2,3,4,5,6,7]]),mx.array([[1,2,3,4,5,6,7,-1]])),
             (mx.array([[2,2,3,4,5,6,7,8]]),mx.array([[2,3,4,5,6,7,8,9]]))]
    packer=candidate_packer(candidate,{'strategy':'rank'})
    records=[]
    for k in range(10):
        la=step(native,oa,batches,k,{'accumulation':2},nn.value_and_grad(native,loss_fn))
        lb=row_step(model,ob,batches,k,{'accumulation':2},gradient_fn(model,{'arm':'bitmap_gpu','sample_bits':256},packer))
        if abs(la-lb)>1e-4:raise AssertionError('Ten-step loss mismatch')
        for label,a,b in [('weights',native.parameters(),model.parameters()),('optimizer',oa.state,ob.state)]:
            av=dict(tree_flatten(a));bv=dict(tree_flatten(b))
            if av.keys()!=bv.keys():raise AssertionError('State tree mismatch')
            for name,v in av.items():np.testing.assert_allclose(np.asarray(v),np.asarray(bv[name]),atol=1e-5,rtol=1e-3,err_msg=label+':'+name)
        records.append({'step':k,'loss_delta':lb-la,'weights_and_optimizer_pass':True})
    return {'row_bit_roundtrip':True,'accumulation_bits_match':True,'nonzero_unvisited_rejected':True,'ten_updates':records}


def worker(root,trial_id):
    plan=json.loads((root/'plan.json').read_text());parent=Path(plan['parent_campaign']);m=json.loads((parent/'manifest.json').read_text())
    if digest(Path(__file__))!=plan['source_sha256'] or digest(root/'candidate.py')!=plan['candidate_sha256']:raise ValueError('Study source changed')
    verify_sources(m['protected_sources']);protect_reserved(Path(m['data_root']))
    from nanochat_mlx.experiments.v2.data import verify
    verify(Path(m['data_root']))
    trial=root/trial_id;request=json.loads((trial/'request.json').read_text());arm=request['scenario'];c=m['config']
    setup_device();write_json(trial/'environment.json',environment())
    packer=candidate_packer(root/'candidate.py',arm) if arm['arm']=='bitmap_gpu' else None
    checkpoint=Path(m['artifact_root'])/'reference/step-0100'
    x=np.load(Path(m['data_root'])/'train_x.npy',mmap_mode='r');y=np.load(Path(m['data_root'])/'train_y.npy',mmap_mode='r')
    def batches(k):return [(mx.array(x[k*2+i]),mx.array(y[k*2+i])) for i in range(2)]
    update=row_step if arm['compact_gradients'] else step
    model,opt=build(c,c['seed']);load_checkpoint(checkpoint,model,opt,c)
    gx,gy=batches(100)[0];gate=numerical_gate(model,gx,gy,arm,packer);write_json(trial/'correctness.json',gate);del gx,gy
    if not gate['pass']:raise ValueError('Full model gradient gate failed')
    fn=gradient_fn(model,arm,packer);update(model,opt,batches(100),100,c,fn)
    del fn,model,opt;mx.synchronize()
    model,opt=build(c,c['seed']);load_checkpoint(checkpoint,model,opt,c);fn=gradient_fn(model,arm,packer)
    times=[];mx.reset_peak_memory();tick=time.monotonic()
    with (trial/'steps.csv').open('x',newline='') as f,(trial/'gradient-storage.jsonl').open('x') as storage,RSSSampler() as rss:
        writer=csv.DictWriter(f,fieldnames=['step','loss','seconds']);writer.writeheader()
        for k in range(100,200):
            t=time.monotonic();loss=update(model,opt,batches(k),k,c,fn);elapsed=time.monotonic()-t;times.append(elapsed)
            writer.writerow({'step':k,'loss':loss,'seconds':elapsed});f.flush()
            if arm['compact_gradients']:storage.write(json.dumps({'step':k,'microbatches':model._row_gradient_accounting})+'\n');storage.flush()
            if rss.swap_violation:raise MemoryError('Swap safety limit exceeded')
            if (k+1)%20==0:print(trial_id,k+1,round(elapsed,4),flush=True)
        metrics=memory();metrics.update(mean_step_seconds=float(np.mean(times)),p95_step_seconds=float(np.quantile(times,.95)),process_tree_rss_peak_bytes=rss.peak,rss_error=rss.error)
    quality=evaluate(model,Path(m['data_root']));write_json(trial/'quality.json',quality)
    result={'status':'MEASURED_EXPLORATORY','correctness_pass':True,'scenario':arm,'steps':100,'start_step':100,
        'metrics':metrics,'quality':{k:v for k,v in quality.items() if k!='rows'},'training_seconds':time.monotonic()-tick,
        'source_sha256':plan['source_sha256'],'limitations':['One exploratory process, no confirmatory CI.','No final checkpoint, reproducible from parent checkpoint and sources.','RSS and MLX overlap.']}
    write_json(trial/'summary.json',result)
    if digest(Path(__file__))!=plan['source_sha256']:raise ValueError('Source changed during worker')


def report(root):
    records=[]
    for folder in root.iterdir():
        if folder.is_dir() and (folder/'artifacts.json').exists():
            for name,h in json.loads((folder/'artifacts.json').read_text()).items():
                if digest(folder/name)!=h:raise ValueError('Trial evidence modified')
            r=json.loads((folder/'summary.json').read_text());r['trial']=folder.name;records.append(r)
    write_json(root/'results.json',records,exclusive=False)
    by_id={r['trial']:r for r in records};candidates=[]
    for name in ('rows-dense','rows-bitmap'):
        r=by_id.get(name);native=by_id.get('native');dense=by_id.get('tape-dense');recompute=by_id.get('tape-recompute')
        if not all(x and x.get('status')=='MEASURED_EXPLORATORY' for x in (r,native,dense,recompute)):continue
        a=r['metrics'];b=native['metrics'];d=dense['metrics'];rc=recompute['metrics']
        conditions={'memory_vs_native':a['mlx_active_peak_bytes']<=.9*b['mlx_active_peak_bytes'],
            'memory_vs_tape_dense':a['mlx_active_peak_bytes']<=.95*d['mlx_active_peak_bytes'],
            'time_vs_native':a['mean_step_seconds']<=1.1*b['mean_step_seconds'],
            'rss_vs_native':a['rss_peak_bytes']<=1.05*b['rss_peak_bytes'],
            'calibration_delta':abs(r['quality']['nll']-native['quality']['nll'])<.01,
            'not_dominated_by_recompute':not(rc['mlx_active_peak_bytes']<=a['mlx_active_peak_bytes'] and rc['mean_step_seconds']<=a['mean_step_seconds'])}
        candidates.append({'trial':name,'conditions':conditions,'decision':'CANDIDATE_FOR_NEW_CONFIRMATORY_PLAN' if all(conditions.values()) else 'NOT_SHORTLISTED'})
    write_json(root/'decisions.json',{'evidence':'engineering_only','candidates':candidates},exclusive=False)
    return candidates


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--worker');parser.add_argument('--root',default='experiments/navarro/v2/20260915/combined-gradients');args=parser.parse_args();root=Path(args.root).resolve()
    if args.worker:worker(root,args.worker);return
    root.mkdir(parents=True,exist_ok=False)
    parent=Path('experiments/navarro/v2/20260915').resolve();m=json.loads((parent/'manifest.json').read_text());verify_sources(m['protected_sources'])
    shutil.copyfile(Path(__file__),root/'source.py');shutil.copyfile('research/navarro/candidate.py',root/'candidate.py')
    scenarios=[{'id':'native','arm':'native','compact_gradients':False},
        {'id':'tape-dense','arm':'tape_dense','compact_gradients':False},
        {'id':'tape-recompute','arm':'tape_recompute','compact_gradients':False},
        {'id':'bitmap','arm':'bitmap_gpu','sample_bits':256,'strategy':'rank','compact_gradients':False},
        {'id':'rows-dense','arm':'tape_dense','compact_gradients':True},
        {'id':'rows-bitmap','arm':'bitmap_gpu','sample_bits':256,'strategy':'rank','compact_gradients':True}]
    # Randomize exploratory order before any measurements of this extension.
    np.random.default_rng(20260915).shuffle(scenarios)
    plan={'created_at':now(),'parent_campaign':str(parent),'parent_manifest_sha256':digest(parent/'manifest.json'),
        'source_sha256':digest(Path(__file__)),'candidate_sha256':digest(root/'candidate.py'),'scenarios':scenarios,
        'hypothesis':'Exact storage of visited embedding-gradient rows reduces accumulator residence; combining it with P bitmaps may cross the unchanged marginal-memory threshold.',
        'rationale_source_sha256':digest(parent/'gradient-census/summary.json'),'steps':100,'checkpoint_step':100,
        'maximum_experimental_variants':2,'fixed_controls':4,'worker_timeout_seconds':900,'study_budget_seconds':3600,
        'evaluation':'calibration only','status':'new exploratory amendment, not part of the original 12-proposal P-only surface',
        'success_rule':'Same point-estimate screening criteria as parent; no statistical success from this pilot.',
        'stop_rule':'Run all registered controls and two candidates. Failed gates are failures. No threshold changes, no reserved reads.'}
    write_json(root/'plan.json',plan);setup_device();write_json(root/'selftest.json',selftest(root/'candidate.py'))
    started=time.monotonic()
    for scenario in scenarios:
        if time.monotonic()-started+plan['worker_timeout_seconds']>plan['study_budget_seconds']:raise TimeoutError('Registered extension budget exhausted')
        trial=root/scenario['id'];trial.mkdir();write_json(trial/'request.json',{'scenario':scenario,'plan_sha256':digest(root/'plan.json')})
        cmd=[sys.executable,'-m','experiments.navarro.v2.studies.combined_gradients','--root',str(root),'--worker',scenario['id']]
        print('START',scenario['id'],flush=True)
        with (trial/'console.log').open('x') as log:
            try:p=subprocess.run(cmd,cwd=Path.cwd(),stdout=log,stderr=subprocess.STDOUT,timeout=plan['worker_timeout_seconds']);code=p.returncode
            except subprocess.TimeoutExpired:code=-1
        if code or not (trial/'summary.json').exists():
            if (trial/'summary.json').exists():shutil.move(trial/'summary.json',trial/'incomplete-summary.json')
            write_json(trial/'summary.json',{'status':'FAILURE_OR_TIMEOUT','returncode':code,'correctness_pass':False,'metrics':None,'quality':None})
        write_json(trial/'artifacts.json',{p.name:digest(p) for p in trial.iterdir() if p.is_file()});report(root)
        print('END',scenario['id'],json.loads((trial/'summary.json').read_text())['status'],flush=True)
    write_json(root/'completion.json',{'completed_at':now(),'seconds':time.monotonic()-started,'decisions':report(root)})
    write_json(root/'artifacts.json',{p.name:digest(p) for p in root.iterdir() if p.is_file()})
    print(root)


if __name__=='__main__':main()
