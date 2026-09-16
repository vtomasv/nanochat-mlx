"""Registered mechanism study: exact CSR on trained H, direct products and VJP.

Separate from immutable discovery. Microbenchmarks do not establish full-training
speedups. No held-out data is opened. Counterfactual cotangents are labelled.
"""
import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import mlx.core as mx

from nanochat_mlx.experiments.v2.records import ROOT,digest,write_json,now
from nanochat_mlx.experiments.metrics import errors


@lru_cache(None)
def kernels():
    scatter=mx.fast.metal_kernel(name='navarro_h_csr_scatter',
        input_names=['p','prefix','shape'],output_names=['values','columns'],source='''
        uint i=thread_position_in_grid.x;
        if(i>=shape[0]*shape[1]) return;
        uint v=p[i];
        if(v && v!=0x80000000u) { uint k=prefix[i]-1;values[k]=v;columns[k]=i%shape[1]; }
        ''',compile_options={'math_mode':'safe'})
    product=mx.fast.metal_kernel(name='navarro_h_csr_product',
        input_names=['values','columns','offsets','w','shape'],output_names=['out'],source='''
        uint element=thread_position_in_grid.x/32, lane=thread_index_in_simdgroup;
        uint row=element/shape[2], column=element%shape[2];
        if(row>=shape[0]) return;
        float sum=0.0f;
        for(uint j=offsets[row]+lane;j<offsets[row+1];j+=32)
            sum+=as_type<float>(values[j])*w[column*shape[1]+columns[j]];
        sum=metal::simd_sum(sum);
        if(lane==0) out[element]=(threads_per_simdgroup==32)?sum:NAN;
        ''',compile_options={'math_mode':'safe'})
    return scatter,product


@dataclass
class CSR:
    shape: tuple
    values: object
    columns: object
    offsets: object

    @property
    def bytes(self):return self.values.nbytes+self.columns.nbytes+self.offsets.nbytes+64

    def exact_decode(self):
        # Codec oracle only, never part of operator timing.
        out=np.zeros(self.shape,np.uint32);v=np.asarray(self.values);c=np.asarray(self.columns);o=np.asarray(self.offsets)
        for i in range(self.shape[0]):out[i,c[o[i]:o[i+1]]]=v[o[i]:o[i+1]]
        return out.view(np.float32)


def pack(a):
    if a.ndim!=2 or a.dtype!=mx.float32:raise TypeError('FP32 matrix required')
    if a.size>=2**32-1:raise ValueError('CSR index overflow')
    a=mx.contiguous(a);p=a.view(mx.uint32)
    invalid=mx.any(((p&0x7f800000)==0x7f800000)|(((p>>31)!=0)&(p!=0x80000000)))
    nz=((p!=0)&(p!=0x80000000)).reshape(-1).astype(mx.uint32)
    if not a.size:
        z=mx.zeros((0,),mx.uint32);return CSR(a.shape,z,z,mx.zeros((a.shape[0]+1,),mx.uint32))
    prefix=mx.cumsum(nz);total=prefix[-1];mx.eval(invalid,total)
    if invalid.item():raise ValueError('CSR domain: finite nonnegative values')
    count=int(total.item())
    offsets=mx.concatenate([mx.zeros((1,),mx.uint32),prefix[a.shape[1]-1::a.shape[1]]])
    if count:
        values,columns=kernels()[0](inputs=[p,prefix,mx.array(a.shape,mx.uint32)],grid=(a.size,1,1),threadgroup=(256,1,1),
            output_shapes=[(count,)]*2,output_dtypes=[mx.uint32]*2)
    else:values=columns=mx.zeros((0,),mx.uint32)
    mx.eval(values,columns,offsets)
    return CSR(a.shape,values,columns,offsets)


def product(csr,w):
    if w.ndim!=2 or w.shape[1]!=csr.shape[1] or w.dtype!=mx.float32:raise ValueError('Product shape/dtype mismatch')
    rows,inner=csr.shape;outcols=w.shape[0]
    if not rows or not outcols:return mx.zeros((rows,outcols),mx.float32)
    return kernels()[1](inputs=[csr.values,csr.columns,csr.offsets,w,mx.array([rows,inner,outcols],mx.uint32)],
        grid=(rows*outcols*32,1,1),threadgroup=(256,1,1),output_shapes=[(rows,outcols)],output_dtypes=[mx.float32])[0]


def operation(h,w,g,compact):
    if compact:
        a=pack(h);at=pack(mx.contiguous(h.T))
        return product(a,w),product(at,mx.contiguous(g.T)).T,g@w
    return h@w.T,g.T@h,g@w


def selftest():
    rng=np.random.default_rng(7129);records=[]
    for shape in ((0,3),(1,1),(3,33),(7,64)):
        a=np.maximum(rng.normal(size=shape).astype(np.float32),0)
        if a.size:a.flat[0]=np.nextafter(np.float32(0),np.float32(1))
        h=mx.array(a);csr=pack(h);ok=csr.exact_decode().tobytes()==a.tobytes()
        if not ok:raise AssertionError('CSR exact roundtrip')
        if not a.size:continue
        w=mx.array(rng.normal(size=(5,shape[1])).astype(np.float32)*.1)
        g=mx.array(rng.normal(size=(shape[0],5)).astype(np.float32)*.1)
        outputs=operation(h,w,g,True);mx.eval(outputs)
        _,derivatives=mx.vjp(lambda x,y:x@y.T,[h,w],[g]);mx.eval(derivatives)
        checks=[errors(np.asarray(outputs[0]),np.asarray(h@w.T)),
                errors(np.asarray(outputs[1]),np.asarray(derivatives[1]),rtol=1e-3),
                errors(np.asarray(outputs[2]),np.asarray(derivatives[0]),rtol=1e-3)]
        if not all(r['pass'] for r in checks):raise AssertionError(checks)
        records.append({'shape':shape,'codec_exact':ok,'operator_vjp':checks})
    for bad in (float('nan'),float('inf'),-1.):
        try:pack(mx.array([[bad]]))
        except ValueError:pass
        else:raise AssertionError('Invalid domain accepted')
    return records


def benchmark(h,w,seed,repeats):
    rng=np.random.default_rng(seed)
    # Controlled cotangent probes: not actual downstream GPT loss gradients.
    g=mx.array(rng.normal(size=(h.shape[0],w.shape[0])).astype(np.float32)*.01)
    dense=operation(h,w,g,False);sparse=operation(h,w,g,True);mx.eval(dense,sparse)
    comparisons={name:errors(np.asarray(a),np.asarray(b),rtol=1e-4 if name=='forward' else 1e-3)
        for name,a,b in zip(('forward','dW2','dH'),sparse,dense)}
    passed=all(r['pass'] and r['relative_l2']<=1e-3 for r in comparisons.values())
    hc=pack(h);htc=pack(mx.contiguous(h.T))
    codec_exact=hc.exact_decode().tobytes()==np.asarray(h).tobytes()
    def resident():return product(hc,w),product(htc,mx.contiguous(g.T)).T,g@w
    times={name:[] for name in ('dense','csr_resident','csr_with_build')}
    callbacks={'dense':lambda:operation(h,w,g,False),'csr_resident':resident,'csr_with_build':lambda:operation(h,w,g,True)}
    # Equal warmup for all paths; separate synchronization for each measurement.
    for callback in callbacks.values():mx.eval(callback());mx.synchronize()
    for repeat in range(repeats):
        order=list(callbacks);rng.shuffle(order)
        for name in order:
            mx.synchronize();tick=time.perf_counter();out=callbacks[name]();mx.eval(out);mx.synchronize()
            times[name].append(time.perf_counter()-tick);del out
    return {'correctness_pass':passed and codec_exact,'codec_exact':codec_exact,'errors':comparisons,
        'h_shape':list(h.shape),'w_shape':list(w.shape),'nonzero_fraction':hc.values.size/h.size,
        'csr_forward_bytes':hc.bytes,'csr_both_orientations_bytes':hc.bytes+htc.bytes,'dense_h_bytes':h.nbytes,
        'seconds':times,'mean_seconds':{k:float(np.mean(v)) for k,v in times.items()},
        'cotangent':'IID normal(0,0.01), diagnostic VJP input; not captured full-model loss gradient',
        'evidence':'one process, repeated operators; descriptive only, no process-level confidence interval'}


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--campaign',default='experiments/navarro/v2/20260915')
    parser.add_argument('--output',default='experiments/navarro/v2/20260915/mechanisms');args=parser.parse_args()
    root=Path(args.campaign);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    m=json.loads((root/'manifest.json').read_text());checkpoint=Path(m['artifact_root'])/'reference/step-1000'
    plan={'created_at':now(),'study':'V2-H direct CSR diagnostic and expanded V2-S activation sample',
        'source_sha256':digest(Path(__file__)),'campaign_manifest_sha256':digest(root/'manifest.json'),
        'checkpoint_hashes_sha256':digest(checkpoint/'hashes.json'),
        'documents':32,'positions_per_document':128,'layers':'all','grammar_objects':['P','H'],
        'direct_product_documents':4,'micro_repeats':5,'seed':7129,'timeout_seconds':900,
        'selection':'first 32 calibration documents, all layers, first 128 positions; no reserved reads',
        'scope':'Mechanisms only. No full-training speedup or quality claim. No threshold amendments.'}
    write_json(out/'plan.json',plan)
    from nanochat_mlx.experiments.v2.worker import protect_reserved
    protect_reserved(Path(m['data_root']))
    from nanochat_mlx.experiments.v2.data import verify
    verify(Path(m['data_root']))
    from nanochat_mlx.experiments.measure import setup_device,environment
    from nanochat_mlx.experiments.runner import build
    from nanochat_mlx.experiments.replay import load_checkpoint
    from nanochat_mlx.experiments.grammar_matrix import encode,decode_exact,resident_bytes
    from nanochat_mlx.gpt import norm
    setup_device();write_json(out/'environment.json',environment());write_json(out/'selftest.json',selftest())
    model,opt=build(m['config'],m['config']['seed']);load_checkpoint(checkpoint,model,opt,m['config']);del opt
    tape=np.load(Path(m['data_root'])/'calibration_x.npy',mmap_mode='r');target=np.load(Path(m['data_root'])/'calibration_y.npy',mmap_mode='r')
    records=[];direct=[];started=time.monotonic()
    with (out/'activations.jsonl').open('x') as activation_file,(out/'operators.jsonl').open('x') as operator_file:
        for document in range(plan['documents']):
            ids=mx.array(tape[document:document+1]);masks=model._get_masks(ids.shape[1]);x=norm(model.wte(ids));x0=x
            valid=min(128,int(np.count_nonzero(target[document]!=-1)))
            for i,block in enumerate(model.blocks):
                x=model.resid_lambdas[i]*x+model.x0_lambdas[i]*x0
                ve=model.value_embeds[str(i)](ids) if str(i) in model.value_embeds else None
                a=x+block.attn(norm(x),ve,mask=masks[i]);p=mx.maximum(block.mlp.c_fc(norm(a)),0);h=p*p
                x=a+block.mlp.c_proj(h);mx.eval(x,p,h)
                for kind,array in (('P',p),('H',h)):
                    block_array=np.array(array[0,:valid],copy=True)
                    grammar=encode(block_array,'re_iv',128);account=resident_bytes(grammar)
                    record={'document':document,'layer':i,'kind':kind,'shape':list(block_array.shape),
                        'sha256':hashlib.sha256(block_array.tobytes()).hexdigest(),'valid_positions':valid,
                        'nonzero_fraction':float(np.count_nonzero(block_array)/block_array.size),
                        'codec_exact':decode_exact(grammar).tobytes()==block_array.tobytes(),**account}
                    activation_file.write(json.dumps(record)+'\n');activation_file.flush();records.append(record)
                if document<plan['direct_product_documents']:
                    result=benchmark(mx.contiguous(h[0,:valid]),block.mlp.c_proj.weight,7129+document*8+i,plan['micro_repeats'])
                    result.update(document=document,layer=i);operator_file.write(json.dumps(result)+'\n');operator_file.flush();direct.append(result)
                if time.monotonic()-started>plan['timeout_seconds']:raise TimeoutError('Registered mechanism budget exhausted')
            print('MECHANISM document',document+1,'/',plan['documents'],flush=True)
    summary={'status':'MEASURED_MECHANISM','activation_samples':len(records),'direct_operator_cases':len(direct),
        'all_codec_exact':all(r['codec_exact'] for r in records),'all_operator_cases_pass':all(r['correctness_pass'] for r in direct),
        'failed_cases':[{k:r[k] for k in ('document','layer','errors')} for r in direct if not r['correctness_pass']],
        'grammar_rules':sum(r['rules'] for r in records),'seconds':time.monotonic()-started,
        'limitations':['Cotangents are synthetic probes.','One process; no confidence interval or general speedup claim.',
            'First 32 documents, capped valid positions, one trained checkpoint; not stratified by source.',
            'CSR stores both orientations for forward/dW2; construction and both buffers are counted.',
            'No custom MLX autodiff registration or full GPT integration for this operator.']}
    write_json(out/'summary.json',summary)
    if digest(Path(__file__))!=plan['source_sha256']:raise ValueError('Study source changed')
    write_json(out/'artifacts.json',{p.name:digest(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
