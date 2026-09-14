"""Untimed correctness and timing diagnostics, distinct from principal arms."""
import json
from pathlib import Path
import time
import numpy as np
from .config import ROOT,dump
from .metrics import errors
from .reporting import Run
from .runner import build
from .replay import load_checkpoint,load_inference_checkpoint
from .measure import setup_device,memory

def e3_correctness(c,checkpoint):
    import mlx.core as mx
    from nanochat_mlx.experiments.grammar_linear import GrammarLinear
    setup_device();a,opt=build(c,17);del opt;load_inference_checkpoint(checkpoint,a,c)
    run=Run('E3',c,'trained_model_correctness','pilot',17,'primary-E3-correctness')
    rows=np.load(Path(c['artifact_dir'])/'val_x.npy',mmap_mode='r');targets=np.load(Path(c['artifact_dir'])/'val_y.npy',mmap_mode='r')
    i=c['depth']//2;weight=a.blocks[i].mlp.c_fc.weight
    replacement=GrammarLinear(weight,variant='re_iv')
    local=[];rng=np.random.default_rng(20260913);w=np.asarray(weight)
    from .grammar_matrix import matvec
    for k in range(8):
        x=rng.normal(size=w.shape[1]).astype(np.float32)
        local.append(errors(matvec(replacement._handle,x),w.astype(np.float64)@x,1e-5,1e-4))
    # Local errors near zero can exceed elementwise relative error while absolute is valid.
    records=[]
    for p in range(8):
        ids=mx.array(rows[32+p:33+p,:512]);y=mx.array(targets[32+p:33+p,:512])
        native=a(ids);nll=a(ids,targets=y);mx.eval(native,nll)
        original=a.blocks[i].mlp.c_fc;a.blocks[i].mlp.c_fc=replacement
        direct=a(ids);nll_d=a(ids,targets=y);mx.eval(direct,nll_d)
        a.blocks[i].mlp.c_fc=original
        error=errors(np.asarray(direct),np.asarray(native),1e-4,1e-3)
        records.append({'prompt':p,'logits':error,'nll_native':nll.item(),'nll_direct':nll_d.item(),'delta_nll':nll_d.item()-nll.item()})
    passed=all(e['pass'] for e in local) and all(r['logits']['pass'] and abs(r['delta_nll'])<=1e-4 for r in records)
    dump(run.root/'correctness.json',{'pass':passed,'local':local,'model':records})
    # Microbenchmark: include MLX host boundaries for direct, no dense reconstruction.
    times={}
    ids=mx.array(rng.normal(size=(1,w.shape[1])).astype(np.float32))
    cpu_vector=np.asarray(ids)[0].copy()
    for arm in ('dense_native','dense_cpu_diagnostic','grammar_direct_cpu'):
        durations=[]
        for rep in range(21):
            tick=time.perf_counter()
            if arm=='dense_cpu_diagnostic':out=w@cpu_vector
            else:
                out=ids@weight.T if arm=='dense_native' else replacement(ids);mx.eval(out);mx.synchronize()
            dt=time.perf_counter()-tick
            if rep:durations.append(dt)
        times[arm]={'mean_seconds':float(np.mean(durations)),'p95_seconds':float(np.quantile(durations,.95))}
    run.finish(status='measured' if passed else 'FAILURE_CORRECTNESS',correctness_pass=passed,algorithm_fidelity=True,decision='INCONCLUSIVE' if passed else 'FAILURE_CORRECTNESS',intervention={'microbenchmark':times,'mean_nll_delta':float(np.mean([r['delta_nll'] for r in records])),**memory()},limitations=['Separate correctness/microbenchmark process; not end-to-end timing.','Local CPU FP64 oracle and full-model FP32 logits on eight identical reserved prefixes.'])
    return run.root

def native_equivalence(c,steps=100,harness_path=None):
    import mlx.core as mx
    setup_device()
    original=Path(c['artifact_dir'])/f'repo-native-s101-{steps}'/'mlx_checkpoints'/f'd{c["depth"]}'
    harness=Path(harness_path) if harness_path else Path(c['artifact_dir'])/'checkpoints'/'primary-pilot-native-s101'/f'step-{steps:04d}'
    result={}
    if (original.parents[1]/'initial.safetensors').exists():
        ai=mx.load(str(original.parents[1]/'initial.safetensors'));bi=mx.load(str(Path(c['artifact_dir'])/'initial-s101/model.safetensors'))
        result['initial']={k:{'bit_exact':np.asarray(v).tobytes()==np.asarray(bi[k]).tobytes(),**errors(np.asarray(v),np.asarray(bi[k]),1e-5,1e-3)} for k,v in ai.items()}
    for label,afile,bfile in [('parameters',original/f'step_{steps:06d}.safetensors',harness/'model.safetensors'),('optimizer',original/f'step_{steps:06d}_optim.safetensors',harness/'optimizer.safetensors')]:
        a=mx.load(str(afile));b=mx.load(str(bfile));assert a.keys()==b.keys()
        result[label]={}
        for key in a:
            av=np.asarray(a[key]);bv=np.asarray(b[key])
            result[label][key]={'bit_exact':av.tobytes()==bv.tobytes(),**errors(av,bv,1e-5,1e-3)}
    passed=all(e['bit_exact'] for group in result.values() for e in group.values())
    dump(ROOT/f'experiments/navarro/results/native-equivalence-{steps}{"-layout" if harness_path else ""}.json',{'pass':passed,'step':steps,'seed':101,'original':str(original),'harness':str(harness),'per_tensor':result})
    return passed

def native_variability(c):
    import mlx.core as mx
    from nanochat_mlx.train import _load_weights_into_model
    setup_device();root=Path(c['artifact_dir']);paths={
        'original':root/'repo-native-s101-10/mlx_checkpoints/d8/step_000010.safetensors',
        'original_repeat':root/'repo-native-s101-10-repeat/mlx_checkpoints/d8/step_000010.safetensors',
        'harness_contiguous':root/'checkpoints/primary-pilot-native-s101/step-0010/model.safetensors',
        'harness_original_views':root/'checkpoints/baseline/native-layout-check/step-0010/model.safetensors'}
    arrays={k:mx.load(str(p)) for k,p in paths.items()};records={}
    for label in list(paths)[1:]:
        records[label]={'parameters':{key:{'bit_exact':np.asarray(v).tobytes()==np.asarray(arrays['original'][key]).tobytes(),**errors(np.asarray(v),np.asarray(arrays['original'][key]),1e-5,1e-3)} for key,v in arrays[label].items()}}
    model,opt=build(c,101);del opt
    x=np.load(root/'val_x.npy',mmap_mode='r');y=np.load(root/'val_y.npy',mmap_mode='r');ref=None
    for label,path in paths.items():
        _load_weights_into_model(model,str(path));out=model(mx.array(x[32:33]),targets=None);loss=model(mx.array(x[32:33]),targets=mx.array(y[32:33]));mx.eval(out,loss)
        if ref is None:ref=np.asarray(out).copy();base=loss.item()
        else:records[label].update(logits=errors(np.asarray(out),ref,1e-4,1e-3),nll_delta=loss.item()-base)
    dump(ROOT/'experiments/navarro/results/native-variability.json',{'seed':101,'steps':10,'paths':{k:str(v) for k,v in paths.items()},'comparisons':records})
    print(json.dumps({k:{'max_parameter_difference':max(e['max_abs'] for e in v['parameters'].values()),'logits':v['logits'],'nll_delta':v['nll_delta']} for k,v in records.items()},indent=2))

def external_checkpoint(c,run_id=None):
    import mlx.core as mx
    from nanochat_mlx.gpt import GPT,GPTConfig
    from nanochat_mlx.train import _load_weights_into_model
    from .grammar_matrix import encode,decode_exact,resident_bytes
    from .grammar_linear import GrammarLinear
    from .config import sha256
    setup_device();base=Path(c['data_dir'])/'mlx_checkpoints/d4_sft'
    metadata=base/'step_257033_meta.json';weights=base/'step_257033.safetensors'
    if not metadata.exists() or not weights.exists():raise FileNotFoundError('Recorded external checkpoint unavailable')
    meta=json.loads(metadata.read_text());cfg=GPTConfig(sequence_len=meta['sequence_len'],vocab_size=meta['vocab_size'],n_layer=meta['depth'],n_head=meta['n_head'],n_kv_head=meta['n_kv_head'],n_embd=meta['n_embd'],window_pattern=meta['window_pattern'])
    model=GPT(cfg);_load_weights_into_model(model,str(weights));mx.eval(model.parameters())
    run=Run('E3',c,'external_sft_validation','pilot',17,run_id or 'external-E3-sft-d4')
    run.event('external_checkpoint',metadata=meta,weights_path=str(weights),weights_sha256=sha256(weights),metadata_sha256=sha256(metadata))
    rows=[]
    for i,b in enumerate(model.blocks):
        for name in ('c_fc','c_proj'):
            w=getattr(b.mlp,name).weight;h=encode(w)
            assert decode_exact(h).tobytes()==np.asarray(w).tobytes()
            row=dict(path=f'blocks.{i}.mlp.{name}.weight',step=meta['step'],**resident_bytes(h));rows.append(row);run.row('tensors.csv',row)
    index=meta['depth']//2;original=model.blocks[index].mlp.c_fc;replacement=GrammarLinear(original.weight,variant='re_iv')
    from .grammar_matrix import matvec
    weight=np.asarray(original.weight);local=[];rng=np.random.default_rng(20260913)
    for _ in range(8):
        vector=rng.normal(size=weight.shape[1]).astype(np.float32)
        local.append(errors(matvec(replacement._handle,vector),weight.astype(np.float64)@vector,1e-5,1e-4))
    decoded=GrammarLinear(original.weight,mode='compressed_decode_dense',variant='re_iv')
    x=np.load(Path(c['artifact_dir'])/'val_x.npy',mmap_mode='r');y=np.load(Path(c['artifact_dir'])/'val_y.npy',mmap_mode='r');checks=[]
    for i in range(8):
        ids=mx.array(x[32+i:33+i,:512]);targets=mx.array(y[32+i:33+i,:512])
        native=model(ids);nl=model(ids,targets=targets);mx.eval(native,nl)
        model.blocks[index].mlp.c_fc=replacement
        direct=model(ids);dl=model(ids,targets=targets);mx.eval(direct,dl)
        model.blocks[index].mlp.c_fc=decoded
        dense_reconstruction=model(ids);mx.eval(dense_reconstruction)
        model.blocks[index].mlp.c_fc=original
        checks.append({'prompt':i,'logits':errors(np.asarray(direct),np.asarray(native),1e-4,1e-3),'decoded_dense_logits':errors(np.asarray(dense_reconstruction),np.asarray(native),1e-4,1e-3),'nll_delta':dl.item()-nl.item()})
    passed=all(e['pass'] for e in local) and all(e['logits']['pass'] and abs(e['nll_delta'])<=1e-4 for e in checks)
    dump(run.root/'correctness.json',{'pass':passed,'local_fp64_oracle':local,'per_prompt':checks})
    run.finish(status='measured' if passed else 'FAILURE_CORRECTNESS',correctness_pass=passed,algorithm_fidelity=True,decision='INCONCLUSIVE' if passed else 'FAILURE_CORRECTNESS',intervention={'eligible_tensors':len(rows),'selected_tensors':sum(r['resident_bytes']<=.95*r['raw_bytes'] for r in rows),'resident_ratio':sum(r['resident_bytes'] for r in rows)/sum(r['raw_bytes'] for r in rows),'parameters':model.num_scaling_params()['total'],'step':meta['step']},limitations=['External depth4 SFT checkpoint, not the preregistered depth8 reference.','Existing tokenizer has compatible vocabulary size; checkpoint does not record tokenizer hash or full training provenance.','Representation and teacher-forced model correctness only; no external latency SUCCESS claimed.','Local FP64 oracle and decoded-dense control isolate lossless storage from changed FP32 reduction order; tolerances unchanged.'])
    return run.root
