"""Trained checkpoint profiling and forced-layer E3 engineering measurement."""
import json
import hashlib
from pathlib import Path
import time
import numpy as np
from .config import dump
from .grammar_matrix import encode,decode_exact,resident_bytes,matvec
from .reporting import Run
from .replay import verify_files,load_checkpoint,load_inference_checkpoint
from .measure import setup_device,memory,RSSSampler
from .runner import build

def profile_checkpoint(c,checkpoint,experiment,run_id=None):
    import mlx.core as mx
    if experiment=='E1': return activation_profile(c,checkpoint,run_id)
    setup_device();root=Path(checkpoint);verify_files(json.loads((root/'hashes.json').read_text()))
    state=json.loads((root/'state.json').read_text())
    run=Run(experiment,c,'compression_screen','pilot',state['seed'],run_id)
    filename='optimizer.safetensors' if experiment=='E2' else 'model.safetensors'
    tensors=mx.load(str(root/filename));allrows=[]
    with RSSSampler() as rss:
        for path,a in tensors.items():
            eligible=a.ndim==2 and a.size>=65536 if experiment=='E2' else '.mlp.' in path and a.ndim==2
            if not eligible:continue
            original=np.asarray(a);h=encode(original,rows_per_block=c['rows_per_block'])
            if decode_exact(h).tobytes()!=original.tobytes():raise ValueError(f'Not bit-exact: {path}')
            row=dict(path=path,step=state['step'],**resident_bytes(h));allrows.append(row);run.row('tensors.csv',row)
            run.event('tensor',**row)
            print(f'{experiment} checkpoint={state["step"]} {path}: {row["resident_bytes"]/row["raw_bytes"]:.4f} RAW blocks={row["raw_blocks"]}/{row["blocks"]}',flush=True)
            del h,original
    ratio=sum(r['resident_bytes'] for r in allrows)/sum(r['raw_bytes'] for r in allrows)
    dump(run.root/'correctness.json',{'bit_exact_all_eligible':True,'tensor_count':len(allrows)})
    run.finish(status='measured',correctness_pass=True,algorithm_fidelity=True,intervention={'eligible_resident_ratio':ratio,'eligible_tensors':len(allrows),'selected_tensors':sum(r['resident_bytes']<=.95*r['raw_bytes'] for r in allrows),'adaptive_policy':'NO_ELIGIBLE_LAYERS' if experiment=='E3' and not any(r['resident_bytes']<=.95*r['raw_bytes'] for r in allrows) else None,'encode_seconds':sum(r['encode_seconds'] for r in allrows),'process_tree_rss_peak_bytes':rss.peak,**memory()},decision='INCONCLUSIVE',limitations=['Representation screening only; complete-model pilot required for a negative verdict.','Independent compiled frequency-scan RePair, not author linear-time compressor.'])
    return run.root

def inference_run(c,checkpoint,arm='dense_native',run_id=None,prompt_count=8,prompt_length=512):
    import mlx.core as mx
    from nanochat_mlx.engine import KVCache
    from .grammar_linear import GrammarLinear
    setup_device();model,opt=build(c,17);del opt;s=load_inference_checkpoint(checkpoint,model,c)
    if s['step']!=1000 or s['seed']!=17:raise ValueError('E3 principal checkpoint must be native step1000 seed17')
    run=Run('E3',c,arm,'pilot',17,run_id)
    x=np.load(Path(c['artifact_dir'])/'val_x.npy',mmap_mode='r')
    try:
        from mlx.utils import tree_flatten
        conversion=0.
        weights_resident_bytes=sum(v.nbytes for _,v in tree_flatten(model.parameters()))
        if arm!='dense_native':
            tick=time.perf_counter();i=c['depth']//2
            replacement=GrammarLinear(model.blocks[i].mlp.c_fc.weight,mode=arm,variant='re_iv',rows_per_block=c['rows_per_block'])
            original_bytes=model.blocks[i].mlp.c_fc.weight.nbytes
            model.blocks[i].mlp.c_fc=replacement
            weights_resident_bytes+=replacement.accounting()['resident_bytes']-original_bytes
            mx.eval(model.parameters());mx.synchronize();conversion=time.perf_counter()-tick
            run.row('tensors.csv',dict(path=f'blocks.{i}.mlp.c_fc.weight',step=1000,**replacement.accounting()))
        run.event('load_and_conversion_memory',conversion_seconds=conversion,**memory())
        # Discardable prompt warmup, fresh KV cache for every measured prompt.
        ids=mx.array(x[32:33,:1]);warm=model(ids,kv_cache=KVCache(c['depth'],model.window_sizes));mx.eval(warm)
        mx.reset_peak_memory();records=[]
        with RSSSampler() as rss:
            for p in range(prompt_count):
                seq=x[32+p];cache=KVCache(c['depth'],model.window_sizes)
                tick=time.perf_counter();logits=model(mx.array(seq[None,:prompt_length]),kv_cache=cache);mx.eval(logits);mx.synchronize();prefill=time.perf_counter()-tick
                row={'prompt':p,'prompt_ids':seq[:prompt_length].tolist(),'continuation_ids':seq[prompt_length:prompt_length+128].tolist(),'prompt_sha256':hashlib.sha256(seq[:prompt_length].tobytes()).hexdigest(),'prompt_length':prompt_length,'prefill_seconds':prefill,'ttft_seconds':prefill,'decode_seconds':[],'greedy_ids':[]}
                # Fixed valid continuation; no EOS early exit in teacher-forced timing.
                for k in range(128):
                    tick=time.perf_counter();logits=model(mx.array([[int(seq[prompt_length+k])]]),kv_cache=cache);mx.eval(logits);mx.synchronize();dt=time.perf_counter()-tick
                    row['decode_seconds'].append(dt);row['greedy_ids'].append(int(mx.argmax(logits[0,-1]).item()))
                    run.row('steps.csv',dict(step=p*128+k,seconds=dt,tokens=1,**memory()))
                records.append(row);run.event('teacher_forced_prompt',**row)
                print(f'E3 {arm} prompt {p+1}/{prompt_count} prefill={prefill:.3f}s decode={sum(row["decode_seconds"]):.3f}s',flush=True)
        # Separate free greedy generation with natural stop tokens, no calculator execution.
        from nanochat_mlx.tokenizer import get_tokenizer
        tokenizer=get_tokenizer();stops={tokenizer.get_bos_token_id(),tokenizer.encode_special('<|assistant_end|>')}
        for p in range(prompt_count):
            seq=x[32+p];cache=KVCache(c['depth'],model.window_sizes)
            logits=model(mx.array(seq[None,:prompt_length]),kv_cache=cache);mx.eval(logits)
            generated=[];tick=time.perf_counter()
            for k in range(128):
                token=int(mx.argmax(logits[0,-1]).item());generated.append(token)
                if token in stops:break
                logits=model(mx.array([[token]]),kv_cache=cache);mx.eval(logits)
            mx.synchronize()
            run.event('free_greedy_generation',prompt=p,ids=generated,effective_tokens=len(generated),natural_stop=generated[-1] in stops,decode_seconds=time.perf_counter()-tick)
        times=np.array([v for r in records for v in r['decode_seconds']])
        run.finish(status='measured',correctness_pass=None,algorithm_fidelity=True,decision='INCONCLUSIVE',intervention={'mean_decode_seconds':float(times.mean()),'p50_decode_seconds':float(np.quantile(times,.5)),'p95_decode_seconds':float(np.quantile(times,.95)),'tokens_per_second':float(1/times.mean()),'mean_ttft_seconds':float(np.mean([r['ttft_seconds'] for r in records])),'mean_request_seconds':float(np.mean([r['prefill_seconds']+sum(r['decode_seconds']) for r in records])),'conversion_seconds':conversion,'weights_resident_bytes':weights_resident_bytes,'prompt_count':prompt_count,'decode_tokens':128*prompt_count,'process_tree_rss_peak_bytes':rss.peak,**memory()},limitations=['Forced central c_fc, CPU transfers included.','Teacher-forced timing; separate free greedy generations with natural EOS are in events.jsonl.','One pilot process; no process-paired CI.'])
        return run.root
    except BaseException as e:
        run.event('error',error=repr(e));run.finish(status='INCONCLUSIVE',decision='INCONCLUSIVE',failure_reason=repr(e));raise


def activation_profile(c,checkpoint,run_id=None):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from nanochat_mlx.gpt import loss_fn
    from .activation_tape import loss_and_grad
    from .metrics import errors
    setup_device();model,opt=build(c,17);state=load_checkpoint(checkpoint,model,opt,c);del opt
    run=Run('E1',c,'trained_activation_profile','pilot',17,run_id)
    tx=np.load(Path(c['artifact_dir'])/'train_x.npy',mmap_mode='r');ty=np.load(Path(c['artifact_dir'])/'train_y.npy',mmap_mode='r')
    index=min(state['step'],999)*c['accumulation'];x=mx.array(tx[index]);y=mx.array(ty[index])
    loss,g=nn.value_and_grad(model,loss_fn)(model,x,y);mx.eval(loss,g)
    value,h=loss_and_grad(model,x,y,'tape_bitmap');mx.eval(value,h)
    ref=dict(tree_flatten(g));test=dict(tree_flatten(h));results={}
    for path in ref: results[path]=errors(np.asarray(test[path]),np.asarray(ref[path]),1e-5,1e-3)
    passed=all(e['pass'] and e['relative_l2']<=1e-3 for e in results.values()) and abs(loss.item()-value.item())<=1e-4
    for row in model._navarro_accounting:run.row('tensors.csv',dict(step=state['step'],**row))
    dump(run.root/'correctness.json',{'pass':passed,'per_parameter':results,'loss_delta':value.item()-loss.item()})
    rows=model._navarro_accounting
    run.finish(status='measured' if passed else 'FAILURE_CORRECTNESS',correctness_pass=passed,algorithm_fidelity=True,decision='INCONCLUSIVE' if passed else 'FAILURE_CORRECTNESS',intervention={'activation_resident_ratio':sum(r['resident_bytes'] for r in rows)/sum(r['raw_bytes'] for r in rows),'step':state['step'],**memory()},limitations=['Detailed trained-state profile; not a principal timing process.'])
    return run.root
