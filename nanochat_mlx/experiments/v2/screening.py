"""Trained checkpoint census and activation grammar mechanism diagnostics."""
import csv
import json
from pathlib import Path
import time

import numpy as np

from .records import digest,write_json,ledger_append


def screen(root,manifest):
    from scripts.navarro_structural_census import census
    destination=root/'screening'
    if (destination/'summary.json').exists():return json.loads((destination/'summary.json').read_text())
    destination.mkdir(exist_ok=True)
    checkpoint_root=Path(manifest['artifact_root'])/'reference'
    summaries=[]
    for step in (100,500,1000):
        checkpoint=checkpoint_root/f'step-{step:04d}'
        if not checkpoint.exists():continue
        expected=json.loads((checkpoint/'hashes.json').read_text())
        for name in ('model.safetensors','optimizer.safetensors'):
            path=checkpoint/name
            if expected[str(path.resolve())]!=digest(path):raise ValueError('Checkpoint hash mismatch')
            for block_size in (32,128,512):
                records,skipped=census(path,block_size)
                filename=f'step-{step}-{name}-rows-{block_size}.csv'
                if not (destination/filename).exists():
                    with (destination/filename).open('x',newline='') as f:
                        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
                total=sum(r['dense_bytes'] for r in records)
                rejected=sum(r['dense_bytes'] for r in records if r['rejected_by_dictionary_bound'])
                summaries.append({'step':step,'file':name,'rows_per_block':block_size,'matrices':len({r['tensor'] for r in records}),
                    'blocks':len(records),'rejected_blocks':sum(r['rejected_by_dictionary_bound'] for r in records),
                    'dense_bytes':total,'rejected_dense_bytes':rejected,'maximum_savings_under_policy':1-rejected/total,
                    'csv':filename,'sha256':digest(destination/filename),'skipped':skipped})
                print('CENSUS',step,name,block_size,summaries[-1]['rejected_blocks'],'/',len(records),flush=True)
    activation=activation_probe(manifest,checkpoint_root/'step-1000') if (checkpoint_root/'step-1000').exists() else []
    write_json(destination/'activations.json',activation)
    result={'kind':'structural_screening_not_confirmatory','checkpoint_census':summaries,
        'activation_sha256':digest(destination/'activations.json'),'activation_samples':len(activation),'activation_rules':sum(r['rules'] for r in activation),
        'activation_roundtrip_pass':all(r['roundtrip_exact'] for r in activation) if activation else None,
        'limitations':['Local dictionaries only, three preregistered row block sizes.',
            'Activation probe covers first 128 tokens of four calibration documents, all MLPs; not an exhaustive activation census.',
            'No time/speedup conclusion from this structural probe.',
            'Direct sparse activation products and global dictionaries need separate implementations.']}
    write_json(destination/'summary.json',result)
    ledger_append(root/'ledger.jsonl',{'event':'structural_screening','summary_sha256':digest(destination/'summary.json')})
    return result


def activation_probe(manifest,checkpoint):
    import mlx.core as mx
    from nanochat_mlx.gpt import norm
    from nanochat_mlx.experiments.runner import build
    from nanochat_mlx.experiments.replay import load_checkpoint
    from nanochat_mlx.experiments.grammar_matrix import encode,decode_exact,resident_bytes
    model,opt=build(manifest['config'],manifest['config']['seed']);load_checkpoint(checkpoint,model,opt,manifest['config']);del opt
    tape=np.load(Path(manifest['data_root'])/'calibration_x.npy',mmap_mode='r')
    records=[]
    for document in range(4):
        ids=mx.array(tape[document:document+1]);masks=model._get_masks(ids.shape[1])
        x=norm(model.wte(ids));x0=x
        for i,block in enumerate(model.blocks):
            x=model.resid_lambdas[i]*x+model.x0_lambdas[i]*x0
            ve=model.value_embeds[str(i)](ids) if str(i) in model.value_embeds else None
            a=x+block.attn(norm(x),ve,mask=masks[i])
            p=mx.maximum(block.mlp.c_fc(norm(a)),0);h=p*p;x=a+block.mlp.c_proj(h);mx.eval(x,p,h)
            for kind,array in (('P',p),('H',h)):
                block_array=np.array(array[0,:128],copy=True)
                grammar=encode(block_array,'re_iv',128);account=resident_bytes(grammar)
                restored=decode_exact(grammar);count=int(np.count_nonzero(block_array))
                records.append({'document_index':document,'layer':i,'kind':kind,'shape':list(block_array.shape),
                    'sha256':__import__('hashlib').sha256(block_array.tobytes()).hexdigest(),
                    'roundtrip_exact':restored.tobytes()==block_array.tobytes(),
                    'positive_fraction':count/block_array.size,'rules':account['rules'],
                    'grammar_bytes':account['resident_bytes'],'dense_bytes':block_array.nbytes,
                    'bitmap_payload_estimate_bytes':4*count+4*((block_array.size+31)//32)+4*((block_array.size+255)//256)+64,
                    'encode_seconds_diagnostic_only':grammar.encode_seconds})
    return records
