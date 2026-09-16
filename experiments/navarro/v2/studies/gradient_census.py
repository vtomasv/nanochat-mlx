"""Native gradient census before and after accumulation, without optimizer edits."""
import csv
import json
from pathlib import Path
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten,tree_map
from nanochat_mlx.experiments.v2.records import digest,write_json,now
from nanochat_mlx.experiments.runner import build
from nanochat_mlx.experiments.replay import load_inference_checkpoint
from nanochat_mlx.experiments.grammar_matrix import encode,decode_exact,resident_bytes
from nanochat_mlx.gpt import loss_fn


def census(tree,step,phase):
    rows=[];summaries=[];samples=[]
    for name,value in tree_flatten(tree):
        if value.ndim!=2 or value.dtype!=mx.float32:continue
        array=np.asarray(value);bits=array.view(np.uint32);tensor=[]
        for start in range(0,len(array),128):
            block=bits[start:start+128];nz=block[block!=0];distinct=len(np.unique(nz));lower=64+4*distinct
            special=int(np.count_nonzero((block==0x80000000)|((block&0x7f800000)==0x7f800000)))
            record={'step':step,'phase':phase,'tensor':name,'row_start':start,'rows':len(block),'cols':block.shape[1],
                'dense_bytes':block.nbytes,'positive_zero_cells':int(block.size-nz.size),'cells':block.size,
                'special_cells':special,'dictionary_lower_bytes':lower,'rejected':lower*100>block.nbytes*95}
            rows.append(record);tensor.append(record)
        total=sum(r['dense_bytes'] for r in tensor);rejected=sum(r['dense_bytes'] for r in tensor if r['rejected'])
        summaries.append({'step':step,'phase':phase,'tensor':name,'dense_bytes':total,
            'zero_fraction':sum(r['positive_zero_cells'] for r in tensor)/sum(r['cells'] for r in tensor),
            'special_cells':sum(r['special_cells'] for r in tensor),'maximum_savings_under_raw_policy':1-rejected/total})
        # One preregistered representative per tensor, at most first eight
        # tensors eligible by dictionary bound; samples are not a census ratio.
        eligible=[r for r in tensor if not r['rejected']]
        if eligible and len(samples)<8:
            chosen=min(eligible,key=lambda r:(r['dictionary_lower_bytes']/r['dense_bytes'],r['row_start']))
            start=chosen['row_start'];block=np.array(array[start:start+128],copy=True)
            h=encode(block,'adaptive',128);account=resident_bytes(h)
            samples.append({'step':step,'phase':phase,'tensor':name,'row_start':start,
                'roundtrip_exact':decode_exact(h).tobytes()==block.tobytes(),**account})
    return rows,summaries,samples


def main():
    root=Path('experiments/navarro/v2/20260915');m=json.loads((root/'manifest.json').read_text());out=root/'gradient-census';out.mkdir(exist_ok=False)
    plan={'created_at':now(),'source_sha256':digest(Path(__file__)),'campaign_manifest_sha256':digest(root/'manifest.json'),
        'steps':[0,10,100,500,1000],'phases':['microbatch_0','microbatch_1','accumulated_average'],
        'rows_per_block':128,'data':'Native next two replay microbatches at each checkpoint; no optimizer update.',
        'sample_rule':'First eight eligible tensors in tree order; lowest dictionary-bound-ratio block per tensor, row-start tie break.',
        'interpretation':'Gradient structural census only. Sampled grammar ratios are not extrapolated to full gradients.',
        'scope':'No performance conclusion, no reserved reads, no mutation of saved model.'}
    write_json(out/'plan.json',plan)
    from nanochat_mlx.experiments.v2.worker import protect_reserved
    from nanochat_mlx.experiments.v2.data import verify
    from nanochat_mlx.experiments.measure import setup_device,environment
    protect_reserved(Path(m['data_root']));verify(Path(m['data_root']));setup_device();write_json(out/'environment.json',environment())
    c=m['config'];model,opt=build(c,c['seed']);del opt
    x=np.load(Path(m['data_root'])/'train_x.npy',mmap_mode='r');y=np.load(Path(m['data_root'])/'train_y.npy',mmap_mode='r')
    summaries=[];samples=[];losses=[];started=time.monotonic()
    with (out/'blocks.csv').open('x',newline='') as f:
        writer=None
        for step in plan['steps']:
            checkpoint=Path(m['initial_checkpoint']) if step==0 else Path(m['artifact_root'])/'reference'/f'step-{step:04d}'
            load_inference_checkpoint(checkpoint,model,c)
            gradients=[]
            for micro in range(2):
                loss,g=nn.value_and_grad(model,loss_fn)(model,mx.array(x[step*2+micro]),mx.array(y[step*2+micro]));mx.eval(loss,g)
                losses.append({'step':step,'microbatch':micro,'nll':loss.item()});gradients.append(g)
            gradients.append(tree_map(lambda a,b:(a+b)/2,gradients[0],gradients[1]));mx.eval(gradients[-1])
            for phase,g in zip(plan['phases'],gradients):
                rows,summary,sample=census(g,step,phase)
                if writer is None:writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
                writer.writerows(rows);f.flush();summaries.extend(summary);samples.extend(sample)
            del gradients,g;mx.synchronize();print('GRADIENT CENSUS',step,flush=True)
    write_json(out/'tensors.json',summaries);write_json(out/'grammar-samples.json',samples);write_json(out/'losses.json',losses)
    aggregate=[]
    for step in plan['steps']:
        for phase in plan['phases']:
            rows=[r for r in summaries if r['step']==step and r['phase']==phase];total=sum(r['dense_bytes'] for r in rows)
            aggregate.append({'step':step,'phase':phase,'matrix_count':len(rows),'dense_bytes':total,
                'positive_zero_fraction':sum(r['zero_fraction']*r['dense_bytes'] for r in rows)/total,
                'maximum_savings_under_raw_policy':sum(r['maximum_savings_under_raw_policy']*r['dense_bytes'] for r in rows)/total,
                'special_cells':sum(r['special_cells'] for r in rows)})
    result={'phases':aggregate,'matrix_observations':len(summaries),'grammar_samples':len(samples),
        'all_sample_roundtrips_exact':all(r['roundtrip_exact'] for r in samples),'seconds':time.monotonic()-started,
        'limitations':['Only next two microbatches at each checkpoint, not all training gradients.',
            'Lower-bound eligibility and selected block samples do not prove full-gradient compression.',
            'No compressed gradient optimizer integration or full live-buffer lifetime instrumentation.']}
    write_json(out/'summary.json',result);write_json(out/'artifacts.json',{p.name:digest(p) for p in out.iterdir() if p.is_file()});print(out)


if __name__=='__main__':main()
