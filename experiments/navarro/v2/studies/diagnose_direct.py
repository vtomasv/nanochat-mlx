"""FP64 dot-product oracle for failed direct-CSR forward comparisons.

The existing FP32 gate is kept unchanged. This diagnoses both GPU results
against an independent wider accumulator, and never promotes a failed operator.
"""
import json
from pathlib import Path
import time

import numpy as np
import mlx.core as mx
from nanochat_mlx.experiments.v2.records import digest,write_json,now
from experiments.navarro.v2.studies.direct_activation import pack,product


def main():
    root=Path('experiments/navarro/v2/20260915');m=json.loads((root/'manifest.json').read_text());out=root/'direct-diagnosis';out.mkdir(exist_ok=False)
    plan={'created_at':now(),'source_sha256':digest(Path(__file__)),
        'operator_source_sha256':digest(Path('experiments/navarro/v2/studies/direct_activation.py')),
        'input_study_sha256':digest(root/'mechanisms/artifacts.json'),'documents':4,'positions_cap':128,
        'selection':'First 64 mismatching output coordinates per case plus coordinate of largest absolute difference.',
        'oracle':'NumPy float64 dot of original FP32 operands; coefficients/products representable exactly, accumulation widened.',
        'tolerances_unchanged':{'atol':1e-5,'rtol':1e-4},'scope':'Diagnosis only; failed original gate remains failed.'}
    write_json(out/'plan.json',plan)
    from nanochat_mlx.experiments.v2.worker import protect_reserved
    protect_reserved(Path(m['data_root']))
    from nanochat_mlx.experiments.runner import build
    from nanochat_mlx.experiments.replay import load_checkpoint
    from nanochat_mlx.experiments.measure import setup_device,environment
    from nanochat_mlx.gpt import norm
    setup_device();write_json(out/'environment.json',environment())
    model,opt=build(m['config'],m['config']['seed']);load_checkpoint(Path(m['artifact_root'])/'reference/step-1000',model,opt,m['config']);del opt
    ids_array=np.load(Path(m['data_root'])/'calibration_x.npy',mmap_mode='r');target=np.load(Path(m['data_root'])/'calibration_y.npy',mmap_mode='r')
    rows=[];cases=[];tick=time.monotonic()
    for document in range(plan['documents']):
        ids=mx.array(ids_array[document:document+1]);masks=model._get_masks(ids.shape[1]);x=norm(model.wte(ids));x0=x
        valid=min(128,int(np.count_nonzero(target[document]!=-1)))
        for i,block in enumerate(model.blocks):
            x=model.resid_lambdas[i]*x+model.x0_lambdas[i]*x0
            ve=model.value_embeds[str(i)](ids) if str(i) in model.value_embeds else None
            a=x+block.attn(norm(x),ve,mask=masks[i]);p=mx.maximum(block.mlp.c_fc(norm(a)),0);h=p*p;x=a+block.mlp.c_proj(h);mx.eval(x,h)
            matrix=mx.contiguous(h[0,:valid]);w=block.mlp.c_proj.weight
            sparse=product(pack(matrix),w);dense=matrix@w.T;mx.eval(sparse,dense)
            aa=np.asarray(sparse);bb=np.asarray(dense);diff=np.abs(aa.astype(np.float64)-bb)
            mismatch=diff>1e-5+1e-4*np.abs(bb.astype(np.float64));points=[tuple(map(int,p)) for p in np.argwhere(mismatch)[:64]]
            largest=tuple(map(int,np.unravel_index(np.argmax(diff),diff.shape)))
            if largest not in points:points.append(largest)
            hh=np.asarray(matrix);ww=np.asarray(w);case=[]
            for r,c in points:
                oracle=float(np.dot(hh[r].astype(np.float64),ww[c].astype(np.float64)))
                av=float(aa[r,c]);bv=float(bb[r,c]);tol=1e-5+1e-4*abs(oracle)
                record={'document':document,'layer':i,'row':r,'column':c,'oracle_fp64':oracle,
                    'dense_fp32':bv,'csr_fp32':av,'dense_abs_error':abs(bv-oracle),'csr_abs_error':abs(av-oracle),
                    'dense_pass_vs_oracle':abs(bv-oracle)<=tol,'csr_pass_vs_oracle':abs(av-oracle)<=tol,
                    'failed_original_comparison':bool(mismatch[r,c]),'sum_abs_products':float(np.sum(np.abs(hh[r].astype(np.float64)*ww[c].astype(np.float64))))}
                rows.append(record);case.append(record)
            cases.append({'document':document,'layer':i,'mismatching_outputs':int(mismatch.sum()),'total_outputs':int(mismatch.size),'oracle_coordinates':len(case)})
    write_json(out/'coordinates.json',rows);write_json(out/'cases.json',cases)
    failed=[r for r in rows if r['failed_original_comparison']]
    result={'cases':len(cases),'cases_with_mismatch':sum(c['mismatching_outputs']>0 for c in cases),
        'mismatching_outputs':sum(c['mismatching_outputs'] for c in cases),'total_outputs':sum(c['total_outputs'] for c in cases),
        'diagnosed_coordinates':len(rows),'diagnosed_mismatch_coordinates':len(failed),
        'on_diagnosed_mismatches':{'dense_failed_oracle':sum(not r['dense_pass_vs_oracle'] for r in failed),
            'csr_failed_oracle':sum(not r['csr_pass_vs_oracle'] for r in failed),
            'both_failed_oracle':sum(not r['dense_pass_vs_oracle'] and not r['csr_pass_vs_oracle'] for r in failed),
            'csr_closer_to_oracle':sum(r['csr_abs_error']<r['dense_abs_error'] for r in failed)},
        'decision':'BLOCKED_BY_UNCHANGED_ELEMENTWISE_GATE','seconds':time.monotonic()-tick,
        'interpretation':'Quantifies reduction-order sensitivity and cancellation; cannot prove a race absent. No relaxed tolerances, no GPT integration.'}
    write_json(out/'summary.json',result);write_json(out/'artifacts.json',{p.name:digest(p) for p in out.iterdir() if p.is_file()});print(json.dumps(result,indent=2))


if __name__=='__main__':main()
