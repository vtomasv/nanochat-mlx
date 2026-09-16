"""Supplementary exact-value/RePair singleton bounds, both orientations.

A terminal (value bits,column) appearing only once cannot be absorbed by a
RePair rule (pair frequency >=2); its code must remain in the top-level stream.
The bound omits all other codes, rules, row separators and most metadata.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import time

import numpy as np
from nanochat_mlx.experiments.v2.records import digest,write_json,now
from scripts.navarro_structural_census import census


def tensors(path):
    with path.open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0];meta=json.loads(f.read(size))
    for key,value in meta.items():
        if key=='__metadata__' or value['dtype']!='F32' or len(value['shape'])!=2:continue
        start,end=value['data_offsets'];shape=tuple(value['shape'])
        if end-start!=math.prod(shape)*4:raise ValueError('Invalid safetensors extent')
        yield key,np.memmap(path,dtype='<u4',mode='r',offset=8+size+start,shape=shape)


def bound(bits):
    flat=np.asarray(bits).reshape(-1);nz=flat[flat!=0]
    values=len(np.unique(nz));singletons=0
    for j in range(bits.shape[1]):
        column=np.asarray(bits[:,j]);column=column[column!=0]
        _,counts=np.unique(column,return_counts=True);singletons+=int(np.count_nonzero(counts==1))
    base=1+values*bits.shape[1];width=max(1,(base-1).bit_length())
    lower=64+values*4+(singletons*width+7)//8
    return {'rows':bits.shape[0],'cols':bits.shape[1],'cells':bits.size,'dense_bytes':bits.nbytes,
        'distinct_nonzero_values':values,'singleton_terminals':singletons,'minimum_symbol_width':width,
        'minimum_dictionary_and_singleton_bytes':lower,'bound_ratio':lower/bits.nbytes,
        'rejected_95pct':lower*100>bits.nbytes*95,'current_uint32_terminal_overflow':base>=2**32-1,
        'positive_zero_cells':int(bits.size-nz.size)}


def main():
    root=Path('experiments/navarro/v2/20260915');m=json.loads((root/'manifest.json').read_text())
    out=root/'dictionary-extension';out.mkdir(exist_ok=False)
    plan={'created_at':now(),'source_sha256':digest(Path(__file__)),
        'global_checkpoints':[100,1000],'global_objects':['model.safetensors','optimizer.safetensors'],
        'orientations':['original','transposed'],'early_local_checkpoints':[0,10],
        'local_rows':128,'threshold_ratio':.95,'campaign_manifest_sha256':digest(root/'manifest.json'),
        'interpretation':'Lower bound for exact per-matrix dictionaries and RePair, not every compressor. Unrejected is undecided.',
        'scope':'Structural diagnostics, no performance inference, no held-out data.'}
    write_json(out/'plan.json',plan);records=[];early=[];tick=time.monotonic()
    artifact=Path(m['artifact_root'])
    for step in plan['global_checkpoints']:
        checkpoint=artifact/'reference'/f'step-{step:04d}';expected=json.loads((checkpoint/'hashes.json').read_text())
        for filename in plan['global_objects']:
            path=checkpoint/filename
            if digest(path)!=expected[str(path.resolve())]:raise ValueError('Checkpoint changed')
            for name,array in tensors(path):
                for orientation,bits in [('original',array),('transposed',array.T)]:
                    r={'step':step,'file':filename,'tensor':name,'orientation':orientation,**bound(bits)};records.append(r)
            print('GLOBAL BOUND',step,filename,flush=True)
    with (out/'global.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    for step in plan['early_local_checkpoints']:
        checkpoint=Path(m['initial_checkpoint']) if step==0 else artifact/'reference'/f'step-{step:04d}'
        expected=json.loads((checkpoint/'hashes.json').read_text())
        for filename in plan['global_objects']:
            path=checkpoint/filename
            if digest(path)!=expected[str(path.resolve())]:raise ValueError('Early checkpoint changed')
            rows,skipped=census(path,128)
            if rows:
                with (out/f'local-step-{step}-{filename}.csv').open('x',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            total=sum(r['dense_bytes'] for r in rows);reject=sum(r['dense_bytes'] for r in rows if r['rejected_by_dictionary_bound'])
            early.append({'step':step,'file':filename,'blocks':len(rows),'dense_bytes':total,
                'rejected_blocks':sum(r['rejected_by_dictionary_bound'] for r in rows),
                'maximum_savings_under_raw_policy':1-reject/total if total else None,'skipped':skipped})
    aggregates=[]
    for step in plan['global_checkpoints']:
        for filename in plan['global_objects']:
            for orientation in plan['orientations']:
                rows=[r for r in records if r['step']==step and r['file']==filename and r['orientation']==orientation]
                total=sum(r['dense_bytes'] for r in rows);reject=sum(r['dense_bytes'] for r in rows if r['rejected_95pct'])
                aggregates.append({'step':step,'file':filename,'orientation':orientation,'matrices':len(rows),
                    'rejected_matrices':sum(r['rejected_95pct'] for r in rows),
                    'uint32_overflow_matrices':sum(r['current_uint32_terminal_overflow'] for r in rows),
                    'maximum_savings_under_raw_policy':1-reject/total})
    summary={'global':aggregates,'early_local':early,'seconds':time.monotonic()-tick,
        'proof_scope':'Singleton bound assumes RePair only replaces repeated pairs and fixed-width integer codes of the declared terminal alphabet.'}
    write_json(out/'summary.json',summary)
    write_json(out/'artifacts.json',{p.name:digest(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps(aggregates,indent=2))


if __name__=='__main__':main()
