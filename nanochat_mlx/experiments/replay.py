"""Exact batch tapes and versioned checkpoints, separate from user checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
from .config import dump, sha256

def seed_all(seed):
    import mlx.core as mx
    random.seed(seed); np.random.seed(seed); mx.random.seed(seed)

def verify_files(files):
    for path, digest in files.items():
        if sha256(path) != digest: raise ValueError(f'Hash mismatch: {path}')

def prepare(c):
    import mlx.core as mx
    import pyarrow.parquet as pq
    os.environ['NANOCHAT_BASE_DIR'] = c['data_dir']
    from nanochat_mlx.tokenizer import get_tokenizer
    from nanochat_mlx.dataloader import dataloader_bos_bestfit
    from nanochat_mlx.dataset import get_split_parquet_files
    root = Path(c['artifact_dir']); root.mkdir(parents=True,exist_ok=True)
    meta = root/'tape.json'
    if meta.exists():
        data = json.loads(meta.read_text()); verify_files(data['files']); return data
    tok = get_tokenizer()
    source = {}
    docsets = {}
    counts = {}
    for split in ('train','val'):
        seen = set(); count = 0
        for path in get_split_parquet_files(split):
            source[str(Path(path).resolve())] = sha256(path)
            for batch in pq.ParquetFile(path).iter_batches(columns=['text'],batch_size=1024):
                for text in batch.column(0).to_pylist():
                    seen.add(hashlib.sha256(text.encode()).digest()); count += 1
        docsets[split] = seen; counts[split] = count
    overlap = len(docsets['train'] & docsets['val'])
    for path in (Path(c['data_dir'])/'tokenizer').iterdir():
        if path.is_file(): source[str(path)] = sha256(path)
    files = dict(source)
    states = []
    # 1000 scheduled updates plus 100 engineering measurement batches.
    n = 1100*c['accumulation']; B=c['device_batch_size']; T=c['sequence_len']
    loader = dataloader_bos_bestfit(tok,B,T,'train')
    x = np.lib.format.open_memmap(root/'train_x.npy',mode='w+',dtype=np.int32,shape=(n,B,T))
    y = np.lib.format.open_memmap(root/'train_y.npy',mode='w+',dtype=np.int32,shape=(n,B,T))
    for i in range(n):
        a,b,state = next(loader); mx.eval(a,b); x[i]=np.asarray(a); y[i]=np.asarray(b); states.append(state)
        if i%100==0: print(f'tape {i}/{n}',flush=True)
    x.flush(); y.flush(); del x,y,loader
    loader = dataloader_bos_bestfit(tok,1,T,'val')
    vx,vy=[],[]
    for i in range(128):
        a,b,_=next(loader); vx.append(np.asarray(a)[0]); vy.append(np.asarray(b)[0])
    np.save(root/'val_x.npy',np.array(vx)); np.save(root/'val_y.npy',np.array(vy))
    for path in root.glob('*.npy'): files[str(path)] = sha256(path)
    dump(root/'tape_states.json',states)
    files[str(root/'tape_states.json')] = sha256(root/'tape_states.json')
    data={'files':files,'vocab_size':tok.get_vocab_size(),'source_revision':None,
          'source':'existing local FineWeb-Edu shards; upstream revision unavailable',
          'document_counts':counts,'exact_cross_split_duplicate_documents':overlap,
          'max_epoch':max(s['epoch'] for s in states),'processed_training_tokens':1000*c['total_batch_size'],
          'unique_token_count':None,'masks':'targets != -1; no masked tokens in BOS tape',
          'calibration_sequences':[0,32],'reserved_sequences':[32,128],
          'document_group_limitation':'Best-fit packing loses document provenance; document-grouped E3 CI requires a provenance extension',
          'config_sha256':c['config_sha256']}
    dump(meta,data)
    return data

def save_checkpoint(root, model, optimizer, step, seed, c):
    import mlx.core as mx
    from nanochat_mlx.train import _save_optimizer_state
    root=Path(root); root.mkdir(parents=True,exist_ok=False)
    model.save_weights(str(root/'model.safetensors'))
    if hasattr(optimizer,'save_handles'):
        optimizer.save_handles(root/'handles')
    else: _save_optimizer_state(optimizer,str(root/'optimizer.safetensors'))
    # Training has no stochastic layer; still preserve every RNG for future use.
    rng=np.random.get_state()
    dump(root/'state.json',{'version':1,'step':step,'seed':seed,'cursor':step*c['accumulation'],
        'schedule_iterations':1000,'config_sha256':c['config_sha256'],
        'python_rng':random.getstate(),'numpy_rng':[rng[0],rng[1].tolist(),rng[2],rng[3],rng[4]],
        'mlx_rng':np.asarray(mx.random.state[0]).tolist(),
        'param_config':optimizer.param_config,'initial_lrs':optimizer.initial_lrs})
    dump(root/'hashes.json',{str(p.resolve()):sha256(p) for p in root.rglob('*') if p.is_file()})

def load_checkpoint(root,model,optimizer,c):
    import mlx.core as mx
    from nanochat_mlx.train import _load_weights_into_model, _load_optimizer_state
    root=Path(root); verify_files(json.loads((root/'hashes.json').read_text()))
    s=json.loads((root/'state.json').read_text())
    if s['config_sha256'] != c['config_sha256'] or s['schedule_iterations']!=1000: raise ValueError('Checkpoint protocol mismatch')
    _load_weights_into_model(model,str(root/'model.safetensors'))
    stored_handles=(root/'handles').exists()
    if stored_handles:
        if not hasattr(optimizer,'restore_handles'):raise ValueError('Experimental state requires StagedOptimizer or explicit dense export')
        optimizer.restore_handles(root/'handles')
    else:_load_optimizer_state(optimizer,str(root/'optimizer.safetensors'))
    optimizer.param_config=s['param_config']; optimizer.initial_lrs=s['initial_lrs']
    def tuples(v): return tuple(tuples(x) for x in v) if isinstance(v,list) else v
    random.setstate(tuples(s['python_rng']))
    n=s['numpy_rng']; np.random.set_state((n[0],np.array(n[1],np.uint32),n[2],n[3],n[4]))
    mx.random.state=[mx.array(s['mlx_rng'],dtype=mx.uint32)]
    mx.eval(model.parameters(),optimizer.state)
    if hasattr(optimizer,'stage_existing') and not stored_handles: optimizer.stage_existing()
    return s


def checkpoint_path(c,run_id,step,experiment='baseline'):
    root=Path(c['artifact_dir'])/'checkpoints'
    current=root/experiment/run_id/f'step-{step:04d}'
    legacy=root/run_id/f'step-{step:04d}'
    if current.exists():return current
    if legacy.exists():return legacy
    raise FileNotFoundError(f'Missing checkpoint {run_id}, completed updates={step}')


def load_inference_checkpoint(root,model,c):
    """Validate the checkpoint but load only weights for inference."""
    import mlx.core as mx
    from nanochat_mlx.train import _load_weights_into_model
    root=Path(root);verify_files(json.loads((root/'hashes.json').read_text()))
    state=json.loads((root/'state.json').read_text())
    if state['config_sha256']!=c['config_sha256']:raise ValueError('Inference checkpoint profile mismatch')
    _load_weights_into_model(model,str(root/'model.safetensors'));mx.eval(model.parameters())
    return state
