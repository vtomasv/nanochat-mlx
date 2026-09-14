"""CPU state residency with unchanged inherited Muon/AdamW equations."""
import json
import struct
import time
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from nanochat_mlx.optim import MultiOptimizer
from .grammar_matrix import GrammarMatrix,encode,decode_exact,resident_bytes

class StagedOptimizer(MultiOptimizer):
    def __init__(self,model,config,compress=True,rows_per_block=128):
        super().__init__(model,config)
        self.compress=compress;self.rows_per_block=rows_per_block;self.handles={};self.last_accounting=[]
    def _encode(self,path,array):
        a=np.asarray(array)
        if a.ndim==2:
            variant='adaptive' if self.compress and a.size>=65536 else 'RAW'
            h=encode(a,variant,self.rows_per_block)
            self.last_accounting.append(dict(path=path,**resident_bytes(h)))
            return h
        tick=time.perf_counter();h=np.array(a,copy=True)
        self.last_accounting.append(dict(path=path,raw_bytes=a.nbytes,resident_bytes=h.nbytes,mode='RAW_VECTOR',encode_seconds=time.perf_counter()-tick))
        return h
    def _decode(self,h): return mx.array(decode_exact(h) if isinstance(h,GrammarMatrix) else h)
    def stage_existing(self):
        self.handles={}
        for path in list(self.muon_state):
            self.handles[path]={'buf':self._encode(path+'.buf',self.muon_state.pop(path))}
        for path in list(self.adam_state):
            s=self.adam_state.pop(path)
            self.handles[path]={'m':self._encode(path+'.m',s['m']),'v':self._encode(path+'.v',s['v']),'t':s['t']}
    def update(self,model,grads):
        mx.eval(grads)
        flat_params=dict(tree_flatten(model.parameters()));updates=[];self.last_accounting=[]
        for path,g in tree_flatten(grads):
            if path not in self.param_config: continue
            cfg=self.param_config[path];saved=self.handles.pop(path,None)
            if cfg['kind']=='muon':
                if saved is not None:self.muon_state[path]=self._decode(saved['buf'])
                p=self._muon_step(path,g,flat_params[path],cfg)
                mx.eval(p,self.muon_state[path]);del saved
                self.handles[path]={'buf':self._encode(path+'.buf',self.muon_state.pop(path))}
            else:
                if saved is not None:self.adam_state[path]={'m':self._decode(saved['m']),'v':self._decode(saved['v']),'t':saved['t']}
                p=self._adamw_step(path,g,flat_params[path],cfg)
                s=self.adam_state.pop(path);mx.eval(p,s['m'],s['v']);del saved
                self.handles[path]={'m':self._encode(path+'.m',s['m']),'v':self._encode(path+'.v',s['v']),'t':s['t']};del s
            updates.append((path,p))
        for path,p in updates:
            parts=path.split('.');obj=model
            for key in parts[:-1]:
                obj=obj[int(key)] if isinstance(obj,list) else obj[key] if isinstance(obj,dict) else getattr(obj,key)
            if isinstance(obj,dict): obj[parts[-1]]=p
            else: setattr(obj,parts[-1],p)
    @property
    def state(self):
        # All CPU handles are already materialized. Never decode for mx.eval.
        return []
    def export_dense(self,destination):
        """Stream a standard safetensors file, holding at most one dense state."""
        entries=[];offset=0;header={}
        for path,s in self.handles.items():
            for key,h in s.items():
                name=f'muon.{path}' if key=='buf' else f'adam.{path}.{key}'
                shape=[] if key=='t' else list(h.shape)
                size=4*int(np.prod(shape))
                header[name]={'dtype':'I32' if key=='t' else 'F32','shape':shape,'data_offsets':[offset,offset+size]}
                entries.append((key,h));offset+=size
        encoded=json.dumps(header,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%8)
        with open(destination,'xb') as f:
            f.write(struct.pack('<Q',len(encoded)));f.write(encoded)
            for key,h in entries:
                a=np.array(h,np.int32) if key=='t' else decode_exact(h) if isinstance(h,GrammarMatrix) else h
                f.write(a.tobytes());del a

    def save_handles(self,destination):
        """Versioned experimental checkpoint, preserving chosen block formats."""
        from pathlib import Path
        from .grammar_matrix import serialize
        from .config import dump
        destination=Path(destination);destination.mkdir(exist_ok=False)
        metadata={'version':1,'compress':self.compress,'rows_per_block':self.rows_per_block,'states':{}}
        index=0
        for path,state in self.handles.items():
            record={}
            for key,h in state.items():
                if key=='t':record[key]={'integer':h};continue
                if isinstance(h,GrammarMatrix):
                    name=f'{index:04d}.ng';serialize(h,destination/name);kind='grammar'
                else:
                    name=f'{index:04d}.npy';np.save(destination/name,h);kind='raw_vector'
                record[key]={'file':name,'kind':kind};index+=1
            metadata['states'][path]=record
        dump(destination/'handles.json',metadata)

    def restore_handles(self,source):
        from pathlib import Path
        from .grammar_matrix import deserialize
        source=Path(source);meta=json.loads((source/'handles.json').read_text())
        if meta['version']!=1 or meta['compress']!=self.compress or meta['rows_per_block']!=self.rows_per_block:raise ValueError('State-store format/configuration mismatch')
        handles={}
        for path,state in meta['states'].items():
            handles[path]={}
            for key,record in state.items():
                if 'integer' in record:h=record['integer']
                elif record['kind']=='grammar':h=deserialize(source/record['file'])
                else:h=np.load(source/record['file'],allow_pickle=False)
                handles[path][key]=h
        self.handles=handles;self.adam_state={};self.muon_state={}
