"""N3 CSRV/RePair blocks with directly readable packed rule/symbol streams."""
import ctypes as C
from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct
import subprocess
import sys
import time
import numpy as np
from .config import sha256

_LIB=None

def library():
    global _LIB
    if _LIB is None:
        source=Path(__file__).with_name('native.cpp')
        output=source.with_name('native-'+sha256(source)[:16]+'.dylib')
        if not output.exists(): subprocess.run(['clang++','-std=c++17','-O3','-ffp-contract=off','-dynamiclib',str(source),'-o',str(output)],check=True)
        lib=C.CDLL(str(output))
        lib.ng_encode.argtypes=[C.c_void_p,C.c_uint32,C.c_uint32,C.c_int,C.POINTER(C.c_size_t)];lib.ng_encode.restype=C.c_void_p
        lib.ng_data.argtypes=[C.c_void_p];lib.ng_data.restype=C.c_void_p
        lib.ng_free.argtypes=[C.c_void_p]
        lib.ng_decode.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p];lib.ng_decode.restype=C.c_int
        lib.ng_matvec.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_void_p,C.c_int];lib.ng_matvec.restype=C.c_int
        _LIB=lib
    return _LIB

@dataclass(frozen=True)
class GrammarMatrix:
    shape: tuple
    dtype: str
    blocks: tuple
    source_sha256: str
    encode_seconds: float=0.
    def accounting(self): return resident_bytes(self)

def header(block):
    if len(block)<64: raise ValueError('Short grammar block')
    h=struct.unpack('<16I',block[:64])
    return dict(zip(('magic','version','rows','cols','values','rules','symbols','base','width','raw','height'),h))

def encode(array,variant='adaptive',rows_per_block=128):
    tick=time.perf_counter();a=np.array(array,copy=True,order='C')
    if a.dtype!=np.dtype('<f4') or a.ndim!=2: raise TypeError('Grammar codec v1 supports little-endian FP32 matrices; other tensors use RAW staging')
    if rows_per_block<=0: raise ValueError('Positive row block size required')
    mode={'RAW':0,'re_32':32,'re_iv':1,'adaptive':-1}[variant];lib=library();blocks=[]
    for start in range(0,a.shape[0],rows_per_block):
        b=a[start:start+rows_per_block];size=C.c_size_t()
        handle=lib.ng_encode(b.ctypes.data,b.shape[0],b.shape[1],mode,C.byref(size))
        if not handle: raise RuntimeError('Compiled RePair construction failed')
        try: blocks.append(C.string_at(lib.ng_data(handle),size.value))
        finally: lib.ng_free(handle)
    result=GrammarMatrix(a.shape,a.dtype.str,tuple(blocks),hashlib.sha256(a.tobytes()).hexdigest(),time.perf_counter()-tick)
    return result

def decode_exact(handle,device='cpu'):
    out=np.empty(handle.shape,dtype=handle.dtype);offset=0;lib=library()
    for block in handle.blocks:
        h=header(block)
        if h['cols']!=handle.shape[1] or offset+h['rows']>handle.shape[0]: raise ValueError('Block shape mismatch')
        dest=out[offset:offset+h['rows']]
        if lib.ng_decode(block,len(block),dest.ctypes.data): raise ValueError('Invalid grammar, cycle, or expansion')
        offset+=h['rows']
    if offset!=handle.shape[0]: raise ValueError('Missing rows')
    if device=='mlx':
        import mlx.core as mx
        return mx.array(out)
    if device!='cpu': raise ValueError(device)
    return out

def resident_bytes(handle):
    hs=[header(b) for b in handle.blocks]
    raw=handle.shape[0]*handle.shape[1]*np.dtype(handle.dtype).itemsize
    payload=sum(len(b) for b in handle.blocks)
    overhead=sys.getsizeof(handle)+sys.getsizeof(handle.__dict__)+sys.getsizeof(handle.blocks)+sys.getsizeof(handle.shape)+sys.getsizeof(handle.dtype)+sys.getsizeof(handle.source_sha256)
    return {'raw_bytes':raw,'resident_bytes':overhead+sum(sys.getsizeof(b) for b in handle.blocks),'payload_bytes':payload,'container_overhead_bytes':overhead,'blocks':len(hs),'raw_blocks':sum(h['raw'] for h in hs),'rules':sum(h['rules'] for h in hs),'symbols':sum(h['symbols'] for h in hs),'values':sum(h['values'] for h in hs),'height':max((h['height'] for h in hs),default=0),'dictionary_bytes':sum(h['values']*4 for h in hs),'header_bytes':len(hs)*64,'encode_seconds':handle.encode_seconds,'mode':'RAW' if all(h['raw'] for h in hs) else 'GRAMMAR'}

def matvec(handle,x,transpose=False):
    x=np.ascontiguousarray(x,dtype=np.float32)
    if x.ndim!=1 or x.size!=handle.shape[0 if transpose else 1]: raise ValueError('Vector dimension mismatch')
    out=np.zeros(handle.shape[1 if transpose else 0],np.float32);offset=0;lib=library()
    for block in handle.blocks:
        h=header(block)
        if transpose:
            v=x[offset:offset+h['rows']];dest=np.empty(handle.shape[1],np.float32)
        else: v=x;dest=out[offset:offset+h['rows']]
        if lib.ng_matvec(block,len(block),v.ctypes.data,dest.ctypes.data,int(transpose)): raise ValueError('Direct product failed')
        if transpose: out+=dest
        offset+=h['rows']
    return out

def serialize(handle,destination):
    import json
    meta=json.dumps({'version':1,'shape':handle.shape,'dtype':handle.dtype,'sha256':handle.source_sha256,'lengths':[len(b) for b in handle.blocks]}).encode()
    with open(destination,'xb') as f:
        f.write(struct.pack('<Q',len(meta)));f.write(meta)
        for b in handle.blocks: f.write(b)

def deserialize(source):
    import json
    with open(source,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        if n>1<<20: raise ValueError('Invalid header length')
        meta=json.loads(f.read(n));blocks=tuple(f.read(k) for k in meta['lengths'])
        if f.read(1): raise ValueError('Trailing bytes')
    if meta['version']!=1 or meta['dtype']!='<f4': raise ValueError('Unknown format')
    h=GrammarMatrix(tuple(meta['shape']),meta['dtype'],blocks,meta['sha256'])
    out=decode_exact(h)
    if hashlib.sha256(out.tobytes()).hexdigest()!=h.source_sha256: raise ValueError('Integrity mismatch')
    return h
