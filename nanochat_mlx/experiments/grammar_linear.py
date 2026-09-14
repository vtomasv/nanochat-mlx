"""Frozen MLX-compatible linear using N3 direct CPU product, 16-token tiles."""
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from .grammar_matrix import encode,decode_exact,matvec,resident_bytes

class GrammarLinear(nn.Module):
    def __init__(self,weight,mode='grammar_direct_cpu',variant='adaptive',rows_per_block=128):
        super().__init__()
        self._handle=encode(weight,variant,rows_per_block)
        self._mode=mode
        # One-time exact reconstruction is outside inference measurement.
        original=np.asarray(weight)
        if decode_exact(self._handle).tobytes()!=original.tobytes(): raise ValueError('Weight conversion not bit-exact')
        self._shape=original.shape
    def __call__(self,x):
        if self._mode=='compressed_decode_dense': return x@decode_exact(self._handle,'mlx').T
        if self._mode!='grammar_direct_cpu': raise ValueError(self._mode)
        values=np.asarray(x).reshape(-1,self._shape[1])
        output=np.empty((values.shape[0],self._shape[0]),np.float32)
        for start in range(0,len(values),16):
            for j in range(start,min(start+16,len(values))): output[j]=matvec(self._handle,values[j])
        return mx.array(output.reshape(*x.shape[:-1],self._shape[0]))
    def accounting(self):return resident_bytes(self._handle)
