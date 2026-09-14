"""N1 §1.3 sampled rank, physically owned CPU payload; MLX transfers included."""
from dataclasses import dataclass
import sys
import time
import numpy as np

@dataclass
class PackedPositive:
    shape: tuple
    dtype: str
    n: int
    sample_bits: int
    bitmap: np.ndarray
    index: np.ndarray
    values: np.ndarray
    mode: str
    raw_bytes: int
    pack_seconds: float
    reason: str | None = None

    @classmethod
    def pack(cls,array,sample_bits=256,force=False):
        tick=time.perf_counter()
        if sample_bits not in (128,256,512): raise ValueError('Unsupported sample period')
        p=np.asarray(array)
        if p.dtype not in (np.float32,np.float64): raise TypeError('Floating payload required')
        if not np.all(np.isfinite(p)) or np.any(p<0): raise ValueError('P must be finite and nonnegative')
        flat=np.ascontiguousarray(p).reshape(-1); positive=flat>0; n=flat.size
        bits=np.packbits(positive,bitorder='little')
        bits=np.pad(bits,(0,(-bits.size)%4)).view('<u4').copy()
        counts=np.zeros((n+sample_bits-1)//sample_bits+1,dtype=np.uint64)
        padded=np.pad(positive,(0,(-n)%sample_bits))
        if n: counts[1:]=np.cumsum(padded.reshape(-1,sample_bits).sum(axis=1),dtype=np.uint64)
        vals=flat[positive].copy()
        obj=cls(p.shape,p.dtype.str,n,sample_bits,bits,counts,vals,'BITMAP',p.nbytes,0.)
        if not force and obj.resident_bytes()>.95*p.nbytes:
            obj.bitmap=np.empty(0,np.uint32); obj.index=np.empty(0,np.uint64)
            obj.values=flat.copy(); obj.values[obj.values==0]=0
            obj.mode='RAW'; obj.reason='compact_exceeds_95_percent_raw'
        obj.pack_seconds=time.perf_counter()-tick
        return obj

    def rank(self,i):
        if not 0<=i<=self.n: raise IndexError(i)
        if self.mode=='RAW': return int(np.count_nonzero(self.values[:i]))
        base=i//self.sample_bits
        count=int(self.index[base]); start=base*(self.sample_bits//32); stop=i//32
        count+=sum(int(w).bit_count() for w in self.bitmap[start:stop])
        if i%32: count+=(int(self.bitmap[stop])&((1<<(i%32))-1)).bit_count()
        return count

    def access(self,i):
        if not 0<=i<self.n: raise IndexError(i)
        if self.mode=='RAW': return self.values[i]
        return self.values[self.rank(i)] if (int(self.bitmap[i//32])>>(i%32))&1 else np.dtype(self.dtype).type(0)

    def unpack(self,device='cpu'):
        if self.mode=='RAW': out=self.values.copy().reshape(self.shape)
        else:
            out=np.zeros(self.n,dtype=self.dtype)
            # Vectorized scan is temporary, never retained as an index per element.
            mask=np.unpackbits(self.bitmap.view(np.uint8),bitorder='little',count=self.n).astype(bool)
            out[mask]=self.values; out=out.reshape(self.shape)
        if device=='mlx':
            import mlx.core as mx
            return mx.array(out)
        return out

    def resident_bytes(self):
        # sys.getsizeof includes owned NumPy capacity and Python object headers.
        return sum(sys.getsizeof(a) for a in (self.bitmap,self.index,self.values))+sys.getsizeof(self)+sys.getsizeof(self.shape)+sys.getsizeof(self.dtype)+sys.getsizeof(self.__dict__)
