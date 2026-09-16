"""Exact positive FP32 payload, physically compact Metal buffers and sampled rank.

Word prefixes are construction temporaries. The scan ablation does not retain
rank metadata; it reconstructs word prefixes when decoding. No CPU payload copy.
"""
from dataclasses import dataclass
from functools import lru_cache
import time

import mlx.core as mx


@lru_cache(None)
def kernels():
    mask = mx.fast.metal_kernel(
        name='navarro_v2_mask', input_names=['p', 'length'],
        output_names=['bits', 'counts', 'invalid'], source='''
        uint w = thread_position_in_grid.x;
        uint n = length[0];
        if (w >= (n+31)/32) return;
        uint b = 0, bad = 0;
        for (uint j=0;j<32 && w*32+j<n;j++) {
            uint v = p[w*32+j];
            bad |= ((v & 0x7f800000u) == 0x7f800000u || ((v >> 31) && v != 0x80000000u));
            if (v && v != 0x80000000u) b |= (1u << j);
        }
        bits[w]=b; counts[w]=metal::popcount(b); invalid[w]=bad;
        ''')
    scatter = mx.fast.metal_kernel(
        name='navarro_v2_scatter', input_names=['p', 'bits', 'prefix', 'length'],
        output_names=['values'], source='''
        uint i=thread_position_in_grid.x;
        if (i>=length[0]) return;
        uint w=i/32, j=i%32, b=bits[w];
        if ((b>>j)&1u) values[prefix[w]+metal::popcount(b&((1u<<j)-1u))]=p[i];
        ''')
    unpack = mx.fast.metal_kernel(
        name='navarro_v2_unpack', input_names=['values', 'bits', 'index', 'length'],
        output_names=['out'], source='''
        uint i=thread_position_in_grid.x;
        if (i>=length[0]) return;
        uint w=i/32, j=i%32, b=bits[w];
        if (!((b>>j)&1u)) { out[i]=0; return; }
        uint group=w/SAMPLE_WORDS, offset=index[group];
        for (uint k=group*SAMPLE_WORDS;k<w;k++) offset+=metal::popcount(bits[k]);
        offset+=metal::popcount(b&((1u<<j)-1u));
        out[i]=values[offset];
        ''')
    counts = mx.fast.metal_kernel(
        name='navarro_v2_popcounts', input_names=['bits', 'length'],
        output_names=['counts'], source='''
        uint i=thread_position_in_grid.x;
        if(i<length[0]) counts[i]=metal::popcount(bits[i]);
        ''')
    return mask, scatter, unpack, counts


@dataclass
class PackedPositiveGPU:
    shape: tuple
    n: int
    bits: object
    index: object
    values: object
    sample_bits: int
    strategy: str
    mode: str
    pack_seconds: float

    @classmethod
    def pack(cls, p, sample_bits=256, strategy='rank', force=False):
        if p.dtype != mx.float32:
            raise TypeError('Positive GPU codec requires FP32')
        if sample_bits not in (128, 256, 512) or strategy not in ('rank', 'scan'):
            raise ValueError('Invalid rank sample or decode strategy')
        n = p.size
        if n >= 2**32 - 32:
            raise ValueError('32-bit bitmap length overflow')
        tick = time.perf_counter()
        words = (n + 31)//32
        if not n:
            z=mx.zeros((0,), mx.uint32)
            return cls(p.shape, n, z, z, mx.zeros((0,),mx.float32), sample_bits, strategy, 'BITMAP_GPU', time.perf_counter()-tick)
        mask, scatter, _, _ = kernels()
        length=mx.array([n],mx.uint32)
        payload_bits=p.view(mx.uint32)
        bits, counts, invalid=mask(inputs=[payload_bits,length], grid=(words,1,1), threadgroup=(128,1,1),
            output_shapes=[(words,)]*3,output_dtypes=[mx.uint32]*3)
        inclusive=mx.cumsum(counts)
        bad=mx.any(invalid); total=inclusive[-1]
        mx.eval(bad,total)
        if bad.item():
            raise ValueError('Positive GPU payload must be finite and nonnegative')
        k=int(total.item())
        index_count=(words+sample_bits//32-1)//(sample_bits//32) if strategy=='rank' else 0
        if not force and 4*(words+index_count+k)+64 > .95*p.nbytes:
            # RAW owns a new canonical-zero buffer, no view of a larger source.
            raw=mx.where(payload_bits==0x80000000,mx.array(0,mx.uint32),payload_bits).view(mx.float32);mx.eval(raw)
            z=mx.zeros((0,),mx.uint32)
            return cls(p.shape,n,z,z,raw,sample_bits,strategy,'RAW_GPU',time.perf_counter()-tick)
        prefix=inclusive-counts
        index=prefix[mx.arange(0,words,sample_bits//32)] if strategy=='rank' else mx.zeros((0,),mx.uint32)
        values=scatter(inputs=[payload_bits,bits,prefix,length],grid=(n,1,1),threadgroup=(256,1,1),
            output_shapes=[(k,)],output_dtypes=[mx.uint32])[0] if k else mx.zeros((0,),mx.uint32)
        mx.eval(bits,index,values)
        return cls(p.shape,n,bits,index,values,sample_bits,strategy,'BITMAP_GPU',time.perf_counter()-tick)

    def resident_bytes(self):
        return self.bits.nbytes+self.index.nbytes+self.values.nbytes+64

    def unpack(self, device='mlx'):
        if device!='mlx':
            raise ValueError('GPU codec decodes to MLX only')
        if self.mode=='RAW_GPU':
            return self.values.reshape(self.shape)
        if not self.n or not self.values.size:
            return mx.zeros(self.shape,mx.float32)
        _,_,decode,count_kernel=kernels()
        index=self.index; words=self.bits.size; sample_words=self.sample_bits//32
        if self.strategy=='scan':
            counts=count_kernel(inputs=[self.bits,mx.array([words],mx.uint32)],
                grid=(words,1,1),threadgroup=(128,1,1),output_shapes=[(words,)],output_dtypes=[mx.uint32])[0]
            index=mx.cumsum(counts)-counts;sample_words=1
        out=decode(inputs=[self.values,self.bits,index,mx.array([self.n],mx.uint32)],
            template=[('SAMPLE_WORDS',sample_words)],grid=(self.n,1,1),threadgroup=(256,1,1),
            output_shapes=[self.shape],output_dtypes=[mx.uint32])[0].view(mx.float32)
        mx.eval(out)
        return out
