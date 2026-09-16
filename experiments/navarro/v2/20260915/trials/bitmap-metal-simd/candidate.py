"""Adaptive proposal 01: coalesced SIMD mask construction.

Same exact representation and decode as the registered GPU codec. Only the mask
construction changes: 32 adjacent lanes read adjacent values and reduce bits.
Apple M3 Max uses 32-lane SIMD groups; reject another width explicitly.
"""
from functools import lru_cache
import time
import mlx.core as mx
from nanochat_mlx.experiments.v2.bitmap_metal import PackedPositiveGPU,kernels


@lru_cache(None)
def mask_kernel():
    return mx.fast.metal_kernel(name='navarro_candidate_simd_mask',
        input_names=['p','length'],output_names=['bits','counts','invalid'],source='''
        uint i=thread_position_in_grid.x, lane=thread_index_in_simdgroup;
        uint v=i<length[0]?p[i]:0u;
        uint positive=(v && v!=0x80000000u)?1u:0u;
        uint bad=((v&0x7f800000u)==0x7f800000u || ((v>>31) && v!=0x80000000u))?1u:0u;
        uint b=metal::simd_sum(positive<<lane);
        uint c=metal::simd_sum(positive), invalid_count=metal::simd_sum(bad);
        if(lane==0) {
            bits[i/32]=b; counts[i/32]=c;
            invalid[i/32]=invalid_count+(threads_per_simdgroup!=32);
        }
        ''')


def pack_positive(p,sample_bits=256,strategy='rank'):
    if p.dtype!=mx.float32:raise TypeError('FP32 required')
    if sample_bits not in (128,256,512) or strategy not in ('rank','scan'):raise ValueError('Invalid codec strategy')
    n=p.size
    if not n:return PackedPositiveGPU.pack(p,sample_bits,strategy)
    if n>=2**32-32:raise ValueError('32-bit length overflow')
    tick=time.perf_counter();words=(n+31)//32;length=mx.array([n],mx.uint32);pb=p.view(mx.uint32)
    bits,counts,invalid=mask_kernel()(inputs=[pb,length],grid=(words*32,1,1),threadgroup=(256,1,1),
        output_shapes=[(words,)]*3,output_dtypes=[mx.uint32]*3)
    inclusive=mx.cumsum(counts);bad=mx.any(invalid);total=inclusive[-1];mx.eval(bad,total)
    if bad.item():raise ValueError('Invalid positive payload or unsupported SIMD width')
    k=int(total.item());index_count=(words+sample_bits//32-1)//(sample_bits//32) if strategy=='rank' else 0
    if 4*(words+index_count+k)+64>.95*p.nbytes:
        raw=mx.where(pb==0x80000000,mx.array(0,mx.uint32),pb).view(mx.float32);mx.eval(raw)
        z=mx.zeros((0,),mx.uint32)
        return PackedPositiveGPU(p.shape,n,z,z,raw,sample_bits,strategy,'RAW_GPU',time.perf_counter()-tick)
    prefix=inclusive-counts
    index=prefix[mx.arange(0,words,sample_bits//32)] if strategy=='rank' else mx.zeros((0,),mx.uint32)
    scatter=kernels()[1]
    values=scatter(inputs=[pb,bits,prefix,length],grid=(n,1,1),threadgroup=(256,1,1),
        output_shapes=[(k,)],output_dtypes=[mx.uint32])[0] if k else mx.zeros((0,),mx.uint32)
    mx.eval(bits,index,values)
    return PackedPositiveGPU(p.shape,n,bits,index,values,sample_bits,strategy,'BITMAP_GPU',time.perf_counter()-tick)
