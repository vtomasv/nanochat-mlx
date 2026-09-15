import numpy as np
import pytest
from nanochat_mlx.experiments.grammar_matrix import encode,decode_exact,matvec,serialize,deserialize,resident_bytes,header

@pytest.mark.parametrize('variant',['RAW','re_32','re_iv','adaptive'])
@pytest.mark.parametrize('shape',[(0,5),(3,0),(1,1),(7,17),(129,33)])
def test_codec_and_products(variant,shape,tmp_path):
    rng=np.random.default_rng(9); a=rng.integers(-2,3,shape).astype(np.float32)
    h=encode(a,variant,32)
    np.testing.assert_array_equal(decode_exact(h).view(np.uint32),a.view(np.uint32))
    x=rng.normal(size=shape[1]).astype(np.float32); y=rng.normal(size=shape[0]).astype(np.float32)
    np.testing.assert_allclose(matvec(h,x),a.astype(np.float64)@x,atol=1e-5,rtol=1e-4)
    np.testing.assert_allclose(matvec(h,y,True),a.astype(np.float64).T@y,atol=1e-5,rtol=1e-4)
    np.testing.assert_allclose(np.dot(matvec(h,x),y),np.dot(x,matvec(h,y,True)),atol=1e-4,rtol=1e-4)
    serialize(h,tmp_path/'matrix.ng'); h2=deserialize(tmp_path/'matrix.ng')
    assert decode_exact(h2).tobytes()==a.tobytes()

def test_columns_shared_rules_and_specials():
    a=np.tile(np.array([[2,3,0],[2,3,5],[3,2,0]],np.float32),(50,1))
    h=encode(a,'re_iv');assert resident_bytes(h)['rules']>0
    np.testing.assert_allclose(matvec(h,np.array([1,2,3],np.float32)),a@np.array([1,2,3],np.float32))
    bits=np.array([[0,0x80000000,0x7fc12345,0x7f800000,0xff800000]],np.uint32)
    for mode in ('adaptive','re_iv','re_32'):
        h=encode(bits.view(np.float32),mode);assert header(h.blocks[0])['raw']==1
        assert decode_exact(h).tobytes()==bits.tobytes()

def test_high_entropy_and_compression():
    a=np.random.default_rng(13).normal(size=(128,128)).astype(np.float32)
    h=encode(a);assert resident_bytes(h)['raw_blocks']==1
    repeated=np.tile(np.arange(128,dtype=np.float32),(128,1))
    h=encode(repeated);assert resident_bytes(h)['resident_bytes']<repeated.nbytes*.95
    assert decode_exact(h).tobytes()==repeated.tobytes()

def test_corrupt_grammar_is_rejected(tmp_path):
    import struct
    from nanochat_mlx.experiments.grammar_matrix import GrammarMatrix
    a=np.tile(np.array([[2,3,0]],np.float32),(16,1))
    h=encode(a,'re_32');b=bytearray(h.blocks[0]);meta=header(b)
    assert meta['rules']>0
    # Make the first production reference itself: no unsafe DAG evaluation allowed.
    struct.pack_into('<I',b,64+meta['values']*4,meta['base'])
    bad=GrammarMatrix(h.shape,h.dtype,(bytes(b),),h.source_sha256)
    with pytest.raises(ValueError):decode_exact(bad)
    serialize(h,tmp_path/'matrix.ng')
    raw=(tmp_path/'matrix.ng').read_bytes();(tmp_path/'short.ng').write_bytes(raw[:-1])
    with pytest.raises(ValueError):deserialize(tmp_path/'short.ng')


def scan_reference(matrix):
    """Small independent oracle for v1's exact pair order and row delimiters."""
    from collections import Counter
    values={};seq=[];cols=matrix.shape[1]
    for row in matrix.view(np.uint32):
        for col,bits in enumerate(row):
            if bits:
                value_id=values.setdefault(int(bits),len(values))
                seq.append(1+value_id*cols+col)
        seq.append(0)
    base=1+len(values)*cols;rules=[]
    while True:
        counts=Counter((a,b) for a,b in zip(seq,seq[1:]) if a and b)
        if not counts or max(counts.values())<2:break
        pair=min(counts,key=lambda p:(-counts[p],p))
        rule_id=base+len(rules);rules.append(pair)
        out=[];i=0
        while i<len(seq):
            if tuple(seq[i:i+2])==pair:out.append(rule_id);i+=2
            else:out.append(seq[i]);i+=1
        seq=out
    return rules,seq


@pytest.mark.parametrize('seed',range(12))
def test_incremental_matches_scan_grammar(seed):
    # Sparse rows, ties, unequal occurrence spacing and nested substitutions.
    rng=np.random.default_rng(seed)
    rows=rng.integers(-2,3,(7,19)).astype(np.float32)
    a=rows[rng.integers(0,7,43)]
    h=encode(a,'re_32',128);meta=header(h.blocks[0])
    rules,seq=scan_reference(a)
    ro=64+meta['values']*4
    co=ro+((len(rules)*2*32+63)//64)*8
    actual_rules=np.frombuffer(h.blocks[0],'<u4',len(rules)*2,ro).reshape(-1,2)
    actual_seq=np.frombuffer(h.blocks[0],'<u4',len(seq),co)
    assert meta['rules']==len(rules) and meta['symbols']==len(seq)
    assert actual_rules.tolist()==[list(p) for p in rules]
    assert actual_seq.tolist()==seq
    assert decode_exact(h).tobytes()==a.tobytes()


@pytest.mark.parametrize('kind',['unique','repeated','zeros','sparse','mixed',
                                 'distinct_0.60','distinct_0.67','distinct_0.70','distinct_0.90'])
def test_adaptive_bound_preserves_final_format_policy(kind):
    rng=np.random.default_rng(81)
    a=rng.normal(size=(129,64)).astype(np.float32)
    if kind=='repeated':a[:]=a[0]
    elif kind=='zeros':a[:]=0
    elif kind=='sparse':a[rng.random(a.shape)<.98]=0
    elif kind=='mixed':a[:128]=a[0]
    elif kind.startswith('distinct_'):
        count=int(a.size*float(kind.split('_')[1]))
        values=np.arange(1,count+1,dtype=np.float32)
        a=np.concatenate((values,rng.choice(values,a.size-count)))
        rng.shuffle(a);a=a.reshape(129,64)
    adaptive=encode(a,'adaptive')
    fixed=encode(a,'re_32');packed=encode(a,'re_iv')
    raw=encode(a,'RAW')
    for i,(f,p,r) in enumerate(zip(fixed.blocks,packed.blocks,raw.blocks)):
        budget=.95*header(r)['rows']*a.shape[1]*4
        expected=r if min(len(f),len(p))>budget else f if len(f)<=len(p) else p
        assert adaptive.blocks[i]==expected
    assert decode_exact(adaptive).tobytes()==a.tobytes()
