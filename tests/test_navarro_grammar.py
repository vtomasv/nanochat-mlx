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
