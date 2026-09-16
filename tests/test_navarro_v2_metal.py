"""Real Metal correctness gates for the v2 candidate codec and complete GPT VJP."""
import gc
import numpy as np
import pytest

mx=pytest.importorskip('mlx.core')
import mlx.nn as nn
from mlx.utils import tree_flatten
from nanochat_mlx.experiments.v2.bitmap_metal import PackedPositiveGPU
from nanochat_mlx.experiments.activation_tape import loss_and_grad
from nanochat_mlx.gpt import GPT,GPTConfig,loss_fn


@pytest.mark.parametrize('strategy',['rank','scan'])
@pytest.mark.parametrize('n',[0,1,31,32,33,127,128,129,255,256,257,1025])
def test_bitmap_roundtrip(strategy,n):
    rng=np.random.default_rng(719)
    p=np.maximum(rng.normal(size=n).astype(np.float32),0)
    if n>1:p[1]=np.nextafter(np.float32(0),np.float32(1))
    for sample in (128,256,512):
        handle=PackedPositiveGPU.pack(mx.array(p),sample,strategy,force=True)
        np.testing.assert_array_equal(np.asarray(handle.unpack()).view(np.uint32),p.view(np.uint32))
        assert handle.values.size==np.count_nonzero(p)
        assert handle.index.size==0 if strategy=='scan' else True


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1.])
def test_reject_invalid(value):
    with pytest.raises(ValueError):PackedPositiveGPU.pack(mx.array([0.,value]))


def test_raw_and_zeros():
    for p in (np.ones(1024,np.float32),np.zeros(1024,np.float32)):
        h=PackedPositiveGPU.pack(mx.array(p))
        assert h.mode==('RAW_GPU' if p[0] else 'BITMAP_GPU')
        np.testing.assert_array_equal(np.asarray(h.unpack()),p)


@pytest.mark.parametrize('strategy',['rank','scan'])
def test_all_parameter_vjp(strategy):
    mx.random.seed(713)
    model=GPT(GPTConfig(sequence_len=16,vocab_size=67,n_layer=4,n_head=2,n_kv_head=2,n_embd=32,window_pattern='SSSL'))
    model.init_weights()
    for block in model.blocks:
        block.mlp.c_proj.weight=mx.random.normal(block.mlp.c_proj.weight.shape)*.03
        block.attn.c_proj.weight=mx.random.normal(block.attn.c_proj.weight.shape)*.03
    x=mx.array([[1,1,2,3,4,5,6,7],[2,2,3,4,5,6,7,8]])
    y=mx.array([[1,2,3,4,5,6,7,-1],[2,3,4,5,6,7,8,9]])
    ref,g=nn.value_and_grad(model,loss_fn)(model,x,y);mx.eval(ref,g)
    val,h=loss_and_grad(model,x,y,packer=lambda p,s:PackedPositiveGPU.pack(p,s,strategy))
    mx.eval(val,h);assert abs(val.item()-ref.item())<1e-4
    a=dict(tree_flatten(g));b=dict(tree_flatten(h));assert a.keys()==b.keys()
    for key in a:np.testing.assert_allclose(np.asarray(b[key]),np.asarray(a[key]),atol=1e-5,rtol=1e-3,err_msg=key)


def test_releases_dense_buffer():
    mx.synchronize();gc.collect()
    p=mx.maximum(mx.random.normal((1024,1024)),0);mx.eval(p)
    h=PackedPositiveGPU.pack(p);mx.eval(h.bits,h.index,h.values)
    before=mx.get_active_memory();del p;gc.collect();mx.synchronize()
    assert before-mx.get_active_memory()>=4*1024*1024
    assert h.resident_bytes()<.6*4*1024*1024
