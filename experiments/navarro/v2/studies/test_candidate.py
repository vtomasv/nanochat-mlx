"""Post-pilot validation of the archived SIMD proposal, separate from fixed gate."""
import gc
import numpy as np
import pytest
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from research.navarro.candidate import pack_positive


@pytest.mark.parametrize('strategy',['rank','scan'])
@pytest.mark.parametrize('n',[0,1,31,32,33,127,128,129,255,256,257,1025,4096])
def test_exact_domain_boundaries(strategy,n):
    rng=np.random.default_rng(1931+n)
    p=np.maximum(rng.normal(size=n).astype(np.float32),0)
    if n>1:p[1]=np.nextafter(np.float32(0),np.float32(1))
    if n>2:p[2]=np.finfo(np.float32).max
    for sample in (128,256,512):
        a=pack_positive(mx.array(p),sample,strategy)
        np.testing.assert_array_equal(np.asarray(a.unpack()).view(np.uint32),p.view(np.uint32))


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),float('-inf'),-1.,-float(np.nextafter(np.float32(0),np.float32(1)))])
def test_invalid_values_rejected(bad):
    with pytest.raises(ValueError):pack_positive(mx.array([0.,bad]+[0.]*1024))


def test_zero_raw_noncontiguous_and_ownership():
    p=mx.array(np.array([[0.,-0.],[0.,-0.]],np.float32).T)
    np.testing.assert_array_equal(np.asarray(pack_positive(p).unpack()).view(np.uint32),np.zeros((2,2),np.uint32))
    assert pack_positive(mx.ones((1024,),mx.float32)).mode=='RAW_GPU'
    assert pack_positive(mx.zeros((1024,),mx.float32)).mode=='BITMAP_GPU'
    source=mx.maximum(mx.random.normal((1024,1024)),0);mx.eval(source)
    packed=pack_positive(source);mx.eval(packed.bits,packed.index,packed.values)
    before=mx.get_active_memory();del source;gc.collect();mx.synchronize()
    assert before-mx.get_active_memory()>=4*1024*1024


def test_ten_updates_and_midpoint_resume(tmp_path):
    from nanochat_mlx.gpt import GPT,GPTConfig,loss_fn
    from nanochat_mlx.optim import MultiOptimizer,OptimizerConfig
    from nanochat_mlx.experiments.v2.worker import gradient_fn,step
    from nanochat_mlx.experiments.replay import save_checkpoint,load_checkpoint
    mx.random.seed(477)
    config=GPTConfig(sequence_len=16,vocab_size=67,n_layer=4,n_head=2,n_kv_head=2,n_embd=32,window_pattern='SSSL')
    a=GPT(config);a.init_weights()
    for block in a.blocks:
        block.attn.c_proj.weight=mx.random.normal(block.attn.c_proj.weight.shape)*.03
        block.mlp.c_proj.weight=mx.random.normal(block.mlp.c_proj.weight.shape)*.03
    b=GPT(config);b.update(a.parameters());mx.eval(a.parameters(),b.parameters())
    oc=OptimizerConfig(32,matrix_lr=.0001,embedding_lr=.0001,unembedding_lr=.0001,scalar_lr=.0001)
    oa=MultiOptimizer(a,oc);ob=MultiOptimizer(b,oc)
    c={'config_sha256':'candidate-post-pilot-c4','accumulation':2}
    x=mx.array([[1,1,2,3,4,5,6,7],[2,2,3,4,5,6,7,8]])
    y=mx.array([[1,2,3,4,5,6,7,-1],[2,3,4,5,6,7,8,9]])
    batches=lambda:[(x,y),(x,y)]
    for k in range(10):
        la=step(a,oa,batches(),k,c,nn.value_and_grad(a,loss_fn))
        lb=step(b,ob,batches(),k,c,gradient_fn(b,{'arm':'bitmap_gpu','sample_bits':256},pack_positive))
        assert abs(la-lb)<1e-4
        for (name,v),(other,w) in zip(tree_flatten(a.parameters()),tree_flatten(b.parameters())):
            assert name==other
            np.testing.assert_allclose(np.asarray(v),np.asarray(w),atol=1e-5,rtol=1e-3,err_msg=name)
        if k==4:
            save_checkpoint(tmp_path/'step5',b,ob,5,477,c)
            restored=GPT(config);optimizer=MultiOptimizer(restored,oc)
            state=load_checkpoint(tmp_path/'step5',restored,optimizer,c)
            assert state['cursor']==10
            b,ob=restored,optimizer
