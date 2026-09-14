"""These tests require real Metal; a skip is not a passing integration result."""
import numpy as np
import pytest
mx=pytest.importorskip('mlx.core')
import mlx.nn as nn
from mlx.utils import tree_flatten
from nanochat_mlx.gpt import GPT,GPTConfig,loss_fn
from nanochat_mlx.experiments.activation_tape import loss_and_grad
from nanochat_mlx.experiments.metrics import errors

@pytest.mark.parametrize('arm',['tape_dense','tape_recompute','tape_bitmap'])
def test_gpt_all_parameter_vjp(arm):
    mx.random.seed(71)
    model=GPT(GPTConfig(sequence_len=16,vocab_size=67,n_layer=4,n_head=2,n_kv_head=2,n_embd=32,window_pattern='SSSL'))
    model.init_weights()
    # Nonzero output projections expose all hidden gradient paths.
    for b in model.blocks:
        b.attn.c_proj.weight=mx.random.normal(b.attn.c_proj.weight.shape)*.04
        b.mlp.c_proj.weight=mx.random.normal(b.mlp.c_proj.weight.shape)*.04
    x=mx.array([[1,1,2,3,4,5,6,7,8],[2,2,3,4,5,6,7,8,9]])
    y=mx.array([[2,3,4,-1,6,7,8,9,10],[3,4,5,6,7,8,9,10,11]])
    ref,g=nn.value_and_grad(model,loss_fn)(model,x,y); mx.eval(ref,g)
    val,h=loss_and_grad(model,x,y,arm); mx.eval(val,h)
    assert abs(val.item()-ref.item())<=1e-4
    gr=dict(tree_flatten(g)); hr=dict(tree_flatten(h)); assert gr.keys()==hr.keys()
    for k in gr:
        err=errors(np.asarray(hr[k]),np.asarray(gr[k]),atol=1e-5,rtol=1e-3)
        assert err['pass'] and err['relative_l2']<=1e-3,(k,err)

@pytest.mark.parametrize('compress',[False,True])
def test_optimizer_twenty_steps(compress,tmp_path):
    from nanochat_mlx.optim import MultiOptimizer,OptimizerConfig
    from nanochat_mlx.experiments.staged_optimizer import StagedOptimizer
    from nanochat_mlx.experiments.grammar_matrix import decode_exact,GrammarMatrix
    from nanochat_mlx.train import _load_optimizer_state
    from mlx.utils import tree_unflatten
    mx.random.seed(23)
    cfg=GPTConfig(sequence_len=8,vocab_size=67,n_layer=2,n_head=2,n_kv_head=2,n_embd=32)
    a=GPT(cfg);a.init_weights();b=GPT(cfg);b.update(a.parameters())
    oc=OptimizerConfig(32)
    native=MultiOptimizer(a,oc);staged=StagedOptimizer(b,oc,compress)
    for step in range(20):
        grads=tree_unflatten([(p,mx.random.normal(v.shape)*.001) for p,v in tree_flatten(a.parameters())]);mx.eval(grads)
        for opt in (native,staged):
            opt.set_lr_multiplier(1-step*.02);opt.set_muon_momentum(.85+step*.003);opt.set_muon_weight_decay(.1-step*.002)
        native.update(a,grads);mx.eval(a.parameters(),native.state)
        staged.update(b,grads);mx.eval(b.parameters())
        for (p,v),(q,w) in zip(tree_flatten(a.parameters()),tree_flatten(b.parameters())):
            assert p==q
            np.testing.assert_array_equal(np.asarray(v).view(np.uint32),np.asarray(w).view(np.uint32))
        assert not staged.adam_state and not staged.muon_state and not staged.state
        for path,h in staged.handles.items():
            for key,v in h.items():
                ref=native.muon_state[path] if key=='buf' else native.adam_state[path][key]
                actual=decode_exact(v) if isinstance(v,GrammarMatrix) else v
                if key=='t':assert actual==ref
                else:np.testing.assert_array_equal(np.asarray(actual).view(np.uint32),np.asarray(ref).view(np.uint32))
    staged.export_dense(tmp_path/'optim.safetensors')
    restored=MultiOptimizer(b,oc);_load_optimizer_state(restored,str(tmp_path/'optim.safetensors'))
    for k,s in native.adam_state.items():
        assert restored.adam_state[k]['t']==20
        np.testing.assert_array_equal(np.asarray(s['m']).view(np.uint32),np.asarray(restored.adam_state[k]['m']).view(np.uint32))

def test_mlp_local_and_finite_difference():
    from nanochat_mlx.experiments.activation_tape import mlp_backward
    from nanochat_mlx.gpt import norm
    mx.random.seed(117)
    a=mx.random.normal((1,3,4));w1=mx.random.normal((7,4))*.2;w2=mx.random.normal((4,7))*.2;g=mx.random.normal((1,3,4))
    def scalar(a,w1,w2):return mx.sum((a+(mx.maximum(norm(a)@w1.T,0)**2)@w2.T)*g)
    value,reference=mx.vjp(scalar,[a,w1,w2],[mx.array(1.)])
    actual=mlp_backward(a,mx.maximum(norm(a)@w1.T,0),w1,w2,g)
    for x,y in zip(actual,reference):np.testing.assert_allclose(np.asarray(x),np.asarray(y),atol=1e-5,rtol=1e-4)
    direction=mx.ones_like(w2)*.1;analytic=mx.sum(reference[2]*direction).item()
    approximations=[]
    for eps in (1e-2,1e-3,1e-4):
        fd=((scalar(a,w1,w2+eps*direction)-scalar(a,w1,w2-eps*direction))/(2*eps)).item()
        approximations.append(abs(fd-analytic))
    assert min(approximations)<1e-4

@pytest.mark.parametrize('arm',['tape_dense','tape_recompute','tape_bitmap'])
def test_ten_updates_accumulation(arm):
    from nanochat_mlx.optim import MultiOptimizer,OptimizerConfig
    from mlx.utils import tree_map
    mx.random.seed(333)
    cfg=GPTConfig(sequence_len=16,vocab_size=67,n_layer=2,n_head=2,n_kv_head=2,n_embd=32,window_pattern='SL')
    a=GPT(cfg);a.init_weights();b=GPT(cfg);b.update(a.parameters());mx.eval(a.parameters(),b.parameters())
    oa=MultiOptimizer(a,OptimizerConfig(32,matrix_lr=.0001,embedding_lr=.0001,scalar_lr=.0001,unembedding_lr=.0001))
    ob=MultiOptimizer(b,OptimizerConfig(32,matrix_lr=.0001,embedding_lr=.0001,scalar_lr=.0001,unembedding_lr=.0001))
    for step in range(10):
        ga=gb=None
        for micro in range(2):
            x=mx.array([[1,2,2,3,4,5,6,7,8]]);y=mx.array([[2,2,3,4,5,6,7,8,-1]])
            va,g=nn.value_and_grad(a,loss_fn)(a,x,y);vb,h=loss_and_grad(b,x,y,arm)
            ga=g if ga is None else tree_map(lambda x,y:x+y,ga,g)
            gb=h if gb is None else tree_map(lambda x,y:x+y,gb,h)
            mx.eval(ga,gb)
        oa.update(a,tree_map(lambda x:x*.5,ga));ob.update(b,tree_map(lambda x:x*.5,gb));mx.eval(a.parameters(),b.parameters(),oa.state,ob.state)
        assert abs(va.item()-vb.item())<1e-4
        for (k,x),(l,y) in zip(tree_flatten(a.parameters()),tree_flatten(b.parameters())):
            assert k==l
            np.testing.assert_allclose(np.asarray(x),np.asarray(y),atol=1e-5,rtol=1e-3)

def test_replay_checkpoint_resume(tmp_path):
    from nanochat_mlx.experiments.replay import save_checkpoint,load_checkpoint
    from nanochat_mlx.optim import MultiOptimizer,OptimizerConfig
    from nanochat_mlx.experiments.staged_optimizer import StagedOptimizer
    from mlx.utils import tree_unflatten
    mx.random.seed(42)
    cfg=GPTConfig(sequence_len=8,vocab_size=67,n_layer=2,n_head=2,n_kv_head=2,n_embd=32)
    a=GPT(cfg);a.init_weights();oa=StagedOptimizer(a,OptimizerConfig(32),compress=True)
    c={'config_sha256':'test','accumulation':2}
    for step in range(10):
        grads=tree_unflatten([(k,mx.ones_like(v)*.001) for k,v in tree_flatten(a.parameters())]);oa.update(a,grads)
    save_checkpoint(tmp_path/'step10',a,oa,10,42,c)
    b=GPT(cfg);b.init_weights();ob=StagedOptimizer(b,OptimizerConfig(32),compress=True)
    s=load_checkpoint(tmp_path/'step10',b,ob,c);assert s['cursor']==20
    for model,opt in ((a,oa),(b,ob)):
        grads=tree_unflatten([(k,mx.ones_like(v)*.002) for k,v in tree_flatten(model.parameters())]);opt.update(model,grads)
    for (k,v),(l,w) in zip(tree_flatten(a.parameters()),tree_flatten(b.parameters())):
        assert k==l
        np.testing.assert_array_equal(np.asarray(v).view(np.uint32),np.asarray(w).view(np.uint32))

@pytest.mark.parametrize('prompt_length',[1,128,512])
def test_direct_layer_incremental_model(prompt_length):
    from nanochat_mlx.experiments.grammar_linear import GrammarLinear
    from nanochat_mlx.engine import KVCache
    mx.random.seed(83)
    cfg=GPTConfig(sequence_len=1024,vocab_size=67,n_layer=2,n_head=2,n_kv_head=2,n_embd=32,window_pattern='SL')
    a=GPT(cfg);a.init_weights();b=GPT(cfg);b.update(a.parameters())
    for i in range(2):
        a.blocks[i].mlp.c_proj.weight=mx.random.normal(a.blocks[i].mlp.c_proj.weight.shape)*.02
    b.update(a.parameters());b.blocks[1].mlp.c_fc=GrammarLinear(b.blocks[1].mlp.c_fc.weight,variant='re_iv')
    # Four independent sequences; no claim of parallel batching.
    for sequence in range(4):
        ca=KVCache(2,a.window_sizes);cb=KVCache(2,b.window_sizes)
        ids=mx.array([[(i+sequence)%67 for i in range(prompt_length)]])
        for k in range(3):
            la=a(ids,kv_cache=ca);lb=b(ids,kv_cache=cb);mx.eval(la,lb)
            np.testing.assert_allclose(np.asarray(la),np.asarray(lb),atol=1e-4,rtol=1e-3)
            assert ca.offset==cb.offset==prompt_length+k
            ids=mx.array([[(k+sequence)%67]])

def test_optimizer_compressed_eligible_states_twenty_steps(tmp_path):
    from nanochat_mlx.optim import MultiOptimizer,OptimizerConfig
    from nanochat_mlx.experiments.staged_optimizer import StagedOptimizer
    from nanochat_mlx.experiments.grammar_matrix import resident_bytes,GrammarMatrix
    from mlx.utils import tree_unflatten
    class Fixtures(nn.Module):
        def __init__(self):
            super().__init__();self.blocks=[nn.Linear(128,512,bias=False),nn.Linear(512,128,bias=False)];self.wte=nn.Embedding(256,256)
    mx.random.seed(9);a=Fixtures();b=Fixtures();b.update(a.parameters())
    oa=MultiOptimizer(a,OptimizerConfig(256));ob=StagedOptimizer(b,OptimizerConfig(256),True)
    for step in range(20):
        grads=tree_unflatten([(k,mx.full(v.shape,.001*(step+1))) for k,v in tree_flatten(a.parameters())])
        for opt in (oa,ob):opt.set_lr_multiplier(1-step*.01);opt.set_muon_weight_decay(.2-step*.005)
        oa.update(a,grads);mx.eval(a.parameters(),oa.state);ob.update(b,grads);mx.eval(b.parameters())
        for (k,v),(l,w) in zip(tree_flatten(a.parameters()),tree_flatten(b.parameters())):
            np.testing.assert_array_equal(np.asarray(v).view(np.uint32),np.asarray(w).view(np.uint32))
    handles=[h for s in ob.handles.values() for h in s.values() if isinstance(h,GrammarMatrix)]
    assert any(resident_bytes(h)['raw_blocks']<resident_bytes(h)['blocks'] for h in handles)
    ob.save_handles(tmp_path/'handles')
    restored=StagedOptimizer(b,OptimizerConfig(256),True)
    restored.restore_handles(tmp_path/'handles')
    for path,state in ob.handles.items():
        for key,h in state.items():
            saved=restored.handles[path][key]
            if isinstance(h,GrammarMatrix):assert h.blocks==saved.blocks and h.source_sha256==saved.source_sha256
            elif key=='t':assert h==saved
            else:np.testing.assert_array_equal(h,saved)

def test_bitmap_releases_dense_metal_storage():
    import gc
    from nanochat_mlx.experiments.bitmap import PackedPositive
    mx.synchronize();gc.collect();before=mx.get_active_memory()
    p=mx.maximum(mx.random.normal((1024,1024)),0);mx.eval(p)
    dense_active=mx.get_active_memory();h=PackedPositive.pack(p)
    assert h.mode=='BITMAP' and h.values.flags.owndata
    del p;gc.collect();mx.synchronize();after=mx.get_active_memory()
    assert dense_active-after>=1024*1024*4
    assert after<=before+4096
    restored=h.unpack('mlx');mx.eval(restored)
    assert restored.nbytes==1024*1024*4
    del restored,h;gc.collect();mx.synchronize()
