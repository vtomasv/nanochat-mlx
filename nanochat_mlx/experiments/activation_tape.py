"""Imperative segmented GPT VJP. No global autodiff wraps the tape."""
import time
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten,tree_unflatten
from nanochat_mlx.gpt import norm
from .bitmap import PackedPositive

def detached(x):
    mx.eval(x)
    y=mx.stop_gradient(x)
    mx.eval(y)
    return y

def mlp_backward(a,p,w1,w2,g):
    u=norm(a); h=p*p
    g2=g.reshape(-1,g.shape[-1]); h2=h.reshape(-1,h.shape[-1]); u2=u.reshape(-1,u.shape[-1])
    dw2=g2.T@h2
    dz=(2*p)*(g@w2)
    dw1=dz.reshape(-1,dz.shape[-1]).T@u2
    du=dz@w1
    _,adj=mx.vjp(norm,[a],[du])
    da=g+adj[0]
    mx.eval(dw1,dw2,da)
    return da,dw1,dw2

def loss_and_grad(model,ids,targets,arm='tape_bitmap',sample_bits=256,profile=False):
    if arm not in ('tape_dense','tape_bitmap','tape_recompute'): raise ValueError(arm)
    started=time.perf_counter() if profile else 0.;unpack_seconds=0.
    masks=model._get_masks(ids.shape[1]); tape=[]; accounting=[]
    x0=detached(norm(model.wte(ids))); x=x0
    for i,block in enumerate(model.blocks):
        xin=x
        s=model.resid_lambdas[i]*xin+model.x0_lambdas[i]*x0
        ve=model.value_embeds[str(i)](ids) if str(i) in model.value_embeds else None
        a=detached(s+block.attn(norm(s),ve,mask=masks[i]))
        p=detached(mx.maximum(block.mlp.c_fc(norm(a)),0))
        x=detached(a+block.mlp.c_proj(p*p))
        if arm=='tape_bitmap':
            saved=PackedPositive.pack(p,sample_bits)
            accounting.append(dict(path=f'blocks.{i}.P',raw_bytes=p.nbytes,resident_bytes=saved.resident_bytes(),mode=saved.mode,encode_seconds=saved.pack_seconds))
        elif arm=='tape_dense':
            saved=p; accounting.append(dict(path=f'blocks.{i}.P',raw_bytes=p.nbytes,resident_bytes=p.nbytes,mode='DENSE',encode_seconds=0.))
        else:
            saved=None; accounting.append(dict(path=f'blocks.{i}.P',raw_bytes=p.nbytes,resident_bytes=0,mode='RECOMPUTE',encode_seconds=0.))
        tape.append((xin,a,saved))
        del p,s,ve,saved
    model._navarro_accounting=accounting
    if profile:
        mx.synchronize();forward_seconds=time.perf_counter()-started;backward_started=time.perf_counter()
    def final(z,w):
        logits=(norm(z)@w.T)[...,:model.config.vocab_size].astype(mx.float32)
        logits=15*mx.tanh(logits/15)
        mask=targets!=-1; safe=mx.where(mask,targets,mx.zeros_like(targets))
        return mx.sum(nn.losses.cross_entropy(logits,safe,reduction='none')*mask)/mx.maximum(mx.sum(mask),1)
    loss,adj=mx.vjp(final,[x,model.lm_head.weight],[mx.array(1.,dtype=mx.float32)])
    loss=loss[0] if isinstance(loss,list) else loss
    g,dhead=adj; mx.eval(loss,g,dhead)
    grads={'lm_head.weight':dhead}; gx0=mx.zeros_like(x0); resid=[]; xzero=[]
    for i in reversed(range(len(model.blocks))):
        block=model.blocks[i]; xin,a,saved=tape.pop()
        if arm=='tape_bitmap':
            unpack_started=time.perf_counter() if profile else 0.
            p=saved.unpack('mlx')
            if profile:
                mx.eval(p);mx.synchronize();unpack_seconds+=time.perf_counter()-unpack_started
        elif arm=='tape_dense': p=saved
        else: p=mx.maximum(block.mlp.c_fc(norm(a)),0)
        da,dw1,dw2=mlp_backward(a,p,block.mlp.c_fc.weight,block.mlp.c_proj.weight,g)
        grads[f'blocks.{i}.mlp.c_fc.weight']=dw1; grads[f'blocks.{i}.mlp.c_proj.weight']=dw2
        del p,saved,a
        ap=block.attn.parameters(); af=tree_flatten(ap); has_ve=str(i) in model.value_embeds
        args=[xin,x0,model.resid_lambdas[i],model.x0_lambdas[i],*[v for _,v in af]]
        if has_ve: args.append(model.value_embeds[str(i)].weight)
        def attention_segment(xin,xzero,r,s,*weights):
            # All trainable tensors are explicit VJP inputs. Restore module after tracing.
            block.attn.update(tree_unflatten([(k,v) for (k,_),v in zip(af,weights)]))
            z=r*xin+s*xzero
            ve=weights[-1][ids] if has_ve else None
            return z+block.attn(norm(z),ve,mask=masks[i])
        try:
            _,adj=mx.vjp(attention_segment,args,[da])
            mx.eval(adj)
        finally: block.attn.update(ap)
        g=detached(adj[0]); gx0=detached(gx0+adj[1]); resid.append(adj[2]); xzero.append(adj[3])
        for (path,_),value in zip(af,adj[4:4+len(af)]): grads[f'blocks.{i}.attn.{path}']=value
        if has_ve: grads[f'value_embeds.{i}.weight']=adj[-1]
        del adj,da,args,xin
    _,adj=mx.vjp(lambda w:norm(w[ids]),[model.wte.weight],[g+gx0])
    grads['wte.weight']=adj[0]
    grads['resid_lambdas']=mx.stack(resid[::-1]); grads['x0_lambdas']=mx.stack(xzero[::-1])
    mx.eval(loss,grads)
    if profile:
        mx.synchronize()
        model._navarro_profile={'forward_including_pack_seconds':forward_seconds,'backward_including_unpack_seconds':time.perf_counter()-backward_started,'nested_pack_seconds':sum(r['encode_seconds'] for r in accounting),'nested_unpack_seconds':unpack_seconds,'additional_attention_forwards':len(model.blocks),'additional_fc_forwards':len(model.blocks) if arm=='tape_recompute' else 0}
    return loss,tree_unflatten(list(grads.items()))
