"""Scientific controls: immutable records, document partitioning and gates."""
import json
import subprocess
import sys

import numpy as np
import pytest

from nanochat_mlx.experiments.v2.data import split_for,packed_rows
from nanochat_mlx.experiments.v2.records import ledger_append,ledger_read,write_json
from nanochat_mlx.experiments.v2.confirmation import paired_estimate,quality_estimate
from nanochat_mlx.experiments.v2.reporting import decision


def test_ledger_detects_modified_history(tmp_path):
    p=tmp_path/'ledger.jsonl'
    ledger_append(p,{'event':'start'});ledger_append(p,{'event':'failure','metric':None})
    assert ledger_read(p)[1]['metric'] is None
    p.write_text(p.read_text().replace('failure','success'))
    with pytest.raises(ValueError):ledger_read(p)


def test_exclusive_manifest(tmp_path):
    p=tmp_path/'m.json';write_json(p,{'x':1})
    with pytest.raises(FileExistsError):write_json(p,{'x':2})
    with pytest.raises(ValueError):write_json(tmp_path/'nan.json',{'x':float('nan')})
    assert not (tmp_path/'nan.json').exists()


def test_doc_duplicates_always_share_partition():
    ids=[str(i) for i in range(1000)]
    a=[split_for(i,'fixed') for i in ids]
    assert set(a)=={'train','calibration','reserved'}
    assert a==[split_for(i,'fixed') for i in ids]


def test_packing_preserves_document_segments():
    class Tokenizer:
        def get_bos_token_id(self):return 0
        def encode(self,texts,prepend):return [[prepend]+list(map(int,t.split())) for t in texts]
    docs=[('same',i,'1 2 3') for i in range(2000)]
    rows=list(packed_rows(docs,Tokenizer(),7,2,buffer_size=4))
    for row,meta in rows:
        assert row.tolist()==[0,1,2,3,0,1,2,3]
        assert len(meta['segments'])==2
        assert meta['segments'][0]['occurrence']!=meta['segments'][1]['occurrence']
        assert meta['segments'][-1]['row_end']==8


def test_insufficient_replicas_cannot_confirm():
    with pytest.raises(ValueError):paired_estimate([1]*9,[1]*9)
    with pytest.raises(ValueError):quality_estimate([1]*3,[1]*3)
    assert paired_estimate([.8]*10,[1]*10)['upper95']==pytest.approx(.8)
    assert quality_estimate([1]*5,[1]*5)['pass']
    assert not quality_estimate([1.02]*5,[1]*5)['pass']


def test_missing_or_failed_metrics_do_not_promote():
    assert decision({'correctness_pass':False},None)=='FAILURE_CORRECTNESS_OR_RUN'
    r={'correctness_pass':True,'trial_id':'bitmap','scenario':{'arm':'bitmap_gpu'},'metrics':{}}
    assert decision(r,None)=='INCONCLUSIVE_NO_MATCHED_CONTROL'


def test_reserved_results_never_enter_discovery_shortlist():
    r={'correctness_pass':True,'trial_id':'confirm-quality-candidate-s211',
       'request':{'quality_split':'reserved'}}
    assert decision(r,None)=='SEALED_CONFIRMATION_ONLY'


def test_changed_trial_artifact_blocks_report(tmp_path):
    from nanochat_mlx.experiments.v2.records import digest
    from nanochat_mlx.experiments.v2.reporting import verified_trials
    trial=tmp_path/'trials'/'one';trial.mkdir(parents=True)
    write_json(trial/'summary.json',{'status':'CRASH_OR_TIMEOUT'})
    write_json(trial/'request.json',{'hypothesis':'test'})
    write_json(trial/'artifacts.json',{p.name:digest(p) for p in trial.iterdir()})
    ledger_append(tmp_path/'ledger.jsonl',{'event':'trial_finished','trial':'one',
        'summary_sha256':digest(trial/'summary.json'),'artifacts_sha256':digest(trial/'artifacts.json')})
    assert verified_trials(tmp_path)[0]['status']=='CRASH_OR_TIMEOUT'
    (trial/'request.json').write_text('{}')
    with pytest.raises(ValueError,match='artifact changed'):verified_trials(tmp_path)


def test_reserved_read_guard_in_fresh_process(tmp_path):
    reserved=tmp_path/'reserved_x.npy';reserved.write_bytes(b'secret')
    code='from pathlib import Path; from nanochat_mlx.experiments.v2.worker import protect_reserved; protect_reserved(Path('+repr(str(tmp_path))+')); open('+repr(str(reserved))+').read()'
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode!=0 and 'Discovery cannot read reserved' in result.stderr


def test_checkpoint_control_gradients():
    mx=pytest.importorskip('mlx.core')
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from nanochat_mlx.gpt import GPT,GPTConfig,loss_fn
    from nanochat_mlx.experiments.v2.worker import gradient_fn
    mx.random.seed(71)
    model=GPT(GPTConfig(sequence_len=16,vocab_size=67,n_layer=4,n_head=2,n_kv_head=2,n_embd=32))
    model.init_weights()
    for block in model.blocks:block.mlp.c_proj.weight=mx.random.normal(block.mlp.c_proj.weight.shape)*.03
    x=mx.array([[1,2,2,3,4,5,6,7]]);y=mx.array([[2,2,3,4,5,6,7,-1]])
    a,g=nn.value_and_grad(model,loss_fn)(model,x,y);mx.eval(a,g)
    b,h=gradient_fn(model,{'arm':'checkpoint_native'})(model,x,y);mx.eval(b,h)
    assert abs(a.item()-b.item())<1e-4
    ga=dict(tree_flatten(g));gb=dict(tree_flatten(h));assert ga.keys()==gb.keys()
    for key in ga:np.testing.assert_allclose(np.asarray(ga[key]),np.asarray(gb[key]),atol=1e-5,rtol=1e-3,err_msg=key)
