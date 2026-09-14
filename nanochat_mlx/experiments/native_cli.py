"""Invoke the unmodified CLI in a safe data root, stopping after a saved prefix.

The original 1000-step schedule remains intact. A checkpoint hook only stops the
engineering prefix after its state has been written; no arithmetic is replaced.
"""
import os
from pathlib import Path
import time
from .config import dump
from .replay import seed_all
from .measure import setup_device,memory

def run_original(c,seed=101,steps=100,label=None):
    base=Path(c['artifact_dir'])/(f'repo-native-s{seed}-{steps}'+(f'-{label}' if label else ''))
    base.mkdir(parents=True,exist_ok=False)
    for name in ('base_data','tokenizer'):
        (base/name).symlink_to(Path(c['data_dir'])/name,target_is_directory=True)
    os.environ['NANOCHAT_BASE_DIR']=str(base)
    from .reporting import Run
    run=Run('baseline',c,'repo_native','pilot',seed)
    from scripts.train import main
    import nanochat_mlx.train as train
    import mlx.core as mx
    setup_device();seed_all(seed)
    original_build=train.build_model
    def build_and_record(*args,**kwargs):
        model=original_build(*args,**kwargs)
        model.save_weights(str(base/'initial.safetensors'))
        return model
    train.build_model=build_and_record
    original_setup=train.setup_optimizer
    def setup_and_record(*args,**kwargs):
        optimizer=original_setup(*args,**kwargs)
        dump(base/'optimizer_config.json',optimizer.param_config)
        return optimizer
    train.setup_optimizer=setup_and_record
    original=train._save_optimizer_state
    def save_then_stop(optimizer,path):
        original(optimizer,path)
        if f'step_{steps:06d}_optim' in path:
            run.event('prefix_complete',step=steps,optimizer_path=path,seconds=time.perf_counter()-tick,**memory())
            raise SystemExit(0)
    train._save_optimizer_state=save_then_stop
    args=['--depth',str(c['depth']),'--max-seq-len',str(c['sequence_len']),'--window-pattern',c['window_pattern'],
          '--num-iterations','1000','--device-batch-size',str(c['device_batch_size']),'--total-batch-size',str(c['total_batch_size']),
          '--eval-every','-1','--eval-steps','0','--sample-every','-1','--save-every',str(steps),'--memory-limit-gb','80']
    run.event('original_cli_arguments',args=args,seed=seed,stop_after_checkpoint=steps)
    tick=time.perf_counter()
    try:main(args)
    except SystemExit as e:
        if e.code:raise
    finally:
        train._save_optimizer_state=original
        train.build_model=original_build
        train.setup_optimizer=original_setup
    mx.synchronize()
    run.finish(status='measured',decision='INCONCLUSIVE',intervention={'steps':steps,'wall_seconds':time.perf_counter()-tick,**memory()},limitations=['Original CLI engineering prefix; schedule remains 1000.','Stop occurs after optimizer checkpoint save; meta file is replaced by experimental event metadata.'])
    dump(base/'experiment.json',{'seed':seed,'step':steps,'run':str(run.root),'config':c})
    return run.root
