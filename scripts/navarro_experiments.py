"""Entry point for the isolated Navarro protocol."""
import argparse
import json
import os
from pathlib import Path
from nanochat_mlx.experiments.config import ROOT,load_config,dump

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('command',choices=['preflight','prepare','baseline','run','report','profile','inference','repo-native','conclude','diagnostics','campaign','detailed'])
    p.add_argument('--config',default='configs/navarro/primary.json')
    p.add_argument('--experiment',choices=['E1','E2','E3'])
    p.add_argument('--stage',choices=['smoke','pilot','confirm','reference'],default='pilot')
    p.add_argument('--arm')
    p.add_argument('--replay-layout',choices=['original','contiguous'],default='contiguous')
    p.add_argument('--dtype',choices=['float32'],default='float32')
    p.add_argument('--mode',choices=['replay','pipeline'],default='replay')
    p.add_argument('--seed',type=int)
    p.add_argument('--steps',type=int)
    p.add_argument('--checkpoint')
    p.add_argument('--run-id')
    p.add_argument('--results',default='experiments/navarro/results')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume-run')
    a=p.parse_args(); c=load_config(a.config); c['measurement_mode']=a.mode; c['replay_layout']=a.replay_layout; c['input_checkpoint']=str(Path(a.checkpoint).resolve()) if a.checkpoint else None; os.environ['NANOCHAT_BASE_DIR']=c['data_dir']
    if a.steps is not None and not 1<=a.steps<=1000:p.error('--steps must be between 1 and 1000')
    if a.command=='run' and a.experiment is None:p.error('run requires --experiment')
    if a.command=='run' and a.arm:
        from nanochat_mlx.experiments.orchestrate import ARMS
        if a.arm not in ARMS.get(a.experiment,[]):p.error('Arm does not belong to this training experiment')
    if a.command=='report':
        rows=[]
        for path in sorted(Path(a.results).glob('*/*/summary.json')):
            s=json.loads(path.read_text()); rows.append(f'| {s["experiment"]} | {s["profile"]} | {s["status"]} | {s["decision"]} | [{path.parent.name}]({path.parent.relative_to(Path(a.results))}/report.md) |')
        (Path(a.results)/'report.md').write_text('# Navarro experiments\n\n| Experiment | Profile | State | Decision | Evidence |\n|---|---|---|---|---|\n'+'\n'.join(rows)+'\n')
        from nanochat_mlx.experiments.reporting import render_existing_report
        from nanochat_mlx.experiments.final_report import render
        from nanochat_mlx.experiments.audit import verify_results
        for p in Path(a.results).glob('*/*/summary.json'):
            if json.loads(p.read_text())['status']!='not_measured':render_existing_report(p.parent)
        render(a.results);verify_results(a.results)
        return
    if a.dry_run:
        import shutil
        from nanochat_mlx.experiments.orchestrate import ARMS
        estimates={}
        for arm in ARMS.get(a.experiment,[]):
            candidate=Path(a.results)/(a.experiment or 'baseline')/f"{c['profile']}-pilot-{arm}-s101"/'summary.json'
            if candidate.exists():
                result=json.loads(candidate.read_text()).get('intervention') or {}
                estimates[arm]=result.get('mean_step_seconds')
        seconds=sum(v*4000 for v in estimates.values()) if estimates and all(v is not None for v in estimates.values()) and len(estimates)==len(ARMS.get(a.experiment,[])) else None
        print(json.dumps({'config':c,'stage':a.stage,'experiment':a.experiment,'estimated_seconds':seconds,'estimate_reason':'4000 updates per arm: 3000 learning + 1000 timing; exclude I/O and diagnostics' if seconds else 'requires completed comparable pilot','pilot_step_seconds':estimates,'free_disk_bytes':shutil.disk_usage(ROOT).free,'tape_bytes':1100*c['total_batch_size']*8,'confirmation':'3 seeds x 1000 steps per arm; 5 process pairs at each of 400 and 900'},indent=2)); return
    if a.resume_run:
        root=Path(a.resume_run)
        s=json.loads((root/'summary.json').read_text())
        if s['status']=='measured': print(json.dumps(s,indent=2)); return
        manifest=json.loads((root/'manifest.json').read_text())
        if manifest['config']['config_sha256']!=c['config_sha256']:raise ValueError('Resume configuration changed')
        events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
        checkpoints=[e for e in events if e['event']=='checkpoint']
        if not checkpoints:raise ValueError('No completed checkpoint to resume; original evidence retained')
        checkpoint=checkpoints[-1]
        loaded=[e for e in events if e['event']=='loaded_state']
        target=loaded[-1]['target_step'] if loaded else next(e['steps'] for e in events if e['event']=='start')
        remaining=target-checkpoint['step']
        if remaining<=0:raise ValueError('Training completed, but final evaluation/report must be recovered separately')
        from nanochat_mlx.experiments.runner import train_run
        c=manifest['config'];c['resumed_from']=str(root.resolve());c['input_checkpoint']=str(Path(checkpoint['path']).resolve())
        resumed=train_run(c,manifest['arm'],manifest['stage'],manifest['seed'],remaining,checkpoint['path'],s['experiment'],a.run_id)
        print(resumed);return
    if a.command=='preflight':
        from nanochat_mlx.experiments.measure import environment
        d=environment(); Path(c['artifact_dir']).mkdir(parents=True,exist_ok=True); dump(Path(c['artifact_dir'])/'preflight.json',d); print(json.dumps(d,indent=2)); return
    if a.command=='prepare':
        from nanochat_mlx.experiments.replay import prepare
        print(json.dumps(prepare(c),indent=2)); return
    if a.command=='run' and a.arm is None:
        from nanochat_mlx.experiments.orchestrate import study
        study(c,a.config,a.experiment,a.stage); return
    if a.stage=='confirm':
        raise ValueError('Confirmatory execution requires a completed correctness and pilot gate dossier; none has been approved by the automated evidence gate yet')
    if a.command=='detailed':
        from nanochat_mlx.experiments.detailed import profile
        print(profile(c,a.arm or 'native'));return
    if a.command=='campaign':
        from nanochat_mlx.experiments.orchestrate import campaign
        campaign(c,a.config);return
    if a.command=='diagnostics':
        from nanochat_mlx.experiments.diagnostics import e3_correctness,native_equivalence,native_variability,external_checkpoint
        if a.arm=='external':print(external_checkpoint(c,a.run_id))
        elif a.arm=='native_repeat':native_variability(c)
        elif a.experiment=='E3':print(e3_correctness(c,a.checkpoint))
        else:print(native_equivalence(c,a.steps or 100,a.checkpoint))
        return
    if a.command=='conclude':
        from nanochat_mlx.experiments.conclusions import conclude
        print(conclude(c,a.experiment)); return
    if a.command=='repo-native':
        from nanochat_mlx.experiments.native_cli import run_original
        run_original(c,a.seed or 101,a.steps or 100,a.run_id); return
    if a.command=='profile':
        from nanochat_mlx.experiments.screening import profile_checkpoint
        profile_checkpoint(c,a.checkpoint,a.experiment,a.run_id); return
    if a.command=='inference':
        from nanochat_mlx.experiments.screening import inference_run
        inference_run(c,a.checkpoint,a.arm,a.run_id); return
    from nanochat_mlx.experiments.runner import train_run
    train_run(c,a.arm or 'native',a.stage,a.seed if a.seed is not None else (17 if a.stage=='reference' else 101),a.steps or (1000 if a.stage=='reference' else 100),a.checkpoint,a.experiment or 'baseline',a.run_id)

if __name__=='__main__': main()
