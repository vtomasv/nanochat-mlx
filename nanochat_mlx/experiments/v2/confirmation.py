"""Separate frozen confirmation: paired seeds, process windows and held-out NLL."""
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.stats import t

from .records import canonical,digest,write_json,ledger_append,ledger_read


def paired_estimate(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if a.shape!=b.shape or len(a)<10 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or np.any(a<=0) or np.any(b<=0):raise ValueError('Ten complete positive process pairs required')
    logs=np.log(a/b);rng=np.random.default_rng(20260915)
    indices=rng.integers(0,len(logs),(10000,len(logs)))
    boot=np.exp(logs[indices].mean(axis=1))
    return {'ratios':(a/b).tolist(),'geomean':float(np.exp(logs.mean())),
        'upper95':float(np.quantile(boot,.95)),'ci95':np.quantile(boot,[.025,.975]).tolist()}


def quality_estimate(a,b):
    diff=np.asarray(a,float)-np.asarray(b,float)
    if diff.shape!=(5,) or not np.all(np.isfinite(diff)):raise ValueError('Five paired seed trajectories required')
    upper=float(diff.mean()+t.ppf(.95,4)*diff.std(ddof=1)/np.sqrt(5))
    return {'differences':diff.tolist(),'upper95':upper,'margin':.01,'pass':upper<.01,
        'assumption':'Student t interval for five independent paired seed differences; no document pseudo-replication.'}


def dispatch_confirmation(root,m,args):
    from .reporting import report,verified_trials
    from .campaign import execute
    report(root)
    if args.command=='seal':
        shortlist=json.loads((root/'decisions.json').read_text())['shortlist']
        selected=args.trial or (shortlist[0] if shortlist else None)
        if selected not in shortlist:raise ValueError('No admissible pilot shortlist; do not open held-out data')
        trial=root/'trials'/selected;request=json.loads((trial/'request.json').read_text())
        result=json.loads((trial/'summary.json').read_text())
        trials={r['trial_id']:r for r in verified_trials(root)}
        native=trials['native']['metrics']['mean_step_seconds'];candidate=result['metrics']['mean_step_seconds']
        dense=trials['tape-dense']['metrics']['mean_step_seconds']
        recompute=trials['tape-recompute']['metrics']['mean_step_seconds']
        estimate=5000*(native+candidate)+2000*(native+candidate+dense+recompute)
        spent=sum(e.get('seconds',0) for e in ledger_read(root/'ledger.jsonl') if e['event']=='trial_finished')
        # Include startup, gates and held-out evaluation headroom; not just updates.
        estimate=estimate*1.3+900
        if spent+estimate>m['config']['campaign_budget_seconds']:raise ValueError('Confirmation exceeds remaining registered budget; new budget amendment required')
        if shutil.disk_usage(Path(m['artifact_root'])).free<(8+m['config']['disk_reserve_gib'])*1024**3:raise OSError('Confirmation needs 8 GiB plus disk reserve')
        plan={'protocol_version':'2.0','candidate_trial':selected,'candidate_sha256':digest(trial/'candidate.py'),
            'scenario':request['scenario'],'seeds':[211,307,401,503,601],'windows':[400,900],
            'process_pairs_per_window':10,'window_steps':100,'trajectory_steps':1000,
            'data_manifest_sha256':m['data_manifest_sha256'],'campaign_manifest_sha256':digest(root/'manifest.json'),
            'endpoint':'V2-A memory','nll_margin':.01,'familywise_comparisons':1,'estimated_training_seconds':estimate,
            'sample_size_basis':'Operational minimum from protocol; small seed count limits precision. Inconclusive if intervals cross margins.',
            'selection':'Frozen after calibration; no subsequent candidate edits or optional stopping.'}
        write_json(root/'sealed-plan.json',plan);ledger_append(root/'ledger.jsonl',{'event':'confirmation_sealed','plan_sha256':digest(root/'sealed-plan.json')});return
    plan=json.loads((root/'sealed-plan.json').read_text())
    seal_events=[e for e in ledger_read(root/'ledger.jsonl') if e['event']=='confirmation_sealed']
    if not seal_events or seal_events[-1]['plan_sha256']!=digest(root/'sealed-plan.json'):raise ValueError('Sealed plan changed')
    def budget_check():
        spent=sum(e.get('seconds',0) for e in ledger_read(root/'ledger.jsonl') if e['event']=='trial_finished')
        if spent>=m['config']['campaign_budget_seconds']:raise ValueError('Confirmation budget exhausted; no partial decision')
    if plan['campaign_manifest_sha256']!=digest(root/'manifest.json'):raise ValueError('Sealed campaign changed')
    candidate=root/'trials'/plan['candidate_trial']/'candidate.py'
    if digest(candidate)!=plan['candidate_sha256']:raise ValueError('Sealed candidate changed')
    scenarios={'native':{'id':'native','arm':'native'},'candidate':plan['scenario']}
    quality={arm:[] for arm in scenarios};configs={}
    for seed in plan['seeds']:
        c=dict(m['config']);c['seed']=seed;c.pop('config_sha256');c['config_sha256']=hashlib.sha256(canonical(c)).hexdigest();configs[seed]=c
        for arm in ('native','candidate'):
            trial_id=f'confirm-quality-{arm}-s{seed}';path=root/'trials'/trial_id/'summary.json'
            if not path.exists():
                budget_check()
                execute(root,m,scenarios[arm],'reference' if arm=='native' else 'confirm_trajectory',1000,trial_id,
                    hypothesis='Sealed paired quality trajectory',candidate=candidate,config_override=c,quality_split='reserved')
            trials={r['trial_id']:r for r in verified_trials(root)};r=trials[trial_id]
            if r['status']!='MEASURED_EXPLORATORY':raise ValueError('Incomplete confirmation retained; cannot decide')
            quality[arm].append(r['quality']['nll'])
    estimates=[];c=configs[plan['seeds'][0]];rng=np.random.default_rng(20260915)
    for window in plan['windows']:
        timing_scenarios={**scenarios,'dense':{'id':'tape-dense','arm':'tape_dense'},'recompute':{'id':'tape-recompute','arm':'tape_recompute'}}
        metrics={arm:[] for arm in timing_scenarios}
        checkpoint=Path(m['artifact_root'])/f"confirm-quality-native-s{c['seed']}"/f'step-{window:04d}'
        for pair in range(10):
            order=list(timing_scenarios);rng.shuffle(order)
            for arm in order:
                trial_id=f'confirm-time-{window}-{pair:02d}-{arm}';path=root/'trials'/trial_id/'summary.json'
                if not path.exists():
                    budget_check()
                    execute(root,m,timing_scenarios[arm],'confirm_window',100,trial_id,checkpoint,
                    hypothesis='Sealed paired process window',candidate=candidate,config_override=c)
                r={r['trial_id']:r for r in verified_trials(root)}[trial_id]
                if r['status']!='MEASURED_EXPLORATORY':raise ValueError('Missing timing pair')
                metrics[arm].append(r['metrics'])
        estimates.append({'window':window,'contrasts':{control:{key:paired_estimate([r[key] for r in metrics['candidate']],[r[key] for r in metrics[control]]) for key in ('mean_step_seconds','mlx_active_peak_bytes','rss_peak_bytes')} for control in ('native','dense','recompute')}})
    q=quality_estimate(quality['candidate'],quality['native'])
    global_pass=all(e['contrasts']['native']['mean_step_seconds']['upper95']<=1.1 and e['contrasts']['native']['mlx_active_peak_bytes']['upper95']<=.9 and e['contrasts']['native']['rss_peak_bytes']['upper95']<=1.05 for e in estimates)
    incremental=all(e['contrasts']['dense']['mlx_active_peak_bytes']['upper95']<=.95 for e in estimates)
    dominates_recompute=all(e['contrasts']['recompute']['mean_step_seconds']['upper95']<1 or e['contrasts']['recompute']['mlx_active_peak_bytes']['upper95']<1 for e in estimates)
    success=q['pass'] and global_pass and incremental and dominates_recompute
    result={'quality':q,'process_windows':estimates,'global_memory_criteria_pass':global_pass,
        'incremental_memory_pass':incremental,'not_dominated_by_recompute':dominates_recompute,
        'decision':'SUCCESS_MEMORY' if success else 'INCONCLUSIVE_OR_CRITERIA_NOT_MET',
        'limitation':'One model/hardware and early learning; five seed t interval depends on its stated distributional assumption.'}
    write_json(root/'confirmation.json',result);ledger_append(root/'ledger.jsonl',{'event':'confirmation_evaluated','sha256':digest(root/'confirmation.json')});report(root)
