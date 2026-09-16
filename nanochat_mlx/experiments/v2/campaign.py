"""Budgeted autoresearch scheduler; discovery never opens held-out quality data."""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from .records import ROOT,canonical,digest,now,write_json,snapshot,verify_sources,ledger_append,ledger_read


@contextlib.contextmanager
def lock(root):
    with (root/'.controller.lock').open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another campaign controller owns this directory')
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def config_load(path):
    c=json.loads(Path(path).read_text())
    if c['protocol_version']!='2.0' or c['phase']!='engineering_screening':raise ValueError('Wrong discovery protocol')
    if c['dtype']!='float32' or c['num_iterations']!=1000 or c['accumulation']*c['device_batch_size']*c['sequence_len']!=c['total_batch_size']:
        raise ValueError('Invalid scientific configuration')
    if not 1<=c['pilot_steps']<=100 or not c['pilot_steps']<=c['reference_steps']<=1000:raise ValueError('Invalid horizon')
    if c['max_trials']>32 or c['campaign_budget_seconds']>86400 or c['trial_timeout_seconds']>7200:raise ValueError('Explicit bounded search required')
    if c.get('pilot_checkpoint_policy')!='fingerprints_only':raise ValueError('Unsupported checkpoint policy')
    if not 1<=c.get('disk_reserve_gib',0)<=20:raise ValueError('Explicit disk reserve required')
    if any(not 1<=k<=c['reference_steps'] for k in c['reference_checkpoint_steps']):raise ValueError('Invalid checkpoint step')
    if len({s['id'] for s in c['scenarios']})!=len(c['scenarios']):raise ValueError('Duplicate scenario ID')
    c['data_dir']=str(Path(c['data_dir']).expanduser().resolve())
    c['config_sha256']=hashlib.sha256(canonical(c)).hexdigest()
    return c


def initialize(args):
    from .data import verify
    root=Path(args.campaign).resolve();data=Path(args.data).resolve();c=config_load(args.config)
    manifest=verify(data)
    if manifest['split_salt']!=c['split_salt']:raise ValueError('Data split salt mismatch')
    if manifest['vocab_size']!=32768:raise ValueError('Primary tokenizer size mismatch')
    import numpy as np
    shape=np.load(data/'train_x.npy',mmap_mode='r').shape
    if shape[1:]!=(c['device_batch_size'],c['sequence_len']):raise ValueError('Replay/config shape mismatch')
    if shape[0]<(c['reference_steps']+c['pilot_steps'])*c['accumulation']:raise ValueError('Replay too short')
    required_gib=c['disk_reserve_gib']+2*len(c['reference_checkpoint_steps'])+1
    if shutil.disk_usage(ROOT).free<required_gib*1024**3:raise OSError(f'Need {required_gib} GiB for registered checkpoint policy')
    root.mkdir(parents=True,exist_ok=False);(root/'trials').mkdir()
    sources=snapshot(root/'sources.zip');protected={k:v for k,v in sources['files'].items() if k!=c['candidate_file']}
    artifact=ROOT/'experiments/navarro/artifacts'/('v2-'+root.name)
    if artifact.exists():raise FileExistsError('Artifact ID already exists')
    artifact.mkdir()
    value={'schema':1,'protocol_version':'2.0','created_at':now(),'config':c,'data_root':str(data),
        'data_manifest_sha256':digest(data/'manifest.json'),'artifact_root':str(artifact),
        'initial_checkpoint':str(artifact/f"initial-s{c['seed']}"),'sources':sources,'protected_sources':protected,
        'scientific_scope':'Engineering search, calibration only. Not confirmatory evidence.',
        'upstream':{'url':'https://github.com/karpathy/autoresearch','adaptation':'Independent MLX implementation of a propose/test/keep-discard/log workflow; fixed-work comparisons and protected evaluation.'}}
    write_json(root/'manifest.json',value)
    ledger_append(root/'ledger.jsonl',{'event':'campaign_initialized','manifest_sha256':digest(root/'manifest.json')})
    print(root)


def load(root):
    m=json.loads((root/'manifest.json').read_text());events=ledger_read(root/'ledger.jsonl')
    if events[0]['manifest_sha256']!=digest(root/'manifest.json'):raise ValueError('Campaign manifest changed')
    verify_sources(m['protected_sources']);return m


def gate(root,m):
    destination=root/'gate'
    if destination.exists():
        result=json.loads((destination/'summary.json').read_text())
        if result['sources_hash']!=m['sources']['tree_sha256']:raise ValueError('Gate/source mismatch')
        if digest(destination/'junit.xml')!=result['junit_sha256']:raise ValueError('Gate artifact modified')
        if not result['pass']:raise ValueError('Correctness suite did not pass')
        return
    destination.mkdir()
    command=[sys.executable,'-m','pytest','tests/test_navarro_v2_metal.py','tests/test_navarro_v2_protocol.py',
        'tests/test_navarro_mlx.py','tests/test_navarro_grammar.py','tests/test_navarro_bitmap.py',
        '-q','--junitxml='+str(destination/'junit.xml')]
    with (destination/'console.log').open('x') as f:
        process=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=300)
    suites=list(ET.parse(destination/'junit.xml').getroot().iter('testsuite'))
    counts={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ('tests','failures','errors','skipped')}
    passed=process.returncode==0 and counts['tests']>0 and all(counts[k]==0 for k in ('failures','errors','skipped'))
    result={'pass':passed,'counts':counts,'sources_hash':m['sources']['tree_sha256'],
        'junit_sha256':digest(destination/'junit.xml'),'command':command}
    write_json(destination/'summary.json',result);ledger_append(root/'ledger.jsonl',{'event':'correctness_gate','result':result})
    if not passed:raise ValueError('Real MLX correctness gate failed; inspect gate/console.log')


def execute(root,m,scenario,kind,steps,trial_id,checkpoint=None,hypothesis=None,candidate=None,config_override=None,quality_split='calibration'):
    verify_sources(m['protected_sources'])
    trial=root/'trials'/trial_id;trial.mkdir(exist_ok=False)
    shutil.copyfile(candidate or ROOT/m['config']['candidate_file'],trial/'candidate.py')
    sources=snapshot(trial/'sources.zip')
    config=config_override or m['config']
    initial=m['initial_checkpoint'] if config_override is None else str(Path(m['artifact_root'])/f"initial-confirm-s{config['seed']}")
    request={'kind':kind,'created_at':now(),'config':config,'scenario':scenario,'steps':steps,
        'data_root':m['data_root'],'data_manifest_sha256':m['data_manifest_sha256'],
        'artifact_root':m['artifact_root'],'initial_checkpoint':initial,
        'checkpoint':str(checkpoint) if checkpoint else None,'sources':sources,
        'candidate_sha256':digest(trial/'candidate.py'),'hypothesis':hypothesis or scenario['id'],
        'quality_split':quality_split,'sealed_plan_sha256':digest(root/'sealed-plan.json') if quality_split=='reserved' else None}
    write_json(trial/'request.json',request)
    command=[sys.executable,'-m','scripts.navarro_v2','worker','--campaign',str(root),'--trial',trial_id]
    write_json(trial/'command.json',command)
    ledger_append(root/'ledger.jsonl',{'event':'trial_started','trial':trial_id,'request_sha256':digest(trial/'request.json'),
        'candidate_sha256':request['candidate_sha256'],'hypothesis':request['hypothesis']})
    interrupted=False
    tick=time.monotonic();timeout=m['config']['trial_timeout_seconds']
    if kind in ('reference','confirm_trajectory'):timeout=max(timeout,1800)
    print('START',trial_id,kind,'steps',steps,flush=True)
    with (trial/'console.log').open('x') as output:
        process=subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired,KeyboardInterrupt) as exc:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            code=-1;interrupted=isinstance(exc,KeyboardInterrupt)
            write_json(trial/'interruption.json',{'kind':type(exc).__name__,'at':now()})
    if code!=0 or not (trial/'summary.json').exists():
        # Preserve any pre-crash summary; it cannot count as a successful run.
        if (trial/'summary.json').exists():shutil.move(trial/'summary.json',trial/'incomplete-summary.json')
        write_json(trial/'summary.json',{'status':'CRASH_OR_TIMEOUT','returncode':code,'correctness_pass':False,
            'quality':None,'metrics':None,'scenario':scenario,'seconds':time.monotonic()-tick})
    result=json.loads((trial/'summary.json').read_text())
    artifacts={p.name:digest(p) for p in trial.iterdir() if p.is_file()}
    write_json(trial/'artifacts.json',artifacts)
    ledger_append(root/'ledger.jsonl',{'event':'trial_finished','trial':trial_id,'status':result['status'],
        'seconds':time.monotonic()-tick,'summary_sha256':digest(trial/'summary.json'),
        'artifacts_sha256':digest(trial/'artifacts.json')})
    print('END',trial_id,result['status'],flush=True)
    if interrupted:raise KeyboardInterrupt
    return result


def reference(root,m):
    if (root/'trials/reference/summary.json').exists():
        result=json.loads((root/'trials/reference/summary.json').read_text())
        if result['status']!='MEASURED_EXPLORATORY':raise ValueError('Reference failed; preserve and start a new campaign/amendment')
        return result
    return execute(root,m,{'id':'native-reference','arm':'native'},'reference',m['config']['reference_steps'],'reference',hypothesis='Establish native trajectory before experimental interventions')


def trained_checkpoint(m,step=None):
    step=min(100,m['config']['reference_steps']) if step is None else step
    p=Path(m['artifact_root'])/'reference'/f'step-{step:04d}'
    if not p.exists():raise FileNotFoundError('Need native trained checkpoint: '+str(p))
    return p


def discover(root,m,dry_run=False):
    scenarios=m['config']['scenarios']
    if dry_run:print(json.dumps({'steps':m['config']['pilot_steps'],'scenarios':scenarios,'budget_seconds':m['config']['campaign_budget_seconds'],'evaluation':'calibration only'},indent=2));return
    gate(root,m);reference(root,m)
    ledger=ledger_read(root/'ledger.jsonl');spent=sum(e.get('seconds',0) for e in ledger if e['event']=='trial_finished')
    done=sum(e['event']=='trial_finished' and e['trial']!='reference' for e in ledger)
    for scenario in scenarios:
        trial_id=scenario['id']
        if (root/'trials'/trial_id).exists():
            finished={e['trial'] for e in ledger_read(root/'ledger.jsonl') if e['event']=='trial_finished'}
            if trial_id not in finished:raise ValueError('Unfinished trial retained; register amendment/new campaign: '+trial_id)
            continue
        if done>=m['config']['max_trials'] or spent+m['config']['trial_timeout_seconds']>m['config']['campaign_budget_seconds']:
            ledger_append(root/'ledger.jsonl',{'event':'budget_stop','spent_seconds':spent,'completed_trials':done});break
        tick=time.monotonic()
        execute(root,m,scenario,'pilot',m['config']['pilot_steps'],trial_id,trained_checkpoint(m))
        spent+=time.monotonic()-tick;done+=1
        from .reporting import report
        report(root)
    from .screening import screen
    screen(root,m)
    from .reporting import report
    report(root)
    ledger_append(root/'ledger.jsonl',{'event':'discovery_cycle_finished','completed_trials':done,'spent_seconds':spent})


def dispatch(args):
    root=Path(args.campaign).resolve()
    if args.command=='init':initialize(args);return
    if args.command in ('report','status'):
        from .reporting import report
        print(report(root));return
    with lock(root):
        m=load(root)
        if args.command=='gate':gate(root,m)
        elif args.command=='reference':gate(root,m);reference(root,m)
        elif args.command=='campaign':discover(root,m,args.dry_run)
        elif args.command=='screen':
            from .screening import screen
            screen(root,m)
        elif args.command=='submit':
            if not args.hypothesis or not args.trial:raise ValueError('submit requires --trial and --hypothesis')
            if '/' in args.trial or args.trial in ('.','..'):raise ValueError('Invalid trial ID')
            ledger=ledger_read(root/'ledger.jsonl')
            if sum(e['event']=='trial_finished' and e['trial']!='reference' for e in ledger)>=m['config']['max_trials']:raise ValueError('Trial budget exhausted; register a new campaign')
            spent=sum(e.get('seconds',0) for e in ledger if e['event']=='trial_finished')
            if spent+m['config']['trial_timeout_seconds']>m['config']['campaign_budget_seconds']:raise ValueError('Wall-time budget exhausted')
            scenario={'id':args.trial,'arm':'bitmap_gpu','sample_bits':256,'strategy':'rank'}
            gate(root,m);execute(root,m,scenario,'pilot',m['config']['pilot_steps'],args.trial,trained_checkpoint(m),args.hypothesis,Path(args.candidate))
            from .reporting import report
            report(root)
        elif args.command in ('seal','confirm'):
            from .confirmation import dispatch_confirmation
            dispatch_confirmation(root,m,args)
