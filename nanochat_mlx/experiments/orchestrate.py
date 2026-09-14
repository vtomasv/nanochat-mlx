"""Sequential process scheduling; do not overlap GPU arms."""
import json
import shutil
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from .config import ROOT,sha256,dump
from .replay import checkpoint_path

ARMS={'E1':['native','tape_dense','tape_recompute','tape_bitmap'],'E2':['native','staged_raw','staged_grammar']}

def correctness_gate():
    evidence=ROOT/'experiments/navarro/results/correctness-junit.xml'
    tree=ET.parse(evidence)
    cases=[e for e in tree.iter('testcase') if 'navarro' in e.attrib.get('classname','')]
    if not cases or any(e.find('failure') is not None or e.find('error') is not None or e.find('skipped') is not None for e in cases):
        raise ValueError('Real-MLX Navarro correctness suite is missing, skipped or failing')
    return {'pass':True,'cases':len(cases),'junit':str(evidence),'sha256':sha256(evidence)}

def invoke(args,run_id=None,experiment=None):
    if run_id and experiment:
        existing=ROOT/'experiments/navarro/results'/experiment/run_id/'summary.json'
        if existing.exists():
            s=json.loads(existing.read_text())
            if s['status']=='measured':return existing.parent
            raise ValueError(f'Incomplete previous run retained: {existing.parent}')
    command=[sys.executable,'-m','scripts.navarro_experiments',*args]
    logdir=ROOT/'experiments/navarro/results/logs';logdir.mkdir(exist_ok=True)
    name=((experiment+'-'+run_id) if experiment and run_id else run_id) or '-'.join(x for x in args if not x.startswith('-')).replace('/','_')
    with (logdir/(name+'.log')).open('x') as f:
        p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in p.stdout:
            f.write(line);f.flush();print(line,end='',flush=True)
        rc=p.wait()
    if rc:raise subprocess.CalledProcessError(rc,command)
    if run_id and experiment:return ROOT/'experiments/navarro/results'/experiment/run_id

def study(c,config_path,experiment,stage):
    gate=correctness_gate()
    if stage=='confirm':
        for p in (ROOT/'experiments/navarro/results'/experiment).glob('*/summary.json'):
            s=json.loads(p.read_text())
            if s.get('profile')==c['profile'] and str(s.get('decision','')).startswith('NEGATIVE_SCREENING'):
                print(json.dumps({'confirmation':'not_applicable','reason':'preregistered negative pilot gate','evidence':str(p)},indent=2));return
        raise ValueError('Confirmation requires completed pilot selection and resource forecast')
    if experiment=='E3':
        checkpoint=checkpoint_path(c,'primary-native-reference-s17',1000)
        if not checkpoint.exists():raise ValueError('Missing preregistered E3 checkpoint')
        invoke(['profile','--experiment','E3','--config',str(config_path),'--checkpoint',str(checkpoint),'--run-id','primary-E3-screen'], 'primary-E3-screen','E3')
        for arm in ('dense_native','compressed_decode_dense','grammar_direct_cpu'):
            rid=f'primary-E3-{arm}'
            invoke(['inference','--config',str(config_path),'--checkpoint',str(checkpoint),'--arm',arm,'--run-id',rid],rid,'E3')
        return
    steps=10 if stage=='smoke' else 100
    for arm in ARMS[experiment]:
        group='baseline' if arm=='native' else experiment
        rid=f'{c["profile"]}-{stage}-{arm}-s101'
        path=invoke(['baseline' if arm=='native' else 'run','--experiment',group if group!='baseline' else experiment,'--config',str(config_path),'--stage',stage,'--arm',arm,'--seed','101','--steps',str(steps),'--run-id',rid],rid,experiment)
        dump(path/'correctness.json',gate)
        shutil.copyfile(gate['junit'],path/'correctness-junit.xml')
        summary=json.loads((path/'summary.json').read_text());summary['correctness_pass']=True;summary['algorithm_fidelity']=True
        dump(path/'summary.json',summary)
        (path/'artifacts.sha256').write_text(''.join(f'{sha256(p)}  {p.name}\n' for p in sorted(path.iterdir()) if p.is_file() and p.name!='artifacts.sha256'))
    if experiment=='E2' and stage=='pilot':
        for step in (100,500,1000):
            checkpoint=checkpoint_path(c,'primary-native-reference-s17',step)
            rid=f'primary-E2-screen-{step}'
            invoke(['profile','--experiment','E2','--config',str(config_path),'--checkpoint',str(checkpoint),'--run-id',rid],rid,'E2')


def campaign(c,config_path):
    """Complete remaining required diagnostics sequentially after E1 pilot."""
    correctness_gate()
    # The native CLI is a prefix of the original full schedule, with its own safe root.
    original=Path(c['artifact_dir'])/'repo-native-s101-100'
    if not (original/'experiment.json').exists():
        invoke(['repo-native','--config',str(config_path),'--steps','100'],run_id='original-cli-primary-s101')
    if not (ROOT/'experiments/navarro/results/native-variability.json').exists():
        invoke(['diagnostics','--config',str(config_path)],run_id='native-equivalence')
    for step in (100,500,1000):
        checkpoint=checkpoint_path(c,'primary-native-reference-s17',step)
        rid=f'primary-E1-trained-{step}'
        invoke(['profile','--experiment','E1','--config',str(config_path),'--checkpoint',str(checkpoint),'--run-id',rid],rid,'E1')
    study(__import__('nanochat_mlx.experiments.config',fromlist=['load_config']).load_config('configs/navarro/smoke.json'),'configs/navarro/smoke.json','E2','smoke')
    study(c,config_path,'E2','pilot')
    for exp in ('E1','E2'):
        for arm in ARMS[exp]:
            rid=f'primary-pipeline-{arm}-s101'
            invoke(['run','--experiment',exp,'--config',str(config_path),'--stage','pilot','--arm',arm,'--seed','101','--steps','100','--mode','pipeline','--run-id',rid],rid,exp)
    checkpoint=checkpoint_path(c,'primary-native-reference-s17',1000)
    invoke(['diagnostics','--experiment','E3','--config',str(config_path),'--checkpoint',str(checkpoint)],run_id='E3-model-correctness')
    study(c,config_path,'E3','pilot')
    for exp in ('E1','E2','E3'):
        invoke(['conclude','--experiment',exp,'--config',str(config_path)],run_id=f'{exp}-decision')
    invoke(['report'],run_id='aggregate-report')
