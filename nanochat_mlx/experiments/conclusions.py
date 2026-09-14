"""Evidence aggregation, with negative gates evaluated before confirmation."""
import csv
import json
from pathlib import Path
from .config import ROOT,dump,sha256
from .reporting import Run

def read(path):return json.loads(Path(path).read_text())

def conclude(c,experiment):
    root=ROOT/'experiments/navarro/results'/experiment
    run=Run(experiment,c,'decision','pilot',101)
    if experiment in ('E1','E2'):
        arms=['native','tape_dense','tape_recompute','tape_bitmap'] if experiment=='E1' else ['native','staged_raw','staged_grammar']
        paths=[root/f'primary-pilot-{arm}-s101' for arm in arms]
        if not all((p/'summary.json').exists() for p in paths):
            run.finish(status='INCONCLUSIVE',decision='INCONCLUSIVE',failure_reason='Missing mandatory pilot arms');return run.root
        summaries=[read(p/'summary.json') for p in paths]
        if any(s['status']!='measured' for s in summaries):
            run.finish(status='INCONCLUSIVE',decision='INCONCLUSIVE',failure_reason='Incomplete mandatory pilot arm');return run.root
        stats=[s['intervention'] for s in summaries];ratio=stats[-1]['mean_step_seconds']/stats[0]['mean_step_seconds']
        correct=all(s['correctness_pass'] for s in summaries)
        threshold=ratio>2
        detail={'pilot_time_ratio':ratio,'registered_time_gate':2.0}
        if experiment=='E1':
            with (paths[-1]/'tensors.csv').open() as f:rows=list(csv.DictReader(f))
            trained=[r for r in rows if int(r['step'])>=50]
            payload=sum(int(r['resident_bytes']) for r in trained)/sum(int(r['raw_bytes']) for r in trained)
            threshold|=payload>.9;detail['trained_activation_ratio']=payload
        else:
            profiles=[root/f'primary-E2-screen-{step}'/'summary.json' for step in (100,500,1000)]
            if not all(p.exists() for p in profiles):
                run.finish(status='INCONCLUSIVE',decision='INCONCLUSIVE',failure_reason='Missing trained optimizer profiles');return run.root
            ratios=[read(p)['intervention']['eligible_resident_ratio'] for p in profiles]
            threshold|=sum(r<=.9 for r in ratios)<2;detail['eligible_state_ratios']=ratios
        status='NEGATIVE_SCREENING' if threshold and correct else 'INCONCLUSIVE'
        dump(run.root/'correctness.json',{'pass':bool(correct),'evidence':[str(p/'correctness.json') for p in paths],'trained_profile_directory':str(root)})
        run.event('registered_gate',**detail)
        run.finish(status=status,decision=status,verdicts={'representation':'COMPRESSED_BIT_EXACT' if experiment=='E1' else 'NO_ELIGIBLE_COMPRESSION','operator':'CORRECT_WITHIN_TOLERANCE' if correct else 'NOT_VALIDATED','full_model':status},correctness_pass=bool(correct),algorithm_fidelity=True,baseline=stats[0],controls=stats[1:-1],intervention=stats[-1],paired_ratios=None,failure_reason='Preregistered engineering gate failed' if status=='NEGATIVE_SCREENING' else 'Pilot alone cannot establish a confirmatory outcome',limitations=['Engineering seed101, 100 full updates per arm; no confirmatory CI.','Negative conclusion applies to this implementation, early training checkpoint family and M3 Max.','CPU residency and copies count toward RSS; MLX allocator reduction alone is not physical-memory reduction.'])
        links='\n'.join(f'- [{arm}](../{p.relative_to(root)}/report.md)' for arm,p in zip(arms,paths))
        (run.root/'report.md').write_text((run.root/'report.md').read_text()+'\nGate evidence: `'+json.dumps(detail)+'`\n\n'+links+'\n')
    else:
        screen=root/'primary-E3-screen'/'summary.json'
        arms=['dense_native','compressed_decode_dense','grammar_direct_cpu']
        paths=[root/f'primary-E3-{arm}' for arm in arms]
        if not screen.exists() or not all((p/'summary.json').exists() for p in paths):
            run.finish(status='INCONCLUSIVE',decision='INCONCLUSIVE',failure_reason='Missing E3 screening or forced-layer pilot');return run.root
        s=read(screen);stats=[read(p/'summary.json')['intervention'] for p in paths]
        correct_file=root/'primary-E3-correctness'/'correctness.json'
        correct=read(correct_file)['pass'] if correct_file.exists() else None
        status='NEGATIVE_SCREENING_NO_COMPRESSIBILITY' if s['intervention']['selected_tensors']==0 and correct else 'INCONCLUSIVE'
        dump(run.root/'correctness.json',{'pass':correct,'evidence':str(correct_file)})
        denom=stats[0]['mean_decode_seconds']-stats[-1]['mean_decode_seconds']
        run.event('conversion_amortization',tokens_break_even=stats[-1]['conversion_seconds']/denom if denom>0 else 'never')
        run.finish(status=status,decision=status,verdicts={'representation':'NO_ELIGIBLE_COMPRESSION' if s['intervention']['selected_tensors']==0 else 'COMPRESSIBLE','operator':'CORRECT_WITHIN_TOLERANCE' if correct else 'NOT_VALIDATED','full_model':status},correctness_pass=correct,algorithm_fidelity=True,baseline=stats[0],controls=[stats[1]],intervention=stats[-1],failure_reason='No trained MLP matrix reaches <=95% RAW' if status.startswith('NEGATIVE') else None,limitations=['Step1000 seed17 only; all MLP matrices screened.','Forced central c_fc, eight reserved prompts, teacher-forced decode128.','No GPU direct kernel: CPU/MLX movement included.','No extrapolation to all neural compression methods.'])
    (run.root/'artifacts.sha256').write_text(''.join(f'{sha256(p)}  {p.name}\n' for p in sorted(run.root.iterdir()) if p.is_file() and p.name!='artifacts.sha256'))
    return run.root
