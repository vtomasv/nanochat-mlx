"""Append-only raw records and conservative protocol decisions."""
import csv
import datetime
import json
from pathlib import Path
import shlex
import shutil
import sys
import zipfile
from .config import ROOT,dump,sha256
from .measure import environment

class Run:
    def __init__(self,experiment,c,arm,stage,seed,run_id=None):
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        self.root=ROOT/'experiments/navarro/results'/experiment/(run_id or f'{c["profile"]}-{stage}-{arm}-s{seed}-{stamp}')
        self.root.mkdir(parents=True,exist_ok=False)
        env=environment(); dump(self.root/'environment.json',env)
        hashes={str(p.relative_to(ROOT)):sha256(p) for folder in ('nanochat_mlx','scripts','configs/navarro') for p in (ROOT/folder).rglob('*') if p.suffix in ('.py','.cpp','.json')}
        dump(self.root/'manifest.json',{'config':c,'arm':arm,'stage':stage,'seed':seed,'initial_checkpoint':str(Path(c['artifact_dir'])/f'initial-s{seed}'),'input_checkpoint':c.get('input_checkpoint'),'code_hashes':hashes,'uv_lock_sha256':sha256(ROOT/'uv.lock'),'spec_sha256':sha256(ROOT/'ESPECIFICACIONES_NAVARRO_NANOCHAT_MLX.md'),'batch_tape_manifest':str(Path(c['artifact_dir'])/'tape.json'),'batch_tape_manifest_sha256':sha256(Path(c['artifact_dir'])/'tape.json') if (Path(c['artifact_dir'])/'tape.json').exists() else None})
        with zipfile.ZipFile(self.root/'source_snapshot.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
            for relative in hashes:archive.write(ROOT/relative,relative)
            archive.write(ROOT/'uv.lock','uv.lock')
            archive.write(ROOT/'ESPECIFICACIONES_NAVARRO_NANOCHAT_MLX.md','ESPECIFICACIONES_NAVARRO_NANOCHAT_MLX.md')
        source=ROOT/'docs/navarro/source_map.md'
        if source.exists(): shutil.copyfile(source,self.root/'source_map.md')
        (self.root/'commands.sh').write_text('#!/bin/sh\ncd '+shlex.quote(str(ROOT))+'\n'+shlex.join(['uv','run','--frozen','python','-m','scripts.navarro_experiments',*sys.argv[1:]])+'\n')
        self.summary={'experiment':experiment,'protocol_version':'1.0','status':'not_measured','repo_sha':env['repo_sha'],'hardware':env['device'],'profile':c['profile'],'correctness_pass':None,'baseline':None,'controls':[],'intervention':None,'paired_ratios':None,'confidence_intervals':None,'quality_noninferiority':None,'algorithm_fidelity':None,'decision':None,'failure_reason':None,'limitations':[],'raw_files':[]}
        dump(self.root/'correctness.json',{'status':'not_measured'})
        for name,fields in [('steps.csv',['step','loss','seconds','tokens','mlx_active_peak_bytes','mlx_active_bytes','mlx_cache_bytes','rss_peak_bytes']),('tensors.csv',['step','path','raw_bytes','resident_bytes','mode','encode_seconds','payload_bytes','container_overhead_bytes','blocks','raw_blocks','rules','symbols','values','height','dictionary_bytes','header_bytes'])]:
            with (self.root/name).open('w') as f: csv.DictWriter(f,fields).writeheader()
        (self.root/'events.jsonl').touch()
        self.finish()
    def event(self,kind,**data):
        with (self.root/'events.jsonl').open('a') as f: f.write(json.dumps({'event':kind,**data},allow_nan=False)+'\n')
    def row(self,name,data):
        path=self.root/name
        with path.open() as f: fields=next(csv.reader(f))
        with path.open('a') as f: csv.DictWriter(f,fields,extrasaction='ignore').writerow(data)
    def finish(self,**updates):
        self.summary.update(updates)
        self.summary['raw_files']=['steps.csv','tensors.csv','events.jsonl','correctness.json','environment.json','manifest.json']
        dump(self.root/'summary.json',self.summary)
        s=self.summary
        (self.root/'report.md').write_text(f'# {s["experiment"]}: {s["status"]}\n\nSHA: `{s["repo_sha"]}`. Profile: {s["profile"]}. FP32.\n\nDecision: {s["decision"]}. Reason: {s["failure_reason"]}.\n\n'+ '\n'.join(f'- {x}' for x in s['limitations'])+'\n\n[Raw steps](steps.csv) · [Tensor accounting](tensors.csv) · [Events](events.jsonl) · [Summary](summary.json) · [Reproduce](commands.sh)\n')
        (self.root/'artifacts.sha256').write_text(''.join(f'{sha256(p)}  {p.name}\n' for p in sorted(self.root.iterdir()) if p.is_file() and p.name!='artifacts.sha256'))


def render_existing_report(root):
    """Rebuild narrative from saved summary, without recomputing measurements."""
    root=Path(root)
    index=root/'artifacts.sha256'
    if index.exists():
        for line in index.read_text().splitlines():
            digest,name=line.split('  ',1)
            if sha256(root/name)!=digest:raise ValueError(f'Artifact changed before report rendering: {root/name}')
    s=json.loads((root/'summary.json').read_text());m=json.loads((root/'manifest.json').read_text())
    arm=m['arm'];base=s.get('baseline') or {};control=(s.get('controls') or [{}])[0];inter=s.get('intervention') or {}
    if arm in ('native','repo_native','dense_native') and not base:base=inter;inter={}
    time_key='mean_step_seconds' if any('mean_step_seconds' in x for x in (base,control,inter)) else 'mean_request_seconds'
    fields=[('Parámetros','parameters'),('Tokens procesados','tokens'),('NLL final','nll'),('Tiempo completo/paso o solicitud (s)',time_key),('Tokens/s','tokens_per_second'),('Pico MLX (bytes)','mlx_active_peak_bytes'),('RSS máximo (bytes)','rss_peak_bytes'),('Pesos residentes (bytes)','weights_resident_bytes'),('Conversión (s)','conversion_seconds')]
    lines=[f'# {s["experiment"]}: {s["decision"]}', '',f'Brazo: `{arm}`. Estado: `{s["status"]}`. SHA: `{s["repo_sha"]}`. Hardware: Apple M3 Max; FP32; perfil {s["profile"]}; semilla {m["seed"]}; modalidad {m["config"].get("measurement_mode","replay")}.', '',
           f'Corpus: `{m["config"]["data_dir"]}`. Artefactos/checkpoints: `{m["config"]["artifact_dir"]}`. Hashes: [manifest](manifest.json).','',
           '| Medida | Nativo | Control | Intervención |','|---|---:|---:|---:|']
    for label,key in fields:
        lines.append('| '+label+' | '+' | '.join('null' if d.get(key) is None else f'{d[key]:.7g}' if isinstance(d[key],float) else str(d[key]) for d in (base,control,inter))+' |')
    lines+=['',f'Corrección: `{s.get("correctness_pass")}`. Fidelidad: `{s.get("algorithm_fidelity")}`. IC confirmatorios: `{s.get("confidence_intervals")}`. No inferioridad confirmatoria: `{s.get("quality_noninferiority")}`.', '',f'Motivo: {s.get("failure_reason")}.','']
    lines += ['- '+x for x in s['limitations']]
    lines += ['', '[Pasos crudos](steps.csv) · [Tensores](tensors.csv) · [Eventos y errores](events.jsonl) · [Corrección](correctness.json) · [Resumen completo](summary.json) · [Comandos](commands.sh) · [Fuentes](source_map.md)','']
    (root/'report.md').write_text('\n'.join(lines))
    (root/'artifacts.sha256').write_text(''.join(f'{sha256(p)}  {p.name}\n' for p in sorted(root.iterdir()) if p.is_file() and p.name!='artifacts.sha256'))
