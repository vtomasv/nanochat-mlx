"""Rebuild human-readable notebook and paper tables strictly from sealed results."""
import csv
import json
from pathlib import Path

from .records import digest,ledger_read,write_json


def verified_trials(root):
    records=ledger_read(root/'ledger.jsonl');results=[]
    for event in records:
        if event['event']!='trial_finished':continue
        trial=root/'trials'/event['trial']
        if digest(trial/'summary.json')!=event['summary_sha256'] or digest(trial/'artifacts.json')!=event['artifacts_sha256']:
            raise ValueError('Trial evidence was modified: '+event['trial'])
        artifacts=json.loads((trial/'artifacts.json').read_text())
        for name,h in artifacts.items():
            if digest(trial/name)!=h:raise ValueError('Trial artifact changed: '+str(trial/name))
        result=json.loads((trial/'summary.json').read_text());result['trial_id']=event['trial']
        result['request']=json.loads((trial/'request.json').read_text());results.append(result)
    return results


def decision(result,native,dense=None,recompute=None):
    if not result.get('correctness_pass'):return 'FAILURE_CORRECTNESS_OR_RUN'
    if result.get('request',{}).get('quality_split')=='reserved' or result['trial_id'].startswith('confirm-'):return 'SEALED_CONFIRMATION_ONLY'
    if result['trial_id']=='reference':return 'REFERENCE_ONLY'
    if result['scenario']['arm'] in ('native','tape_dense','tape_recompute','checkpoint_native'):return 'CONTROL'
    if not native or not result.get('metrics'):return 'INCONCLUSIVE_NO_MATCHED_CONTROL'
    a=result['metrics'];b=native['metrics']
    if result['start_step']!=native['start_step'] or result['steps']!=native['steps']:return 'INCONCLUSIVE_UNMATCHED_WORK'
    if a.get('payload_ratio') is not None and a['payload_ratio']>.9:return 'NEGATIVE_SCREENING_PAYLOAD'
    if a['mean_step_seconds']/b['mean_step_seconds']>2:return 'NEGATIVE_SCREENING_TIME'
    # Discovery shortlist, NOT statistical success or no-inferiority evidence.
    quality=abs(result['quality']['nll']-native['quality']['nll'])<.01
    practical=a['mean_step_seconds']<=1.10*b['mean_step_seconds'] and a['mlx_active_peak_bytes']<=.9*b['mlx_active_peak_bytes'] and a['rss_peak_bytes']<=1.05*b['rss_peak_bytes']
    incremental=dense is not None and a['mlx_active_peak_bytes']<=.95*dense['metrics']['mlx_active_peak_bytes']
    dominated=recompute is not None and recompute['metrics']['mean_step_seconds']<=a['mean_step_seconds'] and recompute['metrics']['mlx_active_peak_bytes']<=a['mlx_active_peak_bytes']
    if quality and practical and incremental and not dominated:return 'CANDIDATE_FOR_CONFIRMATION'
    return 'INCONCLUSIVE_PILOT_NOT_SHORTLISTED'


def report(root):
    root=Path(root);trials=verified_trials(root)
    by_id={r['trial_id']:r for r in trials};native=by_id.get('native');rows=[]
    for result in trials:
        metrics=result.get('metrics') or {};quality=result.get('quality') or {}
        dec=decision(result,native,by_id.get('tape-dense'),by_id.get('tape-recompute'))
        row={'trial':result['trial_id'],'arm':result['scenario']['arm'],'decision':dec,'steps':result.get('steps'),
            'quality_split':quality.get('split'),'calibration_nll':quality.get('nll') if quality.get('split')=='calibration' else None,'reserved_nll':quality.get('nll') if quality.get('split')=='reserved' else None,'mean_step_seconds':metrics.get('mean_step_seconds'),
            'mlx_peak_bytes':metrics.get('mlx_active_peak_bytes'),'rss_peak_bytes':metrics.get('rss_peak_bytes'),
            'payload_ratio':metrics.get('payload_ratio'),'source_sha256':result['request']['sources']['tree_sha256'],
            'candidate_sha256':result['request']['candidate_sha256'],'hypothesis':result['request']['hypothesis']}
        if native and row['mean_step_seconds'] is not None and result.get('start_step')==native.get('start_step') and result.get('steps')==native.get('steps') and not result['trial_id'].startswith('confirm-'):
            row['time_ratio_vs_native']=row['mean_step_seconds']/native['metrics']['mean_step_seconds']
        else:row['time_ratio_vs_native']=None
        rows.append(row)
    fields=['trial','arm','decision','steps','quality_split','calibration_nll','reserved_nll','mean_step_seconds','time_ratio_vs_native','mlx_peak_bytes','rss_peak_bytes','payload_ratio','source_sha256','candidate_sha256','hypothesis']
    with (root/'results.tsv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,delimiter='\t');writer.writeheader();writer.writerows(rows)
    lines=['# Campaña Navarro v2: cuaderno de autoresearch','',
        '**Estado de evidencia: exploración de ingeniería. No demuestra todavía no inferioridad ni mejora confirmatoria.**','',
        'Cada intento conserva código, hipótesis, controles, errores de gradientes, métricas y eventos. Los descartes no borran artefactos. El test reservado no se usa para seleccionar candidatos.','',
        '| Ensayo | Decisión | Pasos | NLL calibración | s/paso | Tiempo/nativo | Pico MLX GiB | RSS GiB | P/denso |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(v):return 'null' if v is None else f'{v:.6f}'
    for r in rows:
        lines.append(f"| [{r['trial']}](trials/{r['trial']}/summary.json) | {r['decision']} | {r['steps']} | {fmt(r['calibration_nll'])} | {fmt(r['mean_step_seconds'])} | {fmt(r['time_ratio_vs_native'])} | {fmt(r['mlx_peak_bytes']/1024**3 if r['mlx_peak_bytes'] else None)} | {fmt(r['rss_peak_bytes']/1024**3 if r['rss_peak_bytes'] else None)} | {fmt(r['payload_ratio'])} |")
    screening=root/'screening/summary.json'
    if screening.exists():
        events=[e for e in ledger_read(root/'ledger.jsonl') if e['event']=='structural_screening']
        if not events or digest(screening)!=events[-1]['summary_sha256']:raise ValueError('Screening evidence modified')
        s=json.loads(screening.read_text())
        for entry in s['checkpoint_census']:
            if digest(root/'screening'/entry['csv'])!=entry['sha256']:raise ValueError('Census CSV modified')
        if digest(root/'screening/activations.json')!=s['activation_sha256']:raise ValueError('Activation evidence modified')
        lines+=['','## Cotas de diccionario en checkpoints entrenados','',
            '| Paso | Objeto | Filas/bloque | Bloques descartados | Ahorro máximo bajo selector RAW |','|---:|---|---:|---:|---:|']
        for r in s['checkpoint_census']:
            lines.append(f"| {r['step']} | {r['file']} | {r['rows_per_block']} | {r['rejected_blocks']}/{r['blocks']} | {100*r['maximum_savings_under_policy']:.4f}% |")
        lines+=['',f"Activaciones: {s['activation_samples']} muestras P/H, {s['activation_rules']} reglas; roundtrip exacto: {s['activation_roundtrip_pass']}. [Datos](screening/activations.json).",
            'Las cotas solo afectan diccionarios locales del tamaño indicado. Bloques no descartados no se declaran compresibles.']
    lines+=['','## Límites y próximos pasos','',
        '- Una réplica de proceso por escenario: no hay IC confirmatorios ni comparación de convergencia.',
        '- NLL de calibración ponderada por tokens; el conjunto reservado queda fuera de búsqueda.',
        '- MLX y RSS se solapan; no sumarlos como memoria física.',
        '- La calidad al final de una ventana no demuestra estabilidad de una trayectoria completa.',
        '- H directa, diccionario global, KV, cuantización y adaptación congelada son ramas separadas; no se presentan como ejecutadas.',
        '- Toda corrección de implementación exige nuevos snapshots; no reescribe resultados anteriores.','',
        'Fuentes: [protocolo](../../../../ESPECIFICACIONES_NAVARRO_NANOCHAT_MLX.md), [autoresearch](https://github.com/karpathy/autoresearch), [N3](https://users.dcc.uchile.cl/~gnavarro/ps/pvldb22.pdf).']
    if (root/'confirmation.json').exists():
        confirmation=json.loads((root/'confirmation.json').read_text())
        lines+=['','## Confirmación sellada','',confirmation['decision'],'', '[Estimaciones y supuestos](confirmation.json).']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    write_json(root/'decisions.json',{'evidence':'engineering_only','trials':rows,'shortlist':[r['trial'] for r in rows if r['decision']=='CANDIDATE_FOR_CONFIRMATION']},exclusive=False)
    paper=root/'paper';paper.mkdir(exist_ok=True)
    (paper/'results.md').write_text('# Resultados descriptivos para el manuscrito\n\n'+ '\n'.join(lines[6:]).replace('](trials/','](../trials/').replace('](screening/','](../screening/').replace('](../../../../ESPECIFICACIONES','](../../../../../ESPECIFICACIONES').replace('](confirmation.json)','](../confirmation.json)')+'\n')
    (paper/'outline.md').write_text('''# Esqueleto de artículo: estructuras compactas en un GPT sobre Apple Silicon

1. Pregunta y contribuciones: distinguir representación, operador y entrenamiento completo.
2. Antecedentes: CSRV/RePair de Ferragina et al.; bitmaps/rank de Navarro y colaboradores.
3. Modelo de costo: diccionario, metadatos, vida útil, reconstrucción y tráfico.
4. Sistema: nanochat-mlx, Muon/AdamW, ReLU², kernels Metal, controles de recomputación.
5. Método: datos por documento, preregistro, búsqueda separada de confirmación, exactitud IEEE y VJP.
6. Resultados: incluir intentos fallidos, cotas, curvas y todos los controles.
7. Amenazas a validez: hardware único, aprendizaje temprano, procedencia tokenizer, duplicados cercanos.
8. Conclusión: afirmar exclusivamente lo respaldado por experimentos terminados.

Este archivo es una estructura de trabajo. No es un paper concluido ni una afirmación de mejoras estadísticamente demostradas.
''')
    return root/'report.md'
