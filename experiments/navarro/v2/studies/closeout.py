"""CPU-only closeout: verify evidence, derive matched ratios, preserve provenance.

Does not train, open reserved arrays, change thresholds, or rewrite study inputs.
Outputs are exclusive. Use a fresh --output for a new audit.
"""
import argparse
import csv
import json
from pathlib import Path

from nanochat_mlx.experiments.v2.records import digest, now, write_json, verify_sources, ledger_read
from nanochat_mlx.experiments.v2.reporting import verified_trials


def verify_artifacts(folder):
    manifest = folder / 'artifacts.json'
    files = json.loads(manifest.read_text())
    for name, expected in files.items():
        path = folder / name
        if not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError('Artifact escapes its study')
        if digest(path) != expected:
            raise ValueError(f'Changed evidence: {path}')
    return {'manifest_sha256': digest(manifest), 'verified_files': len(files)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--campaign', type=Path, default=Path('experiments/navarro/v2/20260915'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.campaign, args.output
    manifest = json.loads((root / 'manifest.json').read_text())
    verify_sources(manifest['protected_sources'])
    trials = verified_trials(root)
    ledger = ledger_read(root / 'ledger.jsonl')
    studies = {}
    sources = {'mechanisms': 'direct_activation.py', 'dictionary-extension': 'dictionary_bounds.py',
               'direct-diagnosis': 'diagnose_direct.py', 'gradient-census': 'gradient_census.py',
               'combined-gradients': 'source.py', 'combined-gradients-r2': 'source.py'}
    for name, source in sources.items():
        folder = root / name
        studies[name] = verify_artifacts(folder)
        plan = json.loads((folder / 'plan.json').read_text())
        source_path = folder / source if source == 'source.py' else root.parent / 'studies' / source
        if digest(source_path) != plan['source_sha256']:
            raise ValueError(f'Study source mismatch: {name}')
        if 'campaign_manifest_sha256' in plan and plan['campaign_manifest_sha256'] != digest(root / 'manifest.json'):
            raise ValueError('Parent manifest mismatch')
        events = [e for e in ledger if e.get('study') == name]
        if not events or events[-1]['artifacts_sha256'] != studies[name]['manifest_sha256']:
            raise ValueError(f'Study missing or changed in ledger: {name}')
    combined = root / 'combined-gradients-r2'
    results = {r['trial']: r for r in json.loads((combined / 'results.json').read_text())}
    plan = json.loads((combined / 'plan.json').read_text())
    if plan['parent_manifest_sha256'] != digest(root / 'manifest.json'):
        raise ValueError('Extension parent changed')
    if plan['candidate_sha256'] != digest(combined / 'candidate.py'):
        raise ValueError('Extension candidate changed')
    if set(results) != {s['id'] for s in plan['scenarios']}:
        raise ValueError('Incomplete scenario set')
    native, dense, recompute = [results[k]['metrics'] for k in ('native', 'tape-dense', 'tape-recompute')]
    decisions = {r['trial']: r for r in json.loads((combined / 'decisions.json').read_text())['candidates']}
    rows, fingerprints, storage, checks = [], {}, {}, {}
    for scenario in plan['scenarios']:
        name = scenario['id']
        folder, result = combined / name, results[name]
        checks[name] = verify_artifacts(folder)
        original = json.loads((folder / 'summary.json').read_text())
        if {k: v for k, v in result.items() if k != 'trial'} != original:
            raise ValueError('Aggregate diverges from trial')
        request = json.loads((folder / 'request.json').read_text())
        if request['scenario'] != scenario or request['plan_sha256'] != digest(combined / 'plan.json'):
            raise ValueError('Request mismatch')
        if result['status'] != 'MEASURED_EXPLORATORY' or not result['correctness_pass']:
            raise ValueError('Incomplete or failed trial; do not manufacture complete table')
        if result['start_step'] != 100 or result['steps'] != 100 or result['quality']['split'] != 'calibration':
            raise ValueError('Unmatched workload or evaluation split')
        metrics = result['metrics']
        row = {'trial': name, 'mean_step_seconds': metrics['mean_step_seconds'],
               'mlx_peak_gib': metrics['mlx_active_peak_bytes'] / 2**30,
               'time_ratio_vs_native': metrics['mean_step_seconds'] / native['mean_step_seconds'],
               'memory_savings_vs_native': 1 - metrics['mlx_active_peak_bytes'] / native['mlx_active_peak_bytes'],
               'memory_savings_vs_dense': 1 - metrics['mlx_active_peak_bytes'] / dense['mlx_active_peak_bytes'],
               'calibration_nll': result['quality']['nll'], 'decision': decisions.get(name, {}).get('decision', 'CONTROL')}
        if name in decisions:
            conditions = {
                'memory_vs_native': metrics['mlx_active_peak_bytes'] <= .9 * native['mlx_active_peak_bytes'],
                'memory_vs_tape_dense': metrics['mlx_active_peak_bytes'] <= .95 * dense['mlx_active_peak_bytes'],
                'time_vs_native': metrics['mean_step_seconds'] <= 1.1 * native['mean_step_seconds'],
                'rss_vs_native': metrics['rss_peak_bytes'] <= 1.05 * native['rss_peak_bytes'],
                'calibration_delta': abs(result['quality']['nll'] - results['native']['quality']['nll']) < .01,
                'not_dominated_by_recompute': not (recompute['mlx_active_peak_bytes'] <= metrics['mlx_active_peak_bytes'] and recompute['mean_step_seconds'] <= metrics['mean_step_seconds'])}
            if conditions != decisions[name]['conditions'] or all(conditions.values()):
                raise ValueError('Recorded negative decision does not recompute')
            baseline = 'bitmap' if name == 'rows-bitmap' else 'tape-dense'
            a = json.loads((folder / 'final-state-fingerprints.json').read_text())
            b = json.loads((combined / baseline / 'final-state-fingerprints.json').read_text())
            fingerprints[name] = {'control': baseline, 'groups': {}}
            for group, tensors in a.items():
                if tensors.keys() != b[group].keys():
                    raise ValueError('Fingerprint tree mismatch')
                fingerprints[name]['groups'][group] = {'equal_hashes': sum(h == b[group][k] for k, h in tensors.items()), 'tensors': len(tensors)}
            accounting = [json.loads(line) for line in (folder / 'gradient-storage.jsonl').read_text().splitlines()]
            if len(accounting) != 100:
                raise ValueError('Missing accumulator accounting')
            storage[name] = {str(i): sum(v['microbatches'][i]['stored_bytes'] / v['microbatches'][i]['raw_bytes'] for v in accounting) / len(accounting) for i in (0, 1)}
        rows.append(row)
    if json.loads((root / 'decisions.json').read_text())['shortlist']:
        raise ValueError('Primary shortlist is no longer empty')
    if (root / 'sealed-plan.json').exists() or (root / 'confirmation.json').exists():
        raise ValueError('Cannot describe this campaign as unconfirmed')
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'derived.json', {'rows': rows, 'fingerprint_comparison': fingerprints,
               'accumulator_stored_ratio_mean': storage,
               'fingerprint_limit': 'Hashes differ after 100 updates; no bit-identical trajectory or full-size final-state tolerance assertion. Small ten-update tolerance test and full-model initial gradient gate passed.'})
    with (out / 'combined.tsv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader(); writer.writerows(rows)
    lines = ['# Extensión de gradientes: cierre exploratorio', '',
             'Seis procesos de 100 updates desde checkpoint 100. Orden aleatorizado antes de medir; una observación por escenario, sin IC confirmatorios.', '',
             '| Brazo | s/update | MLX GiB | Tiempo/nativo | Ahorro vs. tape denso | Decisión |',
             '|---|---:|---:|---:|---:|---|']
    for row in rows:
        lines.append(f"| {row['trial']} | {row['mean_step_seconds']:.6f} | {row['mlx_peak_gib']:.6f} | {row['time_ratio_vs_native']:.6f} | {100*row['memory_savings_vs_dense']:.3f}% | {row['decision']} |")
    lines += ['', 'Ambos candidatos pasan los criterios puntuales de memoria, RSS y calibración, pero fallan tiempo ≤1,10× nativo. No se preselecciona ninguno.', '',
              'Las filas compactas conservan 36,49% del acumulador denso tras el primer microbatch y 38,77% tras el segundo (medias de 100 updates). Estos ratios incluyen los otros gradientes densos y no son el ahorro de memoria global.', '',
              'Se comprueba en cada empaquetado que las filas omitidas contienen bits de cero positivo. Se restauran todos los gradientes antes del optimizador; se conservan momentos y weight decay. La prueba pequeña de diez updates compara pesos y estados por ruta del parámetro.', '',
              'Las huellas finales de los 100 updates difieren de los controles: 0/60 tensores de modelo, 8/24 estados Adam y 0/52 estados Muon tienen hash idéntico en ambas comparaciones. La representación exacta no demuestra una trayectoria flotante idéntica. No se conservaron checkpoints finales para calcular su distancia; NLL casi igual no reemplaza esa medición.', '',
              'El primer intento se detuvo antes de pilotos porque el verificador comparaba listas de estados por posición, con distinto orden de inserción. Se preserva su fuente y fallo. La revisión r2 compara estados por nombres; no cambia tolerancias.', '',
              'No mezclar este control nativo (0,816694 s/update) con el de la campaña principal (0,958646). La diferencia entre procesos obliga a mantener las comparaciones dentro de cada estudio. El bitmap aislado también cambia su posición temporal; sigue siendo evidencia exploratoria.', '']
    (out / 'report.md').write_text('\n'.join(lines))
    audit = {'at': now(), 'source_sha256': digest(Path(__file__)),
             'campaign_manifest_sha256': digest(root / 'manifest.json'), 'ledger_sha256': digest(root / 'ledger.jsonl'),
             'ledger_records': len(ledger), 'primary_trials_verified': len(trials),
             'protected_sources_verified': True, 'studies': studies, 'combined_trials': checks,
             'primary_shortlist_empty': True, 'combined_shortlist_empty': True,
             'reserved_evaluation_artifacts_present': False,
             'scope': 'File/hash, source, request and arithmetic verification; not a runtime proof that no code ever accessed reserved data.',
             'derived_outputs': {p.name: digest(p) for p in out.iterdir() if p.is_file()}}
    write_json(out / 'audit.json', audit)
    print(json.dumps({'output': str(out), 'primary_trials': len(trials), 'extension_trials': len(rows), 'studies': len(studies)}))


if __name__ == '__main__':
    main()
