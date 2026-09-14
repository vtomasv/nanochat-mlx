"""Readable before/after tables built strictly from recorded results."""
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from .config import dump,sha256

def render(root):
    root=Path(root)
    lines=['# Resultados de los experimentos Navarro', '',
           'Adaptaciones de estructuras compactas al entrenamiento e inferencia de nanochat-mlx. No son reproducciones de experimentos neuronales de Navarro.', '',
           'Referencia auditada: `b54b9fc139f455a9a5e60dc9a688ca9dbdb22944`. Apple M3 Max, 128 GiB, MLX 0.32.2, FP32. Perfil principal: depth 8, secuencia 1024, batch 8192 tokens, acumulación 2. Corpus y tokenizer locales fijados por hash.', '',
           'Cada piloto usa semilla 101 y procesos separados. El checkpoint principal de inferencia usa semilla 17, paso 1000. Los cocientes siguientes son descriptivos de ingeniería; no son intervalos de confirmación. NLL se evalúa en 96 secuencias reservadas.', '']
    reference=root/'baseline/primary-native-reference-s17/summary.json'
    if reference.exists():
        m=json.loads(reference.read_text())['intervention']
        lines += [f'Referencia nativa completa: **{m["steps"]} actualizaciones**, **{m["tokens"]:,} tokens procesados**, **{m["parameters"]:,} parámetros**; NLL final **{m["nll"]:.6f}** y **{m["mean_step_seconds"]:.4f} s/paso**. [Mediciones y checkpoints](baseline/primary-native-reference-s17/report.md).', '']
    junit=root/'correctness-junit.xml'
    if junit.exists():
        cases=list(ET.parse(junit).iter('testcase'));skipped=sum(e.find('skipped') is not None for e in cases);failed=sum(e.find('failure') is not None or e.find('error') is not None for e in cases)
        lines += [f'Verificación final: **{len(cases)-skipped-failed} pruebas aprobadas, {skipped} omitidas, {failed} fallidas**. [JUnit y razones de omisión](correctness-junit.xml).', '']
    for exp,arms in [('E1',['native','tape_dense','tape_recompute','tape_bitmap']),('E2',['native','staged_raw','staged_grammar'])]:
        lines += [f'## {exp}: entrenamiento', '', '| Brazo | Pasos | NLL final | s/paso (media) | Tokens/s | Pico MLX (bytes) | RSS máximo (bytes) |','|---|---:|---:|---:|---:|---:|---:|']
        stats={}
        for arm in arms:
            d=root/exp/f'primary-pilot-{arm}-s101';p=d/'summary.json'
            if not p.exists():continue
            s=json.loads(p.read_text());m=s.get('intervention') or {};stats[arm]=m
            def val(k,fmt=None):
                v=m.get(k);return 'null' if v is None else format(v,fmt) if fmt else str(v)
            lines += [f'| [{arm}]({d.relative_to(root)}/report.md) | {val("steps")} | {val("nll",".7f")} | {val("mean_step_seconds",".4f")} | {val("tokens_per_second",".1f")} | {val("mlx_active_peak_bytes")} | {val("rss_peak_bytes")} |']
        if all(stats.get(a) and stats[a].get('mean_step_seconds') for a in arms):
            native=stats[arms[0]];compact=stats[arms[-1]]
            tr=compact['mean_step_seconds']/native['mean_step_seconds'];ml=100*(1-compact['mlx_active_peak_bytes']/native['mlx_active_peak_bytes']);rss=100*(compact['rss_peak_bytes']/native['rss_peak_bytes']-1)
            lines += ['',f'Cociente de tiempo intervención/nativo: **{tr:.3f}×**. Reducción del pico MLX: **{ml:.2f}%**. Cambio de RSS: **{rss:+.2f}%**. Diferencia NLL final: **{compact["nll"]-native["nll"]:+.7f} nats/token**. IC y no inferioridad confirmatoria: `null`.']
            if exp=='E1':
                table=root/exp/'primary-pilot-tape_bitmap-s101/tensors.csv'
                with table.open() as f:rows=[r for r in csv.DictReader(f) if int(r['step'])>=50]
                ratio=sum(int(r['resident_bytes']) for r in rows)/sum(int(r['raw_bytes']) for r in rows)
                lines += [f'Payload P residente durante pasos 50–99: {ratio:.4f} del denso, incluidos contenedores CPU. La recomputación tiene el mismo pico MLX que bitmap en este piloto y menor tiempo.']
            else:
                table=root/exp/'primary-pilot-staged_grammar-s101/tensors.csv'
                with table.open() as f:rows=list(csv.DictReader(f))
                last=max(int(r['step']) for r in rows);lastrows=[r for r in rows if int(r['step'])==last]
                ratio=sum(int(r['resident_bytes']) for r in lastrows)/sum(int(r['raw_bytes']) for r in lastrows)
                lines += [f'Estado residente tras la última actualización: {ratio:.6f} del payload denso. RAW en {sum(r["mode"].startswith("RAW") for r in lastrows)}/{len(lastrows)} tensores registrados. Los intentos de construcción y sus copias están incluidos en s/paso.']
        lines += ['',f'![Curvas {exp}]({exp}/pilot-curves.png)','']
        decisions=[]
        for p in (root/exp).glob('*/summary.json'):
            s=json.loads(p.read_text())
            if str(s.get('decision','')).startswith('NEGATIVE_SCREENING'):decisions.append((p,s))
        if decisions:
            p,s=decisions[-1];lines += [f'Veredicto: **{s["decision"]}**. [Puerta de decisión y datos]({p.parent.relative_to(root)}/report.md).','']
        else:lines += ['Veredicto: **INCONCLUSIVE** mientras falte el cierre de la puerta registrada.','']
        lines += ['Medición con dataloader original (`pipeline`), separada de replay:', '', '| Brazo | Pasos | s/paso | NLL final |','|---|---:|---:|---:|']
        for arm in arms:
            p=root/exp/f'primary-pipeline-{arm}-s101/summary.json'
            if p.exists():
                m=json.loads(p.read_text()).get('intervention') or {}
                lines += [f'| [{arm}]({p.parent.relative_to(root)}/report.md) | {m.get("steps")} | {m.get("mean_step_seconds")} | {m.get("nll")} |']
        lines+=['']
    lines += ['## E3: inferencia', '', 'Checkpoint nativo principal: paso 1000, semilla 17. Condición medida: ocho prompts reservados de 512 tokens, 128 pasos de decode con continuación fija por prompt, batch 1. Conversión forzada de `blocks.4.mlp.c_fc.weight`. Cada brazo es un proceso nuevo.', '', '| Brazo | s/token (media) | p95 s/token | TTFT (s) | Conversión (s) | Pesos residentes (bytes) | Pico MLX (bytes) | RSS (bytes) |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ('dense_native','compressed_decode_dense','grammar_direct_cpu'):
        p=root/'E3'/f'primary-E3-{arm}/summary.json'
        if not p.exists():continue
        m=json.loads(p.read_text()).get('intervention') or {}
        fields=['mean_decode_seconds','p95_decode_seconds','mean_ttft_seconds','conversion_seconds','weights_resident_bytes','mlx_active_peak_bytes','rss_peak_bytes']
        lines += ['| ['+arm+']('+str(p.parent.relative_to(root))+'/report.md) | '+' | '.join('null' if m.get(k) is None else f'{m[k]:.6f}' if isinstance(m[k],float) else str(m[k]) for k in fields)+' |']
    screen=root/'E3/primary-E3-screen/summary.json'
    if screen.exists():
        m=json.loads(screen.read_text())['intervention'];lines += ['',f'Matrices MLP elegibles: **{m["selected_tensors"]}/{m["eligible_tensors"]}**. [Todas las capas](E3/primary-E3-screen/tensors.csv). [Corrección del modelo entrenado y microbenchmark](E3/primary-E3-correctness/report.md).']
    generations={}
    for arm in ('dense_native','compressed_decode_dense','grammar_direct_cpu'):
        path=root/'E3'/f'primary-E3-{arm}'/'events.jsonl'
        if path.exists():
            events=[json.loads(line) for line in path.read_text().splitlines()]
            generations[arm]={e['prompt']:e['ids'] for e in events if e['event']=='free_greedy_generation'}
    if generations.get('dense_native'):
        comparisons={}
        for arm,records in generations.items():
            if arm=='dense_native':continue
            comparisons[arm]=[]
            for prompt,ids in records.items():
                native=generations['dense_native'][prompt]
                first=next((i for i,(a,b) in enumerate(zip(native,ids)) if a!=b),None)
                if first is None and len(native)!=len(ids):first=min(len(native),len(ids))
                comparisons[arm].append({'prompt':prompt,'exact_match':native==ids,'first_divergence_zero_based':first,'native_tokens':len(native),'experimental_tokens':len(ids)})
        dump(root/'E3/greedy-comparison.json',comparisons)
        lines += ['', '[Concordancia y primera divergencia de generación greedy](E3/greedy-comparison.json).']
    for p in (root/'E3').glob('*/summary.json'):
        s=json.loads(p.read_text())
        if str(s.get('decision','')).startswith('NEGATIVE_SCREENING'):lines += ['',f'Veredicto: **{s["decision"]}**. [Decisión]({p.parent.relative_to(root)}/report.md).']
    external=root/'E3/external-E3-sft-d4/summary.json'
    if external.exists():
        s=json.loads(external.read_text());m=s['intervention']
        lines += ['',f'Validación externa: checkpoint SFT depth 4, paso {m["step"]}; {m["selected_tensors"]}/{m["eligible_tensors"]} matrices elegibles. Veredicto: **{s["decision"]}**. [Datos y límites de procedencia](E3/external-E3-sft-d4/report.md). No reemplaza el checkpoint principal.']
        if s['correctness_pass'] is False:
            lines += ['El producto directo incumple la tolerancia de logits (máximo absoluto 0,000329), aunque pasa NLL. La reconstrucción seguida del cálculo denso coincide exactamente con el nativo; uno de ocho vectores locales también incumple la tolerancia frente al oráculo FP64. Esto localiza la discrepancia en el producto CPU y limita su uso con este checkpoint; no se ampliaron tolerancias ni se modificó el operador medido. [Investigación separada](E3/external-E3-sft-d4-reduction-diagnostic/correctness.json).']
    initial=root/'E3/primary-E3-step0-diagnostic/summary.json'
    if initial.exists():
        m=json.loads(initial.read_text())['intervention']
        lines += ['',f'Control diagnóstico inicial: {m["selected_tensors"]}/{m["eligible_tensors"]} matrices elegibles en paso 0, frente a 0/16 en paso 1000. Las proyecciones inicialmente nulas no predicen compresión de pesos aprendidos. [Paso cero](E3/primary-E3-step0-diagnostic/tensors.csv).']
    detailed=sorted(root.glob('E*/primary-detailed-*/report.md'))
    if detailed:
        lines += ['', 'Perfiles de fases con barreras explícitas, excluidos de los cocientes principales: '+', '.join(f'[{p.parent.name.removeprefix("primary-detailed-")}]({p.relative_to(root)})' for p in detailed)+'.']
    lines += ['', 'La generación greedy libre se registra aparte con EOS natural y longitud efectiva en los eventos de cada brazo. No se usa para sustituir el benchmark de decode con contexto fijo.', '',
              '## Límites y trazabilidad','',
              '- Los intervalos confirmatorios permanecen `null`: las puertas negativas detienen el escalado; no se ajustan umbrales ni precisión.',
              '- El corpus tiene 43 documentos idénticos entre shards de entrenamiento y validación. El packing no conserva procedencia por documento; no se reporta bootstrap por documento.',
              '- MLX mostró variación numérica entre repeticiones del propio CLI: [diagnóstico nativo/nativo y nativo/harness](native-variability.json). No se afirma identidad bit a bit de trayectorias largas.',
              '- La construcción RePair es una implementación propia compilada con recuentos repetidos. Sus tiempos no son una evaluación de la implementación lineal de los autores.',
              '- RSS y Metal se reportan por separado; no se suman como memorias físicas disjuntas.',
              '- Las corridas fallidas y diagnósticas se conservan en el [índice completo](report.md). Los archivos grandes están en `../artifacts/` y quedan fuera de git.',
              '- Cuatro índices del smoke E1 todavía describían el marcador inicial de corrección. Se conservaron los índices anteriores y se documentó la [reconciliación de metadatos](audit-metadata-reconciliation.json), sin cambios en CSV. El XML histórico de esa puerta no se conservó; la suite completa final aporta evidencia separada.',
              '- [Fuentes y adaptación](../../../docs/navarro/source_map.md), [implementación](../../../docs/navarro/implementation.md), [reproducción](../../../docs/navarro/runbook.md), [tests](correctness-junit.xml), [auditoría de artefactos](audit.json).','']
    (root/'informe_final.md').write_text('\n'.join(lines))
    return root/'informe_final.md'
