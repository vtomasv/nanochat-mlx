# E3: INCONCLUSIVE

Brazo: `dense_native`. Estado: `measured`. SHA: `b54b9fc139f455a9a5e60dc9a688ca9dbdb22944`. Hardware: Apple M3 Max; FP32; perfil primary; semilla 17; modalidad replay.

Corpus: `/Users/tomasvera/.cache/nanochat`. Artefactos/checkpoints: `/Users/tomasvera/dev/nanochat-mlx/experiments/navarro/artifacts/primary`. Hashes: [manifest](manifest.json).

| Medida | Nativo | Control | Intervención |
|---|---:|---:|---:|
| Parámetros | null | null | null |
| Tokens procesados | null | null | null |
| NLL final | null | null | null |
| Tiempo completo/paso o solicitud (s) | 0.2692683 | null | null |
| Tokens/s | 497.1536 | null | null |
| Pico MLX (bytes) | 843407996 | null | null |
| RSS máximo (bytes) | 626737152 | null | null |
| Pesos residentes (bytes) | 503318592 | null | null |
| Conversión (s) | 0 | null | null |

Corrección: `None`. Fidelidad: `True`. IC confirmatorios: `None`. No inferioridad confirmatoria: `None`.

Motivo: None.

- Forced central c_fc, CPU transfers included.
- Teacher-forced timing; separate free greedy generations with natural EOS are in events.jsonl.
- One pilot process; no process-paired CI.

[Pasos crudos](steps.csv) · [Tensores](tensors.csv) · [Eventos y errores](events.jsonl) · [Corrección](correctness.json) · [Resumen completo](summary.json) · [Comandos](commands.sh) · [Fuentes](source_map.md)
