# E3: INCONCLUSIVE

Brazo: `compressed_decode_dense`. Estado: `measured`. SHA: `b54b9fc139f455a9a5e60dc9a688ca9dbdb22944`. Hardware: Apple M3 Max; FP32; perfil primary; semilla 17; modalidad replay.

Corpus: `/Users/tomasvera/.cache/nanochat`. Artefactos/checkpoints: `/Users/tomasvera/dev/nanochat-mlx/experiments/navarro/artifacts/primary`. Hashes: [manifest](manifest.json).

| Medida | Nativo | Control | Intervención |
|---|---:|---:|---:|
| Parámetros | null | null | null |
| Tokens procesados | null | null | null |
| NLL final | null | null | null |
| Tiempo completo/paso o solicitud (s) | null | null | 0.902474 |
| Tokens/s | null | null | 145.1088 |
| Pico MLX (bytes) | null | null | 843637372 |
| RSS máximo (bytes) | null | null | 681590784 |
| Pesos residentes (bytes) | null | null | 506602389 |
| Conversión (s) | null | null | 0.08634804 |

Corrección: `None`. Fidelidad: `True`. IC confirmatorios: `None`. No inferioridad confirmatoria: `None`.

Motivo: None.

- Forced central c_fc, CPU transfers included.
- Teacher-forced timing; separate free greedy generations with natural EOS are in events.jsonl.
- One pilot process; no process-paired CI.

[Pasos crudos](steps.csv) · [Tensores](tensors.csv) · [Eventos y errores](events.jsonl) · [Corrección](correctness.json) · [Resumen completo](summary.json) · [Comandos](commands.sh) · [Fuentes](source_map.md)
