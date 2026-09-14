# E1: NEGATIVE_SCREENING

Brazo: `decision`. Estado: `NEGATIVE_SCREENING`. SHA: `b54b9fc139f455a9a5e60dc9a688ca9dbdb22944`. Hardware: Apple M3 Max; FP32; perfil primary; semilla 101; modalidad replay.

Corpus: `/Users/tomasvera/.cache/nanochat`. Artefactos/checkpoints: `/Users/tomasvera/dev/nanochat-mlx/experiments/navarro/artifacts/primary`. Hashes: [manifest](manifest.json).

| Medida | Nativo | Control | Intervención |
|---|---:|---:|---:|
| Parámetros | 125829648 | 125829648 | 125829648 |
| Tokens procesados | 819200 | 819200 | 819200 |
| NLL final | 6.630266 | 6.630269 | 6.630293 |
| Tiempo completo/paso o solicitud (s) | 0.6228055 | 0.803944 | 2.009751 |
| Tokens/s | 13153.38 | 10189.76 | 4076.128 |
| Pico MLX (bytes) | 6161390588 | 5570193744 | 5301758288 |
| RSS máximo (bytes) | 1550008320 | 1549893632 | 2275508224 |
| Pesos residentes (bytes) | null | null | null |
| Conversión (s) | null | null | null |

Corrección: `True`. Fidelidad: `True`. IC confirmatorios: `None`. No inferioridad confirmatoria: `None`.

Motivo: Preregistered engineering gate failed.

- Engineering seed101, 100 full updates per arm; no confirmatory CI.
- Negative conclusion applies to this implementation, early training checkpoint family and M3 Max.
- CPU residency and copies count toward RSS; MLX allocator reduction alone is not physical-memory reduction.

[Pasos crudos](steps.csv) · [Tensores](tensors.csv) · [Eventos y errores](events.jsonl) · [Corrección](correctness.json) · [Resumen completo](summary.json) · [Comandos](commands.sh) · [Fuentes](source_map.md)
