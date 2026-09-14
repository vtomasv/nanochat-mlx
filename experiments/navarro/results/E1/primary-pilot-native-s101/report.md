# E1: INCONCLUSIVE

Brazo: `native`. Estado: `measured`. SHA: `b54b9fc139f455a9a5e60dc9a688ca9dbdb22944`. Hardware: Apple M3 Max; FP32; perfil primary; semilla 101; modalidad replay.

Corpus: `/Users/tomasvera/.cache/nanochat`. Artefactos/checkpoints: `/Users/tomasvera/dev/nanochat-mlx/experiments/navarro/artifacts/primary`. Hashes: [manifest](manifest.json).

| Medida | Nativo | Control | Intervención |
|---|---:|---:|---:|
| Parámetros | 125829648 | null | null |
| Tokens procesados | 819200 | null | null |
| NLL final | 6.630266 | null | null |
| Tiempo completo/paso o solicitud (s) | 0.6228055 | null | null |
| Tokens/s | 13153.38 | null | null |
| Pico MLX (bytes) | 6161390588 | null | null |
| RSS máximo (bytes) | 1550008320 | null | null |
| Pesos residentes (bytes) | null | null | null |
| Conversión (s) | null | null | null |

Corrección: `True`. Fidelidad: `True`. IC confirmatorios: `None`. No inferioridad confirmatoria: `None`.

Motivo: None.

- Single engineering process; no confirmatory confidence interval.
- RSS and Metal are reported separately, never summed.
- Per-phase profiling and pipeline timing are separate requirements.

[Pasos crudos](steps.csv) · [Tensores](tensors.csv) · [Eventos y errores](events.jsonl) · [Corrección](correctness.json) · [Resumen completo](summary.json) · [Comandos](commands.sh) · [Fuentes](source_map.md)
