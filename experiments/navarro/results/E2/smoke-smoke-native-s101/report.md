# E2: INCONCLUSIVE

Brazo: `native`. Estado: `measured`. SHA: `b54b9fc139f455a9a5e60dc9a688ca9dbdb22944`. Hardware: Apple M3 Max; FP32; perfil smoke; semilla 101; modalidad replay.

Corpus: `/Users/tomasvera/.cache/nanochat`. Artefactos/checkpoints: `/Users/tomasvera/dev/nanochat-mlx/experiments/navarro/artifacts/smoke`. Hashes: [manifest](manifest.json).

| Medida | Nativo | Control | Intervención |
|---|---:|---:|---:|
| Parámetros | 36700296 | null | null |
| Tokens procesados | 2560 | null | null |
| NLL final | 10.30819 | null | null |
| Tiempo completo/paso o solicitud (s) | 0.03139427 | null | null |
| Tokens/s | 8154.355 | null | null |
| Pico MLX (bytes) | 1231102944 | null | null |
| RSS máximo (bytes) | 670810112 | null | null |
| Pesos residentes (bytes) | null | null | null |
| Conversión (s) | null | null | null |

Corrección: `True`. Fidelidad: `True`. IC confirmatorios: `None`. No inferioridad confirmatoria: `None`.

Motivo: None.

- Single engineering process; no confirmatory confidence interval.
- RSS and Metal are reported separately, never summed.
- Per-phase profiling and pipeline timing are separate requirements.

[Pasos crudos](steps.csv) · [Tensores](tensors.csv) · [Eventos y errores](events.jsonl) · [Corrección](correctness.json) · [Resumen completo](summary.json) · [Comandos](commands.sh) · [Fuentes](source_map.md)
