# E1: INCONCLUSIVE

Brazo: `tape_dense`. Estado: `measured`. SHA: `b54b9fc139f455a9a5e60dc9a688ca9dbdb22944`. Hardware: Apple M3 Max; FP32; perfil smoke; semilla 101; modalidad replay.

Corpus: `/Users/tomasvera/.cache/nanochat`. Artefactos/checkpoints: `/Users/tomasvera/dev/nanochat-mlx/experiments/navarro/artifacts/smoke`. Hashes: [manifest](manifest.json).

| Medida | Nativo | Control | Intervención |
|---|---:|---:|---:|
| Parámetros | null | null | 36700296 |
| Tokens procesados | null | null | 2560 |
| NLL final | null | null | 10.30819 |
| Tiempo completo/paso o solicitud (s) | null | null | 0.03916653 |
| Tokens/s | null | null | 6536.192 |
| Pico MLX (bytes) | null | null | 1231102944 |
| RSS máximo (bytes) | null | null | 704987136 |
| Pesos residentes (bytes) | null | null | null |
| Conversión (s) | null | null | null |

Corrección: `None`. Fidelidad: `None`. IC confirmatorios: `None`. No inferioridad confirmatoria: `None`.

Motivo: None.

- Single engineering process; no confirmatory confidence interval.
- RSS and Metal are reported separately, never summed.
- Per-phase profiling and pipeline timing are separate requirements.

[Pasos crudos](steps.csv) · [Tensores](tensors.csv) · [Eventos y errores](events.jsonl) · [Corrección](correctness.json) · [Resumen completo](summary.json) · [Comandos](commands.sh) · [Fuentes](source_map.md)
