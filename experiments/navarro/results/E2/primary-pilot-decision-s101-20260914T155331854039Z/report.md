# E2: NEGATIVE_SCREENING

Brazo: `decision`. Estado: `NEGATIVE_SCREENING`. SHA: `b54b9fc139f455a9a5e60dc9a688ca9dbdb22944`. Hardware: Apple M3 Max; FP32; perfil primary; semilla 101; modalidad replay.

Corpus: `/Users/tomasvera/.cache/nanochat`. Artefactos/checkpoints: `/Users/tomasvera/dev/nanochat-mlx/experiments/navarro/artifacts/primary`. Hashes: [manifest](manifest.json).

| Medida | Nativo | Control | Intervención |
|---|---:|---:|---:|
| Parámetros | 125829648 | 125829648 | 125829648 |
| Tokens procesados | 819200 | 819200 | 819200 |
| NLL final | 6.630254 | 6.630263 | 6.630254 |
| Tiempo completo/paso o solicitud (s) | 0.6289537 | 1.464384 | 17.42784 |
| Tokens/s | 13024.81 | 5594.16 | 470.0526 |
| Pico MLX (bytes) | 6161390588 | 5255418752 | 5255418748 |
| RSS máximo (bytes) | 1548943360 | 3236151296 | 2224291840 |
| Pesos residentes (bytes) | null | null | null |
| Conversión (s) | null | null | null |

Corrección: `True`. Fidelidad: `True`. IC confirmatorios: `None`. No inferioridad confirmatoria: `None`.

Motivo: Preregistered engineering gate failed.

- Engineering seed101, 100 full updates per arm; no confirmatory CI.
- Negative conclusion applies to this implementation, early training checkpoint family and M3 Max.
- CPU residency and copies count toward RSS; MLX allocator reduction alone is not physical-memory reduction.

[Pasos crudos](steps.csv) · [Tensores](tensors.csv) · [Eventos y errores](events.jsonl) · [Corrección](correctness.json) · [Resumen completo](summary.json) · [Comandos](commands.sh) · [Fuentes](source_map.md)
