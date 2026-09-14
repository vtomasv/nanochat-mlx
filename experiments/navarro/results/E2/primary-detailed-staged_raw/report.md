# E2: INCONCLUSIVE

Brazo: `staged_raw`. Estado: `measured`. SHA: `b54b9fc139f455a9a5e60dc9a688ca9dbdb22944`. Hardware: Apple M3 Max; FP32; perfil primary; semilla 17; modalidad replay.

Corpus: `/Users/tomasvera/.cache/nanochat`. Artefactos/checkpoints: `/Users/tomasvera/dev/nanochat-mlx/experiments/navarro/artifacts/primary`. Hashes: [manifest](manifest.json).

| Medida | Nativo | Control | Intervención |
|---|---:|---:|---:|
| Parámetros | null | null | null |
| Tokens procesados | null | null | null |
| NLL final | null | null | null |
| Tiempo completo/paso o solicitud (s) | null | null | null |
| Tokens/s | null | null | null |
| Pico MLX (bytes) | null | null | 5255418748 |
| RSS máximo (bytes) | null | null | 2314534912 |
| Pesos residentes (bytes) | null | null | null |
| Conversión (s) | null | null | null |

Corrección: `None`. Fidelidad: `None`. IC confirmatorios: `None`. No inferioridad confirmatoria: `None`.

Motivo: None.

- Detailed cold profile with explicit barriers; excluded from principal timing ratios.
- Nested pack/unpack/decode/encode times overlap their parent phase and must not be added again.
- Additional attention/FC forward counts are summed across both microbatches.

[Pasos crudos](steps.csv) · [Tensores](tensors.csv) · [Eventos y errores](events.jsonl) · [Corrección](correctness.json) · [Resumen completo](summary.json) · [Comandos](commands.sh) · [Fuentes](source_map.md)
