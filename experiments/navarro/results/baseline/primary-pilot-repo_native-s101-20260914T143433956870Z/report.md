# baseline: INCONCLUSIVE

Brazo: `repo_native`. Estado: `measured`. SHA: `b54b9fc139f455a9a5e60dc9a688ca9dbdb22944`. Hardware: Apple M3 Max; FP32; perfil primary; semilla 101; modalidad replay.

Corpus: `/Users/tomasvera/.cache/nanochat`. Artefactos/checkpoints: `/Users/tomasvera/dev/nanochat-mlx/experiments/navarro/artifacts/primary`. Hashes: [manifest](manifest.json).

| Medida | Nativo | Control | Intervención |
|---|---:|---:|---:|
| Parámetros | null | null | null |
| Tokens procesados | null | null | null |
| NLL final | null | null | null |
| Tiempo completo/paso o solicitud (s) | null | null | null |
| Tokens/s | null | null | null |
| Pico MLX (bytes) | 6161390580 | null | null |
| RSS máximo (bytes) | 1614888960 | null | null |
| Pesos residentes (bytes) | null | null | null |
| Conversión (s) | null | null | null |

Corrección: `None`. Fidelidad: `None`. IC confirmatorios: `None`. No inferioridad confirmatoria: `None`.

Motivo: None.

- Original CLI engineering prefix; schedule remains 1000.
- Stop occurs after optimizer checkpoint save; meta file is replaced by experimental event metadata.

[Pasos crudos](steps.csv) · [Tensores](tensors.csv) · [Eventos y errores](events.jsonl) · [Corrección](correctness.json) · [Resumen completo](summary.json) · [Comandos](commands.sh) · [Fuentes](source_map.md)
