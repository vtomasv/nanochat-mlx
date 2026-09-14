# E3: FAILURE_CORRECTNESS

Brazo: `external_sft_validation`. Estado: `FAILURE_CORRECTNESS`. SHA: `b54b9fc139f455a9a5e60dc9a688ca9dbdb22944`. Hardware: Apple M3 Max; FP32; perfil primary; semilla 17; modalidad replay.

Corpus: `/Users/tomasvera/.cache/nanochat`. Artefactos/checkpoints: `/Users/tomasvera/dev/nanochat-mlx/experiments/navarro/artifacts/primary`. Hashes: [manifest](manifest.json).

| Medida | Nativo | Control | Intervención |
|---|---:|---:|---:|
| Parámetros | null | null | 36700296 |
| Tokens procesados | null | null | null |
| NLL final | null | null | null |
| Tiempo completo/paso o solicitud (s) | null | null | null |
| Tokens/s | null | null | null |
| Pico MLX (bytes) | null | null | null |
| RSS máximo (bytes) | null | null | null |
| Pesos residentes (bytes) | null | null | null |
| Conversión (s) | null | null | null |

Corrección: `False`. Fidelidad: `True`. IC confirmatorios: `None`. No inferioridad confirmatoria: `None`.

Motivo: None.

- External depth4 SFT checkpoint, not the preregistered depth8 reference.
- Existing tokenizer has compatible vocabulary size; checkpoint does not record tokenizer hash or full training provenance.
- Representation and teacher-forced model correctness only; no external latency SUCCESS claimed.
- Local FP64 oracle and decoded-dense control isolate lossless storage from changed FP32 reduction order; tolerances unchanged.

[Pasos crudos](steps.csv) · [Tensores](tensors.csv) · [Eventos y errores](events.jsonl) · [Corrección](correctness.json) · [Resumen completo](summary.json) · [Comandos](commands.sh) · [Fuentes](source_map.md)
