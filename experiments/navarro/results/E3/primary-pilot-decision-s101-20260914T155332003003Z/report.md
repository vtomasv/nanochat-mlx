# E3: NEGATIVE_SCREENING_NO_COMPRESSIBILITY

Brazo: `decision`. Estado: `NEGATIVE_SCREENING_NO_COMPRESSIBILITY`. SHA: `b54b9fc139f455a9a5e60dc9a688ca9dbdb22944`. Hardware: Apple M3 Max; FP32; perfil primary; semilla 101; modalidad replay.

Corpus: `/Users/tomasvera/.cache/nanochat`. Artefactos/checkpoints: `/Users/tomasvera/dev/nanochat-mlx/experiments/navarro/artifacts/primary`. Hashes: [manifest](manifest.json).

| Medida | Nativo | Control | Intervención |
|---|---:|---:|---:|
| Parámetros | null | null | null |
| Tokens procesados | null | null | null |
| NLL final | null | null | null |
| Tiempo completo/paso o solicitud (s) | 0.2692683 | 0.902474 | 1.283671 |
| Tokens/s | 497.1536 | 145.1088 | 204.969 |
| Pico MLX (bytes) | 843407996 | 843637372 | 690178356 |
| RSS máximo (bytes) | 626737152 | 681590784 | 663896064 |
| Pesos residentes (bytes) | 503318592 | 506602389 | 506602389 |
| Conversión (s) | 0 | 0.08634804 | 0.08492663 |

Corrección: `True`. Fidelidad: `True`. IC confirmatorios: `None`. No inferioridad confirmatoria: `None`.

Motivo: No trained MLP matrix reaches <=95% RAW.

- Step1000 seed17 only; all MLP matrices screened.
- Forced central c_fc, eight reserved prompts, teacher-forced decode128.
- No GPU direct kernel: CPU/MLX movement included.
- No extrapolation to all neural compression methods.

[Pasos crudos](steps.csv) · [Tensores](tensors.csv) · [Eventos y errores](events.jsonl) · [Corrección](correctness.json) · [Resumen completo](summary.json) · [Comandos](commands.sh) · [Fuentes](source_map.md)
