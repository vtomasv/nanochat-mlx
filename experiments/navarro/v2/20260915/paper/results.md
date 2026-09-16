# Resultados descriptivos para el manuscrito

| Ensayo | Decisión | Pasos | NLL calibración | s/paso | Tiempo/nativo | Pico MLX GiB | RSS GiB | P/denso |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| [reference](../trials/reference/summary.json) | REFERENCE_ONLY | 1000 | 5.299305 | 1.007348 | null | 6.792171 | 2.067444 | null |
| [native](../trials/native/summary.json) | CONTROL | 100 | 6.220081 | 0.958646 | 1.000000 | 6.792171 | 2.911835 | null |
| [checkpoint-native](../trials/checkpoint-native/summary.json) | CONTROL | 100 | 6.220081 | 1.035871 | 1.080556 | 5.180263 | 2.906448 | null |
| [tape-dense](../trials/tape-dense/summary.json) | CONTROL | 100 | 6.220081 | 0.970747 | 1.012623 | 5.187647 | 2.906433 | 1.000000 |
| [tape-recompute](../trials/tape-recompute/summary.json) | CONTROL | 100 | 6.220081 | 1.024781 | 1.068988 | 4.937647 | 2.906372 | 0.000000 |
| [bitmap-cpu](../trials/bitmap-cpu/summary.json) | NEGATIVE_SCREENING_TIME | 100 | 6.220081 | 2.179313 | 2.273323 | 4.937647 | 3.134964 | 0.452252 |
| [bitmap-metal-r128](../trials/bitmap-metal-r128/summary.json) | INCONCLUSIVE_PILOT_NOT_SHORTLISTED | 100 | 6.220081 | 0.980208 | 1.022492 | 5.058817 | 2.906021 | 0.452235 |
| [bitmap-metal-r256](../trials/bitmap-metal-r256/summary.json) | INCONCLUSIVE_PILOT_NOT_SHORTLISTED | 100 | 6.220081 | 1.050415 | 1.095727 | 5.057841 | 2.908112 | 0.448328 |
| [bitmap-metal-r512](../trials/bitmap-metal-r512/summary.json) | INCONCLUSIVE_PILOT_NOT_SHORTLISTED | 100 | 6.220081 | 1.111293 | 1.159232 | 5.057352 | 2.907471 | 0.446375 |
| [bitmap-metal-scan](../trials/bitmap-metal-scan/summary.json) | INCONCLUSIVE_PILOT_NOT_SHORTLISTED | 100 | 6.220081 | 1.056366 | 1.101935 | 5.056864 | 2.908203 | 0.444422 |
| [bitmap-metal-simd](../trials/bitmap-metal-simd/summary.json) | INCONCLUSIVE_PILOT_NOT_SHORTLISTED | 100 | 6.220081 | 0.922360 | 0.962149 | 5.057841 | 2.906631 | 0.448328 |

## Cotas de diccionario en checkpoints entrenados

| Paso | Objeto | Filas/bloque | Bloques descartados | Ahorro máximo bajo selector RAW |
|---:|---|---:|---:|---:|
| 100 | model.safetensors | 32 | 7300/7300 | 0.0000% |
| 100 | model.safetensors | 128 | 1828/1828 | 0.0000% |
| 100 | model.safetensors | 512 | 460/460 | 0.0000% |
| 100 | optimizer.safetensors | 32 | 5576/13444 | 56.9154% |
| 100 | optimizer.safetensors | 128 | 1242/3364 | 61.4003% |
| 100 | optimizer.safetensors | 512 | 266/844 | 66.8980% |
| 500 | model.safetensors | 32 | 7300/7300 | 0.0000% |
| 500 | model.safetensors | 128 | 1828/1828 | 0.0000% |
| 500 | model.safetensors | 512 | 460/460 | 0.0000% |
| 500 | optimizer.safetensors | 32 | 11955/13444 | 10.7711% |
| 500 | optimizer.safetensors | 128 | 3267/3364 | 2.8067% |
| 500 | optimizer.safetensors | 512 | 829/844 | 1.7361% |
| 1000 | model.safetensors | 32 | 7300/7300 | 0.0000% |
| 1000 | model.safetensors | 128 | 1828/1828 | 0.0000% |
| 1000 | model.safetensors | 512 | 460/460 | 0.0000% |
| 1000 | optimizer.safetensors | 32 | 12238/13444 | 8.7239% |
| 1000 | optimizer.safetensors | 128 | 3269/3364 | 2.7488% |
| 1000 | optimizer.safetensors | 512 | 834/844 | 1.1574% |

Activaciones: 64 muestras P/H, 0 reglas; roundtrip exacto: True. [Datos](../screening/activations.json).
Las cotas solo afectan diccionarios locales del tamaño indicado. Bloques no descartados no se declaran compresibles.

## Límites y próximos pasos

- Una réplica de proceso por escenario: no hay IC confirmatorios ni comparación de convergencia.
- NLL de calibración ponderada por tokens; el conjunto reservado queda fuera de búsqueda.
- MLX y RSS se solapan; no sumarlos como memoria física.
- La calidad al final de una ventana no demuestra estabilidad de una trayectoria completa.
- H directa, diccionario global, KV, cuantización y adaptación congelada son ramas separadas; no se presentan como ejecutadas.
- Toda corrección de implementación exige nuevos snapshots; no reescribe resultados anteriores.

Fuentes: [protocolo](../../../../../ESPECIFICACIONES_NAVARRO_NANOCHAT_MLX.md), [autoresearch](https://github.com/karpathy/autoresearch), [N3](https://users.dcc.uchile.cl/~gnavarro/ps/pvldb22.pdf).
