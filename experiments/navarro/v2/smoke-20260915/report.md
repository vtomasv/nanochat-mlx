# Campaña Navarro v2: cuaderno de autoresearch

**Estado de evidencia: exploración de ingeniería. No demuestra todavía no inferioridad ni mejora confirmatoria.**

Cada intento conserva código, hipótesis, controles, errores de gradientes, métricas y eventos. Los descartes no borran artefactos. El test reservado no se usa para seleccionar candidatos.

| Ensayo | Decisión | Pasos | NLL calibración | s/paso | Tiempo/nativo | Pico MLX GiB | RSS GiB | P/denso |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| [reference](trials/reference/summary.json) | REFERENCE_ONLY | 10 | 10.390797 | 0.033439 | null | 0.281626 | 0.419418 | null |
| [native](trials/native/summary.json) | CONTROL | 2 | 10.387176 | 0.030178 | 1.000000 | 0.281626 | 0.451340 | null |
| [tape-dense](trials/tape-dense/summary.json) | CONTROL | 2 | 10.387176 | 0.053059 | 1.758211 | 0.281626 | 0.448181 | 1.000000 |
| [tape-recompute](trials/tape-recompute/summary.json) | CONTROL | 2 | 10.387176 | 0.045580 | 1.510387 | 0.281626 | 0.450806 | 0.000000 |
| [bitmap-metal-r256](trials/bitmap-metal-r256/summary.json) | INCONCLUSIVE_PILOT_NOT_SHORTLISTED | 2 | 10.387176 | 0.056279 | 1.864910 | 0.281626 | 0.452896 | 0.536476 |

## Cotas de diccionario en checkpoints entrenados

| Paso | Objeto | Filas/bloque | Bloques descartados | Ahorro máximo bajo selector RAW |
|---:|---|---:|---:|---:|

Activaciones: 0 muestras P/H, 0 reglas; roundtrip exacto: None. [Datos](screening/activations.json).
Las cotas solo afectan diccionarios locales del tamaño indicado. Bloques no descartados no se declaran compresibles.

## Límites y próximos pasos

- Una réplica de proceso por escenario: no hay IC confirmatorios ni comparación de convergencia.
- NLL de calibración ponderada por tokens; el conjunto reservado queda fuera de búsqueda.
- MLX y RSS se solapan; no sumarlos como memoria física.
- La calidad al final de una ventana no demuestra estabilidad de una trayectoria completa.
- H directa, diccionario global, KV, cuantización y adaptación congelada son ramas separadas; no se presentan como ejecutadas.
- Toda corrección de implementación exige nuevos snapshots; no reescribe resultados anteriores.

Fuentes: [protocolo](../../../../ESPECIFICACIONES_NAVARRO_NANOCHAT_MLX.md), [autoresearch](https://github.com/karpathy/autoresearch), [N3](https://users.dcc.uchile.cl/~gnavarro/ps/pvldb22.pdf).
