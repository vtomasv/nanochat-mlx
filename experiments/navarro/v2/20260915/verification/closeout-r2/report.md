# Extensión de gradientes: cierre exploratorio

Seis procesos de 100 updates desde checkpoint 100. Orden aleatorizado antes de medir; una observación por escenario, sin IC confirmatorios.

| Brazo | s/update | MLX GiB | Tiempo/nativo | Ahorro vs. tape denso | Decisión |
|---|---:|---:|---:|---:|---|
| tape-dense | 0.844858 | 5.187678 | 1.034485 | 0.000% | CONTROL |
| rows-bitmap | 0.926970 | 4.760586 | 1.135027 | 8.233% | NOT_SHORTLISTED |
| bitmap | 0.970227 | 5.057871 | 1.187993 | 2.502% | CONTROL |
| rows-dense | 0.917403 | 4.891736 | 1.123312 | 5.705% | NOT_SHORTLISTED |
| tape-recompute | 0.905295 | 4.937678 | 1.108486 | 4.819% | CONTROL |
| native | 0.816694 | 6.792201 | 1.000000 | -30.930% | CONTROL |

Ambos candidatos pasan los criterios puntuales de memoria, RSS y calibración, pero fallan tiempo ≤1,10× nativo. No se preselecciona ninguno.

Las filas compactas conservan 36,49% del acumulador denso tras el primer microbatch y 38,77% tras el segundo (medias de 100 updates). Estos ratios incluyen los otros gradientes densos y no son el ahorro de memoria global.

Se comprueba en cada empaquetado que las filas omitidas contienen bits de cero positivo. Se restauran todos los gradientes antes del optimizador; se conservan momentos y weight decay. La prueba pequeña de diez updates compara pesos y estados por ruta del parámetro.

Las huellas finales de los 100 updates difieren de los controles: 0/60 tensores de modelo, 8/24 estados Adam y 0/52 estados Muon tienen hash idéntico en ambas comparaciones. La representación exacta no demuestra una trayectoria flotante idéntica. No se conservaron checkpoints finales para calcular su distancia; NLL casi igual no reemplaza esa medición.

El primer intento se detuvo antes de pilotos porque el verificador comparaba listas de estados por posición, con distinto orden de inserción. Se preserva su fuente y fallo. La revisión r2 compara estados por nombres; no cambia tolerancias.

No mezclar este control nativo (0,816694 s/update) con el de la campaña principal (0,958646). La diferencia entre procesos obliga a mantener las comparaciones dentro de cada estudio. El bitmap aislado también cambia su posición temporal; sigue siendo evidencia exploratoria.
