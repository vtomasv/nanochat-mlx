# ¿Puede MM RePair mejorar nuestros experimentos?

Revisión del 14 de septiembre de 2026. Repositorio de autores: [manzai/mm-repair](https://gitlab.com/manzai/mm-repair), fijado al commit [`edd85fa3193dd1e6e60e1d4886f1ad8ded2a138f`](https://gitlab.com/manzai/mm-repair/-/commit/edd85fa3193dd1e6e60e1d4886f1ad8ded2a138f). La consulta del repositorio confirma la misma revisión que se había registrado al comenzar la campaña.

**Sí aporta código de referencia y posibilidades de optimización, sobre todo para E2/E3. La comprobación realizada no demuestra que sustituir nuestro codec por el suyo mejore los experimentos completos.** En los 14 bloques de tensores entrenados examinados, el constructor de los autores generó cero reglas RePair. El caso sintético repetitivo sí produjo reglas y su constructor fue más rápido.

Esta revisión es un diagnóstico adicional. No cambia los resultados, tolerancias, implementaciones ni decisiones de la campaña v1. No se volvió a entrenar ni se ejecutó un nuevo benchmark de inferencia.

**Seguimiento posterior:** la [implementación híbrida y sus pruebas](hybrid-repair.md) utiliza esta revisión como punto de partida. Las cifras de este documento corresponden al diagnóstico previo con nuestro codec de `7787ae2`.

## Qué se revisó y ejecutó

Se descargó el archivo fuente de la revisión fijada, se inspeccionaron `Readme.md`, `makefile`, `bin2csrv.cpp`, `brepair/irepair0.c`, las estructuras auxiliares de `brepair`, `rematrix.h`, `rematrix.hpp`, `remm.c` y `sdsl/encode.cpp`.

Se compilaron **sin modificar los archivos fuente** dos ejecutables de los autores para ARM64: `bin2csrvf` e `irepair0`. Se usaron comandos de compilación explícitos, sin la opción x86 `-msse4.2` del makefile general. No se instaló SDSL ni se ejecutaron `reivmm`, `remm`, ReANS o el operador paralelo. Tampoco se integró código ajeno en los módulos experimentales.

Se compararon:

- Seis bloques Muon: las primeras 128 filas de los estados de `blocks.4.mlp.c_fc` y `c_proj`, en checkpoints nativos 100, 500 y 1000, semilla 17.
- Dos bloques de pesos: las primeras 128 filas de esas dos matrices en el checkpoint 1000.
- Seis bloques AdamW: filas 0–127, 16384–16511 y 32640–32767 de `lm_head.weight.v` y `wte.weight.m`, checkpoint nativo 1000. Esta selección adicional se hizo tras revisar los costos por tensor de E2: estas familias estaban entre las más costosas. Es un diagnóstico dirigido, no una selección confirmatoria.
- Dos fixtures: una matriz con filas idénticas y una matriz aleatoria FP32.

Cada caso tuvo tres repeticiones consecutivas de construcción, alternando el orden autores/propio. Se comprobaron bit a bit las reconstrucciones de **ambos codecs en los 16 casos**. No se introdujo redondeo ni cuantización. La prueba se limita a matrices finitas sin cero negativo; no certifica el comportamiento del conversor ajeno ante todos los patrones IEEE.

## Hallazgos del código

| Componente | Diferencia relevante | Consecuencia para nosotros |
|---|---|---|
| Constructor `irepair0` | Mantiene registros de pares, listas de ocurrencias, hash y una estructura de frecuencias; actualiza vecinos al sustituir un par | Puede evitar las pasadas completas repetidas de nuestro constructor cuando hay muchas sustituciones. Cuando no hay pares repetidos, esa ventaja concreta no se materializa |
| Delimitador 0 | `forbidden_pair()` evita reglas que incluyan el fin de fila | Coincide con la restricción esencial de nuestro codec y sirve como referencia independiente |
| Diccionario de valores | `bin2csrv` comparte un diccionario entre bloques de una matriz; nuestra representación tiene uno por bloque | Puede evitar duplicados entre bloques, pero también aumenta el rango de IDs y el ancho de símbolos. No garantiza un formato menor |
| Límites de símbolos | El conversor rechaza códigos `id * cols + col + 1 >= 2^30` | No se puede asumir que una matriz grande admitida por nuestros bloques se convierte directamente con el diccionario global de los autores |
| Precisión de productos | Por defecto `matval` es `double`; con `FLOAT_VALS`, los pesos son `float` pero `xmatval` sigue siendo `double` | Podría ayudar a investigar el fallo numérico externo de E3, pero usarlo así cambia la aritmética del protocolo FP32. Requeriría un brazo explícito y nuevas mediciones |
| Scratch del producto | `NTval` se conserva en la estructura y se reutiliza entre productos; nuestro operador crea scratch por llamada/bloque | Hay una optimización concreta posible de asignaciones. En los bloques reales muestreados hay cero reglas, por lo que ese scratch particular no explica el cuello de botella observado |
| Paralelismo | `remm.c` reparte bloques de filas mediante pthreads y sincronización POSIX | Candidato a mejorar el producto CPU. No resuelve por sí solo falta de compresión ni elimina transferencias CPU/MLX. Ese camino no fue validado aquí en macOS |
| `re_iv` | Usa SDSL para leer enteros empaquetados; nuestro lector es propio | Útil como implementación de referencia. Este diagnóstico ejecutó `re_32` de los autores, no su implementación SDSL |

Fuentes primarias del código: [conversor](https://gitlab.com/manzai/mm-repair/-/blob/edd85fa3193dd1e6e60e1d4886f1ad8ded2a138f/bin2csrv.cpp), [constructor](https://gitlab.com/manzai/mm-repair/-/blob/edd85fa3193dd1e6e60e1d4886f1ad8ded2a138f/brepair/irepair0.c), [frecuencias](https://gitlab.com/manzai/mm-repair/-/blob/edd85fa3193dd1e6e60e1d4886f1ad8ded2a138f/brepair/heap.h), [producto re32](https://gitlab.com/manzai/mm-repair/-/blob/edd85fa3193dd1e6e60e1d4886f1ad8ded2a138f/rematrix.h), [producto reiv](https://gitlab.com/manzai/mm-repair/-/blob/edd85fa3193dd1e6e60e1d4886f1ad8ded2a138f/rematrix.hpp).

## Resultados de la comprobación CPU

Tamaño = payload del formato dividido por los bytes FP32 densos. Se muestra el formato forzado, **sin fallback RAW**, para observar si realmente comprime. `re_32` de los autores suma `.fval + .vc.R + .vc.C`; nuestro `re_iv` incluye los bytes de sus bloques serializados. No son medidas de RSS ni formatos con cabeceras idénticas. La comparación de tamaño entre esas columnas no atribuye una ventaja al constructor: cambia también el empaquetado.

| Caso | Forma | Reglas autores | Tamaño autores re32 | Tamaño propio reiv | Construcción autores (ms) | Construcción propia re32 (ms) |
|---|---|---:|---:|---:|---:|---:|
| Muon c_fc, paso 100 | 128 × 512 | 0 | 200,16 % | 178,27 % | 31,28 | 6,10 |
| Muon c_proj, paso 100 | 128 × 2048 | 0 | 199,92 % | 190,55 % | 119,78 | 31,21 |
| Muon c_fc, paso 500 | 128 × 512 | 0 | 200,17 % | 178,27 % | 33,06 | 6,00 |
| Muon c_proj, paso 500 | 128 × 2048 | 0 | 199,93 % | 190,56 % | 117,46 | 36,33 |
| Muon c_fc, paso 1000 | 128 × 512 | 0 | 200,17 % | 178,28 % | 33,41 | 6,60 |
| Muon c_proj, paso 1000 | 128 × 2048 | 0 | 199,95 % | 190,58 % | 119,03 | 26,80 |
| Peso c_fc, paso 1000 | 128 × 512 | 0 | 200,15 % | 178,26 % | 32,54 | 6,03 |
| Peso c_proj, paso 1000 | 128 × 2048 | 0 | 199,91 % | 190,53 % | 114,98 | 30,61 |
| Adam lm_head.v, filas 0–127 | 128 × 512 | 0 | 200,17 % | 178,28 % | 31,44 | 6,48 |
| Adam lm_head.v, filas 16384–16511 | 128 × 512 | 0 | 200,14 % | 178,25 % | 31,29 | 5,96 |
| Adam lm_head.v, filas 32640–32767 | 128 × 512 | 0 | 200,17 % | 178,28 % | 30,40 | 5,48 |
| Adam wte.m, filas 0–127 | 128 × 512 | 0 | 148,62 % | 132,36 % | 20,51 | 3,83 |
| Adam wte.m, filas 16384–16511 | 128 × 512 | 0 | 192,62 % | 171,55 % | 30,39 | 5,43 |
| Adam wte.m, filas 32640–32767 | 128 × 512 | 0 | 176,74 % | 157,41 % | 20,62 | 5,08 |
| Sintético repetitivo | 128 × 512 | 399 | 1,61 % | 0,58 % | 22,77 | 37,74 |
| Sintético aleatorio | 128 × 512 | 0 | 200,15 % | 178,25 % | 32,70 | 6,25 |

**Los tiempos son diagnósticos, no speedups de entrenamiento:** la ruta de los autores lanza dos procesos e incluye lectura/escritura de archivos; la propia usa una llamada en memoria. Se reporta la mediana de tres observaciones, sin intervalos de confianza. Una integración en memoria del constructor de los autores podría cambiar esa comparación. No es correcto extrapolar estos cocientes a un paso completo o a un kernel Metal.

El caso repetitivo demuestra que el constructor de los autores puede ayudar cuando existe repetición: 22,77 ms frente a 37,74 ms en este diagnóstico, aproximadamente 1,66× de velocidad de construcción. Es un fixture artificial y no convierte los resultados de E2/E3 en positivos.

En los 14 bloques entrenados no apareció ninguna regla. En los bloques MLP, casi todos los valores son distintos: el propio diccionario FP32 ya ocupa cerca del tamaño denso, antes de añadir columnas y símbolos. En los bloques Adam examinados, el formato tampoco se acerca al objetivo del 95 %.

Aunque no se ejecutó el `re_iv` de SDSL, en estos casos sin reglas se puede calcular una cota conservadora: diccionario FP32 más una aparición de cada terminal distinto al ancho mínimo del flujo C, omitiendo cabeceras, separadores y repeticiones. Esa cota está entre **132,18 % y 190,53 %** del denso para los 14 bloques reales. Por tanto, empaquetar esos mismos flujos sin reglas no alcanzaría el 95 %. Esto es una deducción de los símbolos observados, no una medición del ejecutable `reivmm`.

## Qué cambiaría en cada experimento

### E1: no hay una sustitución directa

MM RePair no implementa nuestra cinta de activaciones ni el bitmap/rank de N1. La mejora más relacionada con el resultado de E1 sigue siendo medir pack/unpack y almacenamiento compacto en GPU, para evitar el costo CPU de esa integración. Ese trabajo no lo aporta este repositorio.

### E2: optimizar el descarte y después medir el constructor

La revisión favorece este orden:

1. **Descarte temprano exacto:** si solo el diccionario y la cabecera ya superan el presupuesto del 95 %, no construir una gramática que nunca podrá ser elegible. La decisión final seguiría siendo RAW; se evitaría trabajo de búsqueda de pares. No se implementó este cambio en la revisión.
2. **Mantener el control `staged_raw`:** ya costó 1,4644 s/paso frente a 0,6290 s nativos, aproximadamente 2,33×. Es evidencia de que también hay que resolver la integración y el traslado del estado, no solo reemplazar el constructor. Es un control observado, no una cota matemática universal.
3. **Probar el constructor incremental en memoria solo donde tenga oportunidad:** seleccionar bloques con repetición real, respetando las pruebas de corrección, y medir junto con decode/update/encode. No se debe introducir un proceso externo por cada estado y cada paso como solución de producción.

La ausencia de reglas en la muestra obliga a precisar la explicación del costo anterior: no puede atribuirse todo el sobrecosto de E2 a muchas pasadas de sustitución. En estos bloques se pagan diccionarios, simbolización y búsqueda inicial, pero no ocurren sustituciones. Los logs v1 por tensor muestran además el peso de procesar las grandes matrices de AdamW.

### E3: referencia útil, sin promesa de aceleración completa

Hay oportunidades para estudiar reutilización de buffers y paralelismo CPU. La aritmética intermedia `double` es además una pista concreta para investigar el fallo de tolerancia del checkpoint SFT externo. Sin embargo, usar acumulación más ancha sería una variante numérica explícita: no se puede presentarla como el mismo operador FP32 de v1 ni omitir su costo.

La falta de compresión medida permanece. Una mejora del microbenchmark CPU no bastaría: hay que superar al nativo MLX incluyendo prefill, decode, copias y memoria total. ReANS, DRV y reordenamiento de columnas son rutas distintas mencionadas por el repositorio; estaban excluidas de v1 y requerirían un estudio separado. No se ensayaron aquí.

## Licencias y portabilidad: corrección de la revisión inicial

El archivo raíz [`LICENSE.md`](https://gitlab.com/manzai/mm-repair/-/blob/edd85fa3193dd1e6e60e1d4886f1ad8ded2a138f/LICENSE.md) declara Apache 2.0. **Eso no describe por sí solo todos los archivos:** `brepair/irepair0.c`, `heap.h` y otros componentes llevan avisos GPL versión 2 o posterior. La revisión inicial de v1 había consultado solo la licencia raíz y no incorporó código ajeno. Antes de redistribuir o integrar componentes debe conservarse el inventario de licencias por archivo; esta comprobación usó ejecutables locales externos.

Los dos componentes utilizados compilaron en el M3 Max con clang, fuera del makefile general. La compilación de `bin2csrvf` emitió un aviso de `dangling else` y terminó correctamente. Esto no valida automáticamente SDSL, las opciones x86 ni la sincronización del producto paralelo en macOS. Los fuentes y binarios externos siguen bajo `experiments/navarro/sources/`, ignorado por Git.

## Evidencia y reproducción

- [Probe MLP/Muon y fixtures](../../experiments/navarro/verification/mm-repair-edd85fa3-probe/probe.json).
- [Probe AdamW dirigido](../../experiments/navarro/verification/mm-repair-edd85fa3-adam-probe/probe.json).
- [Script propio de comprobación](../../scripts/navarro_mm_repair_probe.py): lector de tensores y formatos, tres repeticiones, verificación exacta y registros. El segundo probe conserva una copia del script ejecutado; el primero registra hashes de los fuentes externos y binarios.
- Los JSON contienen formas, segmentos, hashes de los datos exportados, revisión y hashes del código externo, comandos y observaciones de tiempo. Los logs individuales conservan la salida de los dos ejecutables.

Compilación desde la carpeta descargada de la revisión fijada:

```sh
clang -O3 -std=gnu99 -ffp-contract=off \
  brepair/irepair0.c brepair/array.c brepair/hash.c brepair/heap.c \
  brepair/records.c brepair/basics.c -lm -o brepair/irepair0
clang++ -O3 -std=c++17 -ffp-contract=off -DTypecode=2 bin2csrv.cpp -o bin2csrvf
```

Desde nanochat-mlx, con una ruta local de fuentes y nuevos IDs de salida:

```sh
uv run --frozen python -m scripts.navarro_mm_repair_probe \
  --upstream /ruta/a/mm-repair --run-id nueva-comprobacion-mlp
uv run --frozen python -m scripts.navarro_mm_repair_probe \
  --upstream /ruta/a/mm-repair --adam-only --run-id nueva-comprobacion-adam
```

Los checkpoints originales deben estar disponibles localmente. Las pruebas son acotadas a bloques y CPU; no sustituyen una campaña completa ni autorizan extrapolar el resultado a otros modelos o representaciones.
