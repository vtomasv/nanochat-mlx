# Tres experimentos de estructuras compactas para nanochat-mlx

Especificación de implementación y evaluación para Codex · versión 1.0 · 13 de septiembre de 2026.

**Estado: diseño preregistrable; experimentos no ejecutados.** No se dispone de mediciones de tu Mac. Todos los campos de rendimiento inicial y experimental deben comenzar como `null`, con estado `not_measured`. Los números de aceptación de este documento son decisiones de diseño, no resultados de los artículos.

Repositorio objetivo: <https://github.com/vtomasv/nanochat-mlx>.
Revisión auditada: `b54b9fc139f455a9a5e60dc9a688ca9dbdb22944`.
Hardware objetivo: Mac de la familia M3 con 128 GB de memoria unificada; detectar modelo exacto al ejecutar.

## 0. Encargo para Codex y alcance científico

Implementa exactamente estos tres experimentos, primero por separado:

| ID | Fase | Intervención | Pregunta experimental |
|---|---|---|---|
| E1 | Entrenamiento completo | Guardar activaciones positivas del MLP con bitmap y `rank`, conservando los valores originales | ¿Reduce la memoria máxima con un costo temporal aceptable y gradientes equivalentes? |
| E2 | Entrenamiento completo | Mantener estados de Muon y AdamW comprimidos sin pérdida entre actualizaciones | ¿La compresión compensa sus índices, reconstrucción y recompresión en cada paso? |
| E3 | Inferencia | Multiplicar directamente por matrices de pesos comprimidas mediante gramáticas | ¿El modelo aprendido tiene repetición suficiente para mejorar inferencia real en el Mac? |

El fundamento científico de las estructuras debe proceder exclusivamente de trabajos con Gonzalo Navarro como autor. La arquitectura y el optimizador ya presentes en el repositorio constituyen el sistema experimental. La documentación oficial de MLX puede usarse para APIs y kernels; no constituye otro método científico de compresión.

**Estos son estudios de adaptación, no reproducciones de entrenamiento neuronal demostrado por Navarro.** No atribuirle nuestras reglas de selección, derivadas, integración con MLX, umbrales o resultados. Ninguna hipótesis exige que el resultado sea positivo. Un resultado negativo correctamente medido completa el experimento.

No introducir cuantización, poda, cambio de rango, adaptadores, otro optimizador ni modificación del conjunto de parámetros aprendibles. No añadir compresores alternativos como sustitución silenciosa del algoritmo prescrito. No combinar E1 y E2 para el veredicto principal.

Lee este documento completo antes de implementar. Si el usuario pide solo un experimento, ejecuta la infraestructura común y ese ID. Si pide los tres, reutiliza corpus, tokenizer, estados iniciales y controles compatibles, pero conserva resultados separados.

### 0.1 Fuentes obligatorias y qué se toma de cada una

**N1. González, Grabowski, Mäkinen y Navarro. Practical Implementation of Rank and Select Queries (2005).**
Texto: <https://users.dcc.uchile.cl/~gnavarro/ps/wea05.pdf>.
Localizador: sección 1.3, «Using a Single Level Plus Sequential Scan».
Base de E1: bitmap de palabras y contadores muestreados; `rank` se obtiene con un contador y popcounts sobre pocas palabras. Se adopta su compromiso entre índices y recorrido local. No se atribuye al artículo la compresión de activaciones ni el entrenamiento.

**N2. Navarro y Providel. Fast, Small, Simple Rank/Select on Bitmaps (SEA 2012).**
Texto: <https://users.dcc.uchile.cl/~gnavarro/ps/sea12.1.pdf>.
Publicación: <https://doi.org/10.1007/978-3-642-30850-5_26>.
Lectura complementaria para elegir operaciones necesarias. E1 necesita `rank` y acceso; no hay que implementar `select` ni una estructura dinámica sin una necesidad medida.

**N3. Ferragina, Gagie, Köppl, Manzini, Navarro, Striani y Tosoni. Improving Matrix-vector Multiplication via Lossless Grammar-Compressed Matrices (PVLDB 15(10):2175–2187, 2022).**
Texto consultado: <https://arxiv.org/pdf/2203.14540>.
Ficha: <https://arxiv.org/abs/2203.14540>.
Código señalado por los autores: <https://gitlab.com/manzai/mm-repair>.
Localizadores del preprint: §2 CSRV; §3.1 y teorema 3.4 producto derecho; §3.2 y teorema 3.10 producto izquierdo; §4 variantes `re_32` y `re_iv`.
Base de E2/E3: diccionario de valores, símbolos que incluyen columna, límites de fila y reglas RePair que no atraviesan esos límites. E2 usa la representación; E3 también su producto. El artículo evalúa matrices de datos y CPU, no pesos neuronales actualizados en GPU. No trasladar sus aceleraciones al Mac. No implementar `re_ans` ni reordenamiento de columnas en la versión principal.

**N4. Navarro. Practical Adaptive Dynamic Bitvectors (Software: Practice and Experience 55(9):1539–1559, 2025).**
Texto: <https://users.dcc.uchile.cl/~gnavarro/ps/spe25.pdf>.
Código: <https://github.com/gonzalonavarro/AdaptiveDynamicBitvectors>.
Lectura de contraste: su caso de muchas consultas entre actualizaciones no coincide con la vida corta e inmutable de una activación guardada. No incorporar su árbol dinámico a E1 solo por ser más reciente. Esta exclusión es parte de la selección razonada de literatura.

Codex debe registrar en `source_map.md`: referencia, versión descargada, hash del PDF, sección aplicada, función propia correspondiente y diferencias introducidas. Verificar la licencia antes de reutilizar código; conservar atribución. Si no puede acceder al código de N3, implementar a partir de la descripción publicada y declarar que no se validó contra la implementación de los autores. No afirmar que se reprodujeron sus benchmarks.

### 0.2 Instrucción lista para copiar a Codex

> Trabaja en vtomasv/nanochat-mlx y aplica ESPECIFICACIONES_NAVARRO_NANOCHAT_MLX.md. Implementa la infraestructura común y E1, E2 y E3 en módulos aislados, con las referencias N1–N4 y las restricciones del documento. Mide primero el comportamiento actual. Ejecuta los tests reales de MLX en el Mac, el piloto y la confirmación que corresponda. No sustituyas algoritmos ni cambies umbrales para obtener un resultado positivo. Entrega código, comandos reproducibles, datos crudos, informe antes/después y un veredicto por experimento. Si falta hardware o datos, completa lo implementable y entrega un estado BLOCKED o INCONCLUSIVE preciso, sin inventar mediciones. No publiques ni sobrescribas checkpoints existentes.

## 1. Auditoría del repositorio que debe guiar la implementación

Enlaces fijados a la revisión auditada:

| Archivo existente | Punto de integración y observación |
|---|---|
| [gpt.py](https://github.com/vtomasv/nanochat-mlx/blob/b54b9fc139f455a9a5e60dc9a688ca9dbdb22944/nanochat_mlx/gpt.py) | `MLP.__call__`, `GPT.__call__`, `norm`, `has_ve`. El MLP usa ReLU². Hay embeddings de valores, escalas residuales y salida con softcap. |
| [optim.py](https://github.com/vtomasv/nanochat-mlx/blob/b54b9fc139f455a9a5e60dc9a688ca9dbdb22944/nanochat_mlx/optim.py) | `MultiOptimizer.update`, `_adamw_step`, `_muon_step`, `state`. Preservar exactamente sus ecuaciones y grupos. |
| [train.py](https://github.com/vtomasv/nanochat-mlx/blob/b54b9fc139f455a9a5e60dc9a688ca9dbdb22944/nanochat_mlx/train.py) | `build_model`, acumulación de gradientes y funciones de checkpoint. Usar ejecutor experimental con directorios propios. |
| [dataloader.py](https://github.com/vtomasv/nanochat-mlx/blob/b54b9fc139f455a9a5e60dc9a688ca9dbdb22944/nanochat_mlx/dataloader.py) | `dataloader_bos_bestfit`. Su estado no serializa todo el buffer ni el cursor dentro del lote documental: no basta para repetición exacta. |
| [engine.py](https://github.com/vtomasv/nanochat-mlx/blob/b54b9fc139f455a9a5e60dc9a688ca9dbdb22944/nanochat_mlx/engine.py) | `KVCache`, `Engine.generate`. Mantener ventanas y offsets. Separar el motor del tiempo de herramientas y de impresión. |
| [eval.py](https://github.com/vtomasv/nanochat-mlx/blob/b54b9fc139f455a9a5e60dc9a688ca9dbdb22944/nanochat_mlx/eval.py) | `evaluate_bpb`: solo reportar BPB si hay longitudes de tokens y denominador válido. NLL por token es obligatoria. |
| [common.py](https://github.com/vtomasv/nanochat-mlx/blob/b54b9fc139f455a9a5e60dc9a688ca9dbdb22944/nanochat_mlx/common.py) | Memoria y `NANOCHAT_BASE_DIR`. Los helpers pueden devolver cero al fallar: el benchmark debe devolver `null` y error. |
| [scripts/train.py](https://github.com/vtomasv/nanochat-mlx/blob/b54b9fc139f455a9a5e60dc9a688ca9dbdb22944/scripts/train.py) | Flags existentes. No asumir un flag `--dtype` ni `--seed`: agregarlos al ejecutor experimental. |

Otros hechos de auditoría que no deben convertirse en confusores: las dependencias no están fijadas exactamente; varios tests usan dobles de prueba; el entrenamiento puede borrar checkpoints anteriores; la evaluación BPB con `eval_tokens=0` requiere resolver explícitamente un valor positivo en el ejecutor. La generación puede terminar antes del máximo de tokens.

Leer `AGENTS.md` si existe en el checkout y `CLAUDE.md`. En la revisión consultada se prescribe `uv`. Si HEAD cambió, guardar diff respecto del SHA auditado y actualizar este mapa antes de medir. No ejecutar `reset --hard` sobre trabajo del usuario.

## 2. Infraestructura común: contrato de implementación

### 2.1 Entregables que Codex debe crear en el repositorio

Crear, sin sustituir los entrypoints ordinarios:

```text
nanochat_mlx/experiments/
  __init__.py
  config.py                  # validación de perfiles y manifiestos
  replay.py                  # datos y checkpoints reproducibles
  measure.py                 # tiempo, memoria y contexto del sistema
  metrics.py                 # NLL, errores, ratios e intervalos
  bitmap.py                  # E1
  activation_tape.py         # E1 y sus controles
  grammar_matrix.py          # codec compartido E2/E3
  staged_optimizer.py        # E2 y control sin compresión
  grammar_linear.py          # E3
  reporting.py
scripts/navarro_experiments.py
configs/navarro/{smoke,primary,confirm}.json
tests/test_navarro_*.py
docs/navarro/{source_map,implementation,runbook}.md
```

Estos nombres son interfaces a crear, no funciones que ya existan. Se permiten archivos adicionales para kernels Metal o extensión C++; documentar su compilación local. Los tests de integración deben usar MLX real; un test omitido por falta de MLX no equivale a uno aprobado.

Cada ejecución produce `experiments/navarro/results/<experiment>/<run_id>/`:

```text
manifest.json                # configuración efectiva inmutable y hashes
environment.json             # hardware, OS, versiones, límites
source_map.md
correctness.json
events.jsonl                 # sucesos, tiempos, fallback y errores
steps.csv                    # medidas por actualización o token
tensors.csv                  # estructura y tamaño por tensor
summary.json
report.md                    # comparación y conclusión legible
commands.sh                  # comandos exactos de esa ejecución
artifacts.sha256
```

Separar checkpoints grandes de resultados pequeños y usar rutas absolutas en el manifiesto. No añadir datos o pesos al git. El informe debe enlazar sus datos crudos relativos al directorio del run.

### 2.2 CLI a implementar

```bash
uv run python -m scripts.navarro_experiments preflight --config configs/navarro/primary.json
uv run python -m scripts.navarro_experiments prepare --config configs/navarro/primary.json
uv run python -m scripts.navarro_experiments baseline --config configs/navarro/primary.json
uv run python -m scripts.navarro_experiments run --experiment E1 --stage pilot --config configs/navarro/primary.json
uv run python -m scripts.navarro_experiments run --experiment E2 --stage pilot --config configs/navarro/primary.json
uv run python -m scripts.navarro_experiments run --experiment E3 --stage pilot --config configs/navarro/primary.json
uv run python -m scripts.navarro_experiments run --experiment E1 --stage confirm --config configs/navarro/primary.json
uv run python -m scripts.navarro_experiments report --results experiments/navarro/results
uv run python -m pytest tests/ -v
```

`confirm` admite igualmente E2 o E3. `run` debe verificar hashes y bloquear un resultado confirmatorio si faltan la referencia o la corrección. `--dry-run` imprime trabajos, estimación basada en piloto y espacio requerido; no inventa tiempos. `--resume-run` conserva manifiesto, estado, semilla y cursor. Volver a ejecutar un run completo debe ser idempotente: crear otro run explícito o devolver sus resultados, nunca mezclar filas.

### 2.3 Perfiles cerrados y límites

Valores por defecto propuestos, sujetos solo a una enmienda registrada antes de confirmación:

| Campo | Smoke | Principal | Confirmación de escala, solo si procede |
|---|---:|---:|---:|
| depth | 4 | 8 | 12 |
| aspect_ratio / head_dim | 64 / 128 | 64 / 128 | 64 / 128 |
| vocabulario | tokenizer real existente | mismo tokenizer | mismo tokenizer |
| sequence_len | 128 | 1024 | 1024 |
| window_pattern | L | SSSL | SSSL |
| device_batch_size | 2 | 4 | 4 |
| total_batch_size en tokens | 256 | 8192 | 8192 |
| acumulación | 1 | 2 | 2 |
| dtype principal | FP32 | FP32 | FP32 |
| semillas confirmatorias | 17 | 17, 29, 43 | 17, 29, 43 |

El FP32 evita atribuir beneficios a un cambio de precisión. Otros dtypes son exploratorios separados. E3 usa exactamente el dtype del checkpoint elegido y el mismo en todas sus variantes. El vocabulario y las matrices de embedding pueden dominar el tamaño: medir parámetros reales, no copiar estimaciones del README.

Con vocabulario 32768, el cálculo analítico del código auditado da aproximadamente 36,7M, 125,8M y 286,3M parámetros para depth 4, 8 y 12 respectivamente. Son estimaciones estructurales, no mediciones de velocidad; contrastar con `num_scaling_params()` y conteo del árbol real.

Las trayectorias de este protocolo prueban corrección y eficiencia durante aprendizaje temprano; no demuestran convergencia ni calidad competitiva de un modelo de lenguaje. Si un experimento resulta positivo, una trayectoria más larga y un checkpoint maduro constituyen validaciones posteriores, con presupuesto y protocolo propios.

Optimizador principal: `MultiOptimizer` existente con defaults actuales, incluidos escalados que aplica `train.py`; registrar todos los valores efectivos. `num_iterations=1000`, `target_param_data_ratio=12`, `warmup_ratio=0`, `warmdown_ratio=0.5`, `final_lr_frac=0`. No cambiar un schedule al reanudar. No usar auto-batch. Para la configuración principal cada trayectoria ve 8.192.000 tokens. Numerar actualizaciones de 0 a 999; checkpoint k significa k actualizaciones terminadas y la siguiente usa step=k. Guardar también checkpoints 400 y 900 para medir ventanas dentro del schedule original, sin prolongarlo después del paso 999.

Piloto de ingeniería: semilla 101, 100 pasos, separado de las semillas confirmatorias. Hacer primero smoke y luego piloto principal. Establecer un límite de memoria de 80 GiB, o el menor valor que permita el límite recomendado detectado, igual para todos los brazos; no interpretar 128 GB como memoria completamente libre. Registrar swap y abortar comparaciones con incremento sostenido superior a 1 GiB. Presupuesto operativo inicial: 24 horas de cómputo por experimento después del piloto. Si una confirmación no cabe, registrar `INCONCLUSIVE_RESOURCE_BUDGET`; no reducir pasos o semillas a mitad y presentar el resultado como confirmatorio.

### 2.4 Datos, semillas y estado inicial

1. Reutilizar los shards y tokenizer del usuario si existen. Registrar origen, revisión cuando esté disponible, hashes y split. Si faltan, usar los comandos de preparación documentados en este repo; registrar archivos realmente obtenidos. No importar otro corpus como elección implícita.
2. Crear una cinta de lotes con el dataloader original: entradas y targets exactos, máscaras y hashes. Materializar 1000 actualizaciones más los lotes de medición necesarios. Almacenar índices en formato entero suficiente y convertir a `int32` al alimentar MLX.
3. Mantener validación fija y separada: 128 secuencias de 1024 tokens para el perfil principal; 32 de ellas para calibración de ingeniería, 96 reservadas al informe. Si el corpus no lo permite, documentar bloqueo o enmienda previa. Comprobar duplicados exactos de documentos entre splits cuando se disponga de documentos; declarar los límites de esta comprobación.
4. Usar semilla explícita de Python, NumPy y MLX antes de inicializar. Para cada semilla guardar checkpoint inicial completo y hash. Todos los brazos de esa semilla parten de esos bytes.
5. Guardar cursores de cinta y estado RNG. No confiar en el resume parcial del dataloader original para una comparación pareada.
6. Registrar si se repite corpus por épocas, número de documentos, tokens únicos aproximados si es calculable y tokens procesados. No confundir 8M tokens procesados con 8M tokens distintos.
7. No medir solo step 0: algunas proyecciones se inicializan a cero. Para perfil de compresión usar estados de pasos 100, 500 y 1000 entrenados de la referencia; step 0 queda como control diagnóstico.

### 2.5 Línea base actual y controles comunes

Medir dos referencias antes de cada intervención:

- `repo_native`: código matemático y flujo de entrenamiento/inferencia auditados. Agregar solo instrumentación externa y directorios seguros. Conservar una ejecución del CLI original como comprobación de equivalencia.
- `harness_native`: mismo cálculo dentro del ejecutor de cinta y evaluación común. Comparar resultados y costos con `repo_native`; si el harness cambia la aritmética o el costo sustancialmente, explicarlo y arreglarlo antes de ensayar compresión.

El entrenamiento con cinta separa el cuello de botella de tokenización. Reportar throughput del kernel/paso con cinta y un piloto de 100 pasos con dataloader original en cada variante. Este último se etiqueta `pipeline`, incluye preparación de lotes y no mezcla sus cifras con `replay`.

### 2.6 Medición temporal y de memoria

MLX es perezoso. No cronometrar solo la creación del grafo. Usar reloj monotónico de alta resolución y evaluar salidas, gradientes, parámetros actualizados y estados antes de detener el reloj. Sincronizar el dispositivo con la API disponible. Verificar APIs en la versión instalada, sin asumir que la mínima del pyproject las contiene.

Registrar separadamente: carga, conversión, compilación inicial, forward, backward, actualización, codificación/decodificación, I/O de checkpoints, evaluación y duración total. Para el veredicto usar **tiempo de actualización completo**, con todos los costos de la intervención. Las fases instrumentadas pueden solaparse en MLX: no sumar tiempos si no se midieron con barreras explícitas. Ejecutar el perfil detallado separado del timing principal y declararlo.

Medir calentamiento en una copia descartable; después recargar el estado inicial de la ventana. Calentar no puede avanzar silenciosamente el entrenamiento medido. Registrar tiempo frío además del tiempo estable.

Memoria obligatoria:

- `mlx_active_peak_bytes`: pico asignado por MLX, reiniciado cuando la API lo permita.
- `mlx_cache_bytes` y política de caché, separadas.
- RSS máximo del proceso y del conjunto padre/hijos si hay auxiliares, en bytes.
- Bytes de cada representación, capacidad asignada, diccionarios, reglas, índices, temporales y copias CPU.
- Memoria y tiempo de construcción/conversión, además de ejecución estable.

En macOS usar `resource.getrusage` con su unidad documentada y muestreo de RSS, por ejemplo cada 50 ms; para extensiones/subprocesos registrar el máximo concurrente del conjunto, no sumar picos de momentos distintos. No sumar RSS y memoria Metal como si fueran memorias físicas disjuntas. Registrar ambas; memoria unificada exige contabilizar buffers del host. Si una API no funciona, usar `null` con motivo, nunca cero.

Cada brazo se ejecuta en proceso nuevo. Igual política de caché, límites, energía y concurrencia. No vaciar cachés por paso en un solo brazo. Registrar modelo de chip, núcleos, RAM, macOS, Python, MLX, compilador, `uv.lock`, modo de energía, batería/corriente y condiciones térmicas disponibles. Medir conectado a corriente y sin otras cargas intensivas; registrar incidencias.

### 2.7 Estadística y criterios de calidad comunes

Dos tipos de repetición, sin confundirlos:

- **Rendimiento:** 5 pares de procesos por comparación desde cada checkpoint nativo 400 y 900 de la semilla 17. Orden AB/BA repartido 3/2 en la primera ventana y 2/3 en la segunda, aleatorizado con semilla 20260913. Medir 100 actualizaciones por proceso: steps 400–499 o 900–999, con los lotes y schedules correspondientes. Cada brazo comienza con los mismos bytes de parámetros y estados, convertidos a su representación antes de la ventana; medir esa conversión aparte. La unidad de repetición es el proceso/ventana, no cada token ni cada paso. Reportar ambas ventanas por separado y exigir los criterios en ambas; no elegir retrospectivamente la favorable.
- **Aprendizaje:** tres trayectorias completas de 1000 pasos, una por semilla, por brazo necesario. No contar ventanas de timing como semillas nuevas. No añadir semillas después de examinar el resultado para lograr aprobación: una ampliación se registra como un nuevo estudio.

Reportar mediana, p95 y cocientes pareados. Para el tiempo principal calcular primero el promedio de las 100 actualizaciones por proceso; el estimador entre pares es la media geométrica de sus cocientes experimental/control. Para memoria usar el cociente de los picos de cada par. Intervalos percentiles del 95% mediante bootstrap pareado de 10.000 remuestreos de procesos, semilla 20260913; mostrar también todos los puntos. Con cinco pares son intervalos exploratorios frágiles y pueden persistir dependencias térmicas entre procesos: conservar esas limitaciones. Para no inferioridad de NLL usar diferencias por semilla, media y límite superior unilateral del 95% con t de Student, 2 grados de libertad. Si las tres diferencias son exactamente cero, registrar ese caso degenerado y límite cero. No afirmar equivalencia por obtener p>0,05. Estos criterios son decisiones prácticas conjuntas; no hacer afirmaciones de significancia familiar de los tres experimentos sin un análisis adicional preregistrado.

NLL se calcula sumando cross-entropy de todos los tokens válidos y dividiendo por su número. Perplejidad = exp(NLL); no comparar perplejidades de tokenizadores distintos. BPB es secundaria y requiere denominador positivo verificable.

Margen de no inferioridad fijado: 0,01 nats/token de NLL final contra el control correspondiente. Aproximadamente equivale a 1% relativo de perplejidad. El límite superior unilateral debe quedar por debajo del margen. Si los datos no permiten decidir, `INCONCLUSIVE`, no `SUCCESS`.

Pruebas numéricas previas, todas FP32 salvo prueba específica de codec:

- Ida/vuelta de representación: igualdad bit a bit del dominio declarado.
- Operador local: `atol=1e-5`, `rtol=1e-4`; registrar también error máximo y relativo L2.
- Gradientes de cada parámetro: `atol=1e-5`, `rtol=1e-3`, relativo L2 ≤1e-3; comprobar parámetros no nulos individualmente, no solo una media global.
- Logits de modelo: `atol=1e-4`, `rtol=1e-3`; diferencia NLL de un lote ≤1e-4.

Estos umbrales son decisiones del estudio. Si fallan, investigar error o orden de reducción; no ampliarlos después para aprobar. Una revisión de tolerancias exige nueva versión de protocolo y nuevas confirmaciones. No exigir que entrenamiento prolongado produzca pesos bit-idénticos cuando cambia el orden de reducciones.

### 2.8 Clasificación del resultado

- `SUCCESS`: pasa corrección, fidelidad al algoritmo, calidad y los criterios prácticos del ID.
- `FAILURE_CORRECTNESS`: estructura, gradientes o integración incorrectos; no demuestra que la hipótesis científica sea falsa.
- `FAILURE_PERFORMANCE`: implementación validada, confirmación completa, criterios prácticos incumplidos de manera concluyente.
- `NEGATIVE_SCREENING`: compresión/beneficio insuficiente en el piloto según regla previa; debe incluir medidas efectivas y acotar su alcance.
- `INCONCLUSIVE`: intervalos cruzan umbrales, corrida incompleta o variabilidad excesiva.
- `BLOCKED`: no se dispone de hardware, datos, API o dependencia necesaria; detallar acción para desbloquear.

Guardar veredictos separados para representación, operador y modelo completo. Una reducción de archivo no puede convertirse en éxito de entrenamiento. Un resultado en datos sintéticos no es éxito en nanochat-mlx.

## 3. E1 — Entrenamiento con activaciones compactas del MLP

### 3.1 Hipótesis y atribución

H1: guardar solo los valores positivos necesarios para derivar ReLU², con bitmap y acceso por `rank` basado en N1, reduce memoria de entrenamiento con costo aceptable y aprendizaje equivalente. La aplicación a activaciones, la cinta y el backward de abajo son diseño de este experimento; N1/N2 no los evalúan.

Objeto: `P = maximum(Z, 0)`, donde `Z = U @ W1.T`, `H = P*P` y `Y = H @ W2.T`. `U` es la entrada normalizada al MLP. Guardar P, no reconstruirlo con `sqrt(H)`: esa raíz puede introducir error adicional y perder información por redondeo.

Para aplanado de N elementos, dtype de b bytes y k positivos, estimación propia:

`bytes_compact = b*k + ceil(N/32)*4 + rank_index_bytes + headers + allocation_padding`.

Con k/N=0,5 en FP32, el payload de valores más bitmap es aproximadamente 53,125% del P denso, antes de índices. Esto no predice el ahorro global: atención, logits, pesos y estados permanecen.

### 3.2 Representación a implementar

`PackedPositive` contiene forma, dtype, longitud lógica, bitmap en `uint32`, índice de rank, buffer de valores y tamaños físicos. Orden row-major. Usar prefijos **exclusivos**: `rank(i)=sum(bitmap[0:i])`; posición de P[i] en values = rank(i) si su bit vale 1. Definir y probar límites 0 y N. Muestreo principal cada 256 bits; calibración permitida 128/256/512 solo en piloto, con elección fijada antes de confirmación.

La adaptación de N1 a palabras de 32 bits debe mantener el muestreo y recorrido por popcount. No guardar un offset int32 por elemento, pues anularía parte del beneficio. La construcción puede usar prefijos temporales, que deben liberarse y contarse en el pico.

Tamaño de `values` físicamente ajustado a k. Una vista sobre un buffer denso de N elementos no constituye compresión residente. Dos pasadas —conteo y escritura— son aceptables; cualquier sincronización host para conocer k se incluye. No usar truncación, umbral aproximado, top-k ni reserva fija insuficiente. Si el tamaño total compacto no es menor que 95% del denso, usar variante RAW para ese tensor, con razón y costo del intento registrados.

Dominio normal: P finito no negativo. Registrar y abortar aprendizaje con NaN/Inf; tests de codec deben preservar o rechazar explícitamente no finitos. Conservar valores positivos bit a bit y definir cero canónico; comprobar que el backward en cero es cero. No afirmar reconstrucción de Z negativo: no es lo almacenado ni lo necesario.

### 3.3 Integración recomendada: cinta explícita y backward por segmentos

La forma dinámica de `values` puede entrar en conflicto con funciones trazadas de MLX. Para que el experimento tenga una vía ejecutable, implementar primero una cinta imperativa con puntos explícitos de materialización y VJP local. No depender de que exista un hook de activaciones guardadas equivalente al de PyTorch.

Forward:

1. Calcular embedding de tokens y `x0 = norm(wte(ids))`.
2. Para cada bloque i, guardar su entrada `Xin`. Calcular `S = resid_lambdas[i]*Xin + x0_lambdas[i]*x0` y `A = S + attention(norm(S), ve_i)`, con las máscaras originales. Materializar A; descartar el grafo de atención manteniendo solo valores necesarios.
3. Calcular `U=norm(A)`, Z, P, H y salida `A + H@W2.T`. Guardar A y P en la representación elegida. U se puede recalcular desde A. Materializar la salida y soltar Z, H, U y P denso en el brazo compacto.
4. Conservar `x0`, entradas Xin/A y salida final necesarias para el backward; no guardar logits de todos los pasos después de consumirlos.

Backward:

1. VJP del tramo final original —norm, lm_head, recorte de vocabulario, softcap y pérdida— para obtener gradiente de la última salida y de lm_head.
2. Recorrer bloques al revés. Reconstruir P de un solo bloque y calcular U desde A. Con G como adjunto de la salida del bloque:

```text
H       = P * P
dW2     = G.T @ H
dH      = G @ W2
dZ      = (2 * P) * dH
dW1     = dZ.T @ U
dU      = dZ @ W1
dA      = G + VJP_norm(A, dU)
```

Aplanar B y T para estos productos y restaurar formas. Se permite empezar con un bloque completo reconstruido; el mínimo obligatorio es no expandir todos los bloques simultáneamente. Liberar temporales y payload consumido tras materializar sus gradientes.

3. Recalcular con VJP local la función que produce A a partir de Xin, x0, las dos escalas, parámetros de atención y value embedding. Acumular gradientes de todas las rutas, incluidas las contribuciones directas a x0. Pasar cada parámetro aprendible explícitamente al VJP; capturarlo como constante puede eliminar su gradiente.
4. Al terminar el bloque 0, sumar la ruta secuencial a las rutas directas de x0 y propagar por embedding+norm. Acumular correctamente tokens repetidos. Reconstruir el árbol de gradientes original y entregarlo a `MultiOptimizer` sin modificarlo.

Esta cinta vuelve a calcular atención durante backward. Es un cambio de ejecución que debe aislarse con controles; no atribuir sus beneficios al bitmap. Mantener referencias inmutables de los pesos hasta terminar el backward. Materializar y desacoplar residuales del grafo mediante la API de stop-gradient disponible, liberando referencias; demostrar con auditoría de memoria que no se retiene el forward denso por referencias Python o lazy nodes. La cinta sustituye la función global que devuelve pérdida y gradientes: no envolver todo su forward con `nn.value_and_grad` y esperar que desaparezcan las activaciones originales. Usar autodiff solamente en los segmentos locales indicados.

Una versión futura con `mx.custom_function` o primitiva C++ puede sustituir la cinta solo después de pasar los mismos tests. No es necesaria para completar la versión 1.

### 3.4 Brazos obligatorios

| Brazo | Qué guarda | Para qué sirve |
|---|---|---|
| `native` | Autodiff original | Rendimiento actual del repo |
| `tape_dense` | Cinta anterior con P denso | Aísla el efecto de reprogramar y recalcular atención |
| `tape_recompute` | Misma cinta, recalcula P desde A y W1 al hacer backward | Comprueba si guardar una estructura aporta frente a recomputar |
| `tape_bitmap` | Misma cinta con PackedPositive/RAW | Intervención E1 |

Todos entrenan los mismos parámetros con la misma cinta de lotes. `tape_recompute` es un control de ejecución elemental, no una propuesta científica externa. Registrar número de multiplicaciones adicionales por brazo. Prohibido comparar únicamente `tape_bitmap` contra `tape_dense` y omitir la referencia original.

### 3.5 Pruebas obligatorias

- Bitmaps vacíos, todos cero/uno, alternados, longitudes 31/32/33 y 255/256/257, último word parcial y overflow de contadores prevenido.
- Valores positivos subnormales, ceros, magnitudes grandes finitas y tensor no contiguo. Bit-exact del P reconstruido para el dominio admitido.
- MLP pequeño con W2 aleatorio no nulo; comparar forward y dU/dW1/dW2 con autodiff. No usar solo W2 inicial cero, pues ocultaría errores.
- Gradiente numérico por diferencias centrales en una función escalar pequeña, lejos de cambios de signo; barrer epsilon 1e-2, 1e-3, 1e-4 y contrastar con autodiff. No esperar precisión FP64 en la GPU.
- GPT mínimo con todas las funciones activas: value embeddings, escalas, ventanas S/L, vocabulario no múltiplo de 64 y targets -1. Verificar cada grupo de parámetros y dos microbatches acumulados.
- Comparación de 10 actualizaciones desde el mismo estado, luego smoke y entrenamiento principal. Reanudación después de 10 pasos debe coincidir con ejecución continua dentro de tolerancia.
- Medir payload vivo antes/después de guardar y liberar; comprobar que liberar el compacto no deja una copia densa sobreviviente.

### 3.6 Mediciones y éxito

Por capa/paso: fracción positiva, bytes de P denso, bytes de valores/bitmap/rank, capacidad física, tasa RAW, tiempo pack/unpack y máximo de residuales vivos. Medir especialmente pasos 100, 500 y 1000.

Éxito E1 requiere corrección y no inferioridad, además de:

1. Límite superior del IC95 del cociente de pico MLX de `tape_bitmap/native` ≤0,95; RSS del proceso no empeora más de 5%.
2. Límite superior del IC95 del cociente tiempo/paso `tape_bitmap/native` ≤1,10.
3. Contra `tape_dense`, reducción de bytes de P vivo ≥25% y reducción de pico total MLX ≥3%, sin degradación temporal mayor a 10%.
4. `tape_recompute` no domina al compacto: si usa igual o menor pico y tiempo dentro de la incertidumbre, no afirmar que la estructura es la mejor opción. Reportar un resultado mixto o negativo de valor incremental.

Piloto negativo si a pasos entrenados el payload de P no baja al menos 10%, o la penalización del paso supera 2x tras verificar que no es un bucle Python por elemento. En tal caso conservar la comparación completa de al menos 50 pasos principales y emitir `NEGATIVE_SCREENING`. No escalar automáticamente.

## 4. E2 — Entrenamiento con estados del optimizador comprimidos sin pérdida

### 4.1 Hipótesis y alcance

H2: una representación basada en N3 para los estados inactivos entre pasos reduce memoria residente y máxima, manteniendo exactamente las ecuaciones del optimizador. No es un nuevo Adam ni un nuevo Muon. Es una política de almacenamiento y reconstrucción aplicada al entrenamiento.

Objetos: `adam_state[path].m`, `.v` y `muon_state[path]`. Contadores t y configuración permanecen exactos. Los vectores pequeños se mantienen RAW; principal aplica a matrices de al menos 65.536 elementos. No comprimir pesos ni activaciones en E2.

Expectativa previa: **alta probabilidad de resultado negativo en estados flotantes densos poco repetitivos**. No cambiar a 8 bits ni redondear para mejorar la tasa. Medir esa limitación es parte del objetivo.

### 4.2 Codec compartido con E3

Crear `GrammarMatrix` con modos `RAW`, `re_32`, `re_iv`, formas/dtypes/versiones y hashes. Implementar N3 §2 y §4, con reglas que respeten filas y símbolos que distingan valor y columna. Representar la identidad de valores por bits originales, no por comparación flotante aproximada. Diccionario incluye dtype y endianness. Conservar -0 y casos especiales mediante RAW o registros explícitos; no eliminar -0 como si fuera +0 en un codec bit-exact.

Índices y reglas deben tener anchos suficientes; comprobar el posible overflow de la codificación valor-columna y usar remapeo seguro o ancho mayor documentado. `re_iv` empaqueta físicamente enteros al ancho necesario, no los deja como listas Python. Un header identifica terminales y no terminales sin colisión. No permitir ciclos, símbolos fuera de rango o expansión fuera de forma.

El compresor debe ser compilado para mediciones confirmatorias. Un RePair ingenuo que vuelve a recorrer toda la secuencia en Python sirve solo como oráculo de fixtures pequeños. Registrar reglas, símbolos, alturas, valores distintos y coste de construcción. Adaptar el código de N3 cuando sea posible; fijar su revisión. Los detalles normativos de construcción son los de N3, no un diccionario genérico vendido como RePair.

Bloques de almacenamiento principales: grupos de 128 filas completas, independientes; calibrar 32/128/512 filas en piloto. Columna dentro del bloque conserva su identidad original. La división en bloques y fallback son elecciones de este protocolo. No dividir una matriz de Muon para aplicar su ortogonalización por trozos: **se reconstruye el buffer completo de ese parámetro** para su actualización original.

API nueva mínima:

```text
encode(array, variant, rows_per_block) -> GrammarMatrix
decode_exact(handle, device) -> array
resident_bytes(handle) -> accounting
serialize(handle, destination)
deserialize(source) -> GrammarMatrix
matvec(handle, x, transpose=False) -> y       # usada por E3
```

Seleccionar representación por bloque con una regla previa: construir la gramática una vez, calcular tamaños físicos de re_32/re_iv y elegir el menor que ocupe ≤95% de RAW incluyendo header/índices; empate entre comprimidos favorece re_32. Si ninguno cumple, usar RAW. No ensayar una variante cuyos índices desborden su ancho. Contar tiempo y pico del intento aun cuando se descarte. RAW es un brazo honesto, no éxito del algoritmo. Registrar porcentaje de bloques y de bytes realmente comprimidos.

### 4.3 Integración con MultiOptimizer

Crear una tabla separada de handles. No dejar simultáneamente los handles comprimidos y copias permanentes en `adam_state`/`muon_state`.

Para cada actualización:

1. Materializar gradientes completos del modelo según la referencia y capturar la configuración efectiva del paso.
2. En el mismo orden de parámetros que el control, recuperar el estado de un parámetro. Decodificar sus bloques a uno o dos arrays, según sea Muon o AdamW.
3. Ejecutar las ecuaciones originales, con los mismos dtypes, orden de operaciones, momentum, bias correction, eps y weight decay. Mantener completa la operación Newton–Schulz de cada matriz.
4. Materializar nuevo parámetro y nuevo estado; codificar el estado nuevo, verificar su integridad en modo debug y liberar el estado denso transitorio y el handle anterior cuando no sean usados.
5. Aplicar los parámetros actualizados sin que un gradiente de otro parámetro se calcule con pesos mezclados de pasos distintos. Todos los gradientes del paso ya deben estar calculados. Liberar gradientes solo con la misma política en el control staged.
6. `state` debe exponer solo arrays realmente necesarios para evaluación, sin decodificar todos los handles al llamar `mx.eval`. No acumular grafos entre pasos.

La primera versión puede usar CPU compilada para codificar/decodificar y MLX para actualizar. La copia y sincronización se cronometran. El uso de memoria del host se incluye: mover datos fuera del allocator Metal no demuestra ahorro de memoria física en Apple Silicon.

Checkpoint experimental con versión, handles y metadatos; ofrecer exportación a estados densos del repo, parámetro por parámetro. La reanudación experimental debe restaurar t, momentum, LR schedule, dtype, cursor y RNG. No modificar el formato ordinario silenciosamente.

### 4.4 Brazos obligatorios

| Brazo | Estado entre pasos | Control |
|---|---|---|
| `native` | MultiOptimizer original | Rendimiento actual |
| `staged_raw` | Arrays RAW con la misma política por parámetro, materialización y residencia CPU/MLX que el compacto | Aísla staging y cambio de planificación |
| `staged_grammar` | RAW/re_32/re_iv elegido por regla fijada | Intervención |

`staged_raw` debe usar el mismo formato contenedor y las mismas fronteras de copia cuando sean comparables, pero sin trabajo de compresión. No atribuir una mejora de staging a RePair. Ejecutar forward/backward nativos en todos los brazos; E1 está deshabilitado.

### 4.5 Pruebas y puertas de decisión

- Codec con filas vacías, filas repetidas, valores repetidos en columnas distintas, matrices totalmente nulas, una regla usada múltiples veces, tamaños no alineados, enteros bit-packed cruzando palabras y diccionarios grandes.
- Prueba bit-exact para FP32 y para patrones IEEE especiales manejados por fallback. Fixtures aleatorios de alta entropía deben fallar compresión de forma controlada, sin perder valores.
- Alimentar los tres optimizadores con gradientes idénticos predeterminados durante 20 pasos. Decodificar y comparar **bit a bit** estados y parámetros frente a `staged_raw` si se preservó exactamente el mismo grafo aritmético; si el backend introduce diferencias, documentarlas y aplicar tolerancias comunes además del codec bit-exact.
- Casos Muon con matrices altas y anchas; AdamW con t>1; LR y weight decay variables. Prohibido sustituir la fórmula de Muon por otra considerada más correcta: sería otro experimento.
- Entrenamiento integrado de 50 pasos principales como mínimo después de tests; comparación de checkpoint reanudado y continuo.

Puerta antes de confirmación: perfilar y codificar estados reales de la referencia en pasos 100, 500 y 1000. Si los bytes totales elegibles, con fallback, no bajan al menos 10% en dos de esos tres estados, o el costo del paso es >2x, cerrar con `NEGATIVE_SCREENING` después del piloto integrado. No gastar 1000 pasos de tres semillas para demostrar una ausencia de repetición ya medida. Conservar todos los ratios y el rendimiento obtenido, aunque sea peor.

Si pasa esa puerta, ejecutar confirmación común. Durante cada paso registrar bytes al inicio, máximo durante decode/update/encode y bytes al final; el estado persistente pequeño no oculta un pico de reconstrucción grande.

### 4.6 Éxito E2

Corrección y no inferioridad obligatorias, más:

1. Reducción ≥20% de bytes totales persistentes del optimizador contra `staged_raw` en pasos entrenados.
2. Límite superior IC95 del cociente RSS máximo `staged_grammar/native` ≤0,90; pico MLX no aumenta más de 5%.
3. Límite superior IC95 del tiempo/paso `staged_grammar/native` ≤1,10.
4. Contra `staged_raw`, ventaja de memoria total de al menos 5% y tiempo no peor en más de 10%. Si toda la ventaja se obtiene ya con staging, no hay éxito atribuible a la estructura.

Reportar aparte archivo de checkpoint, estado persistente, pico global y tiempo. Un archivo menor con entrenamiento más caro es un resultado parcial, no SUCCESS bajo esta definición.

## 5. E3 — Inferencia con pesos comprimidos y producto directo

### 5.1 Hipótesis y objeto

H3: ciertas matrices de un checkpoint **entrenado** permiten almacenar y calcular mediante N3 con menor costo global que su representación densa. Se trabaja sobre los mismos pesos, sin pérdida de representación y sin cuantizar.

Elegir antes de evaluar: checkpoint nativo del paso 1000, semilla 17, perfil principal. Si existe un checkpoint del usuario más entrenado, añadirlo como validación externa y registrar metadatos; no reemplazar el principal después de ver los resultados.

Objetivo principal: `blocks[i].mlp.c_fc.weight` y `.c_proj.weight`. Las proyecciones de atención y lm_head quedan sin cambiar. No alterar embeddings, escalas, ReLU², softcap, KV cache ni ventanas. Cada `Linear` tiene W de forma `[out_features, in_features]`; para B=T=1, la operación requerida es `W @ x`.

### 5.2 Conversión y operador

Reutilizar el codec validado de E2; E3 puede ejecutarse independientemente implementando solo ese codec y sus tests. Congelar pesos y liberar matrices densas sustituidas después de la conversión.

Implementar producto de N3 §3/§4 mediante evaluación de reglas y suma de los símbolos de cada fila. **No llamar a decode_exact de toda la matriz durante cada token.** Para `re_iv`, leer reglas/símbolos empaquetados durante el recorrido; si se desempaquetan una vez, contar el tamaño residente expandido y etiquetar ese brazo por separado.

Contrato operativo, con notación propia: el terminal `(valor, columna)` aporta `valor*x[columna]`; una regla `r → a b` aporta `eval(a)+eval(b)`; la salida de una fila es la suma de sus símbolos finales. Los delimitadores no aportan valores ni pueden quedar dentro de una regla. En el producto transpuesto, distribuir el adjunto de cada fila por sus símbolos y propagarlo por el DAG en orden inverso, **sumando** contribuciones a reglas compartidas; cada terminal acumula `valor*adjunto` en su columna. Verificar estas recurrencias contra N3 antes de optimizarlas. La matriz se conserva sin pérdida; la reasociación de sumas puede alterar redondeos FP32, por eso el producto tiene tolerancias.

Fixture didáctico propio: para `W=[[2,3,0],[2,3,5]]`, los símbolos de las dos primeras columnas permiten una regla compartida que aporta `q=2*x[0]+3*x[1]`. Las salidas son `q` y `q+5*x[2]`. Una fila `[3,2,0]` no puede reutilizar esa regla solo porque contiene los mismos valores: las columnas importan. Este fixture comprueba corrección; su compresibilidad artificial no predice la de pesos aprendidos.

Implementación de referencia CPU compilada obligatoria. Extensión Metal: agrupar reglas por dependencias y procesar solo niveles válidos; contar metadata, sincronizaciones y scratch. No suponer que las reglas se pueden evaluar en paralelo sin respetar su DAG. Limitar scratch por bloques de filas y medir el máximo real. Implementar producto transpuesto para verificar la estructura y el producto adjunto, aunque el decode principal use solo el derecho.

Prefill necesita múltiples vectores. Procesarlos en microbloques de 16 tokens como elección inicial, usando la misma representación, para no multiplicar scratch por todo el prompt. La optimización de batched evaluation es una adaptación propia. Incluir su costo; no mantener permanentemente una copia densa solo para acelerar prefill sin contabilizarla.

La implementación CPU debe incluir todo movimiento MLX↔CPU y ejecución mixta al medir el modelo. Una mejora CPU contra un CPU lento no es una aceleración del Mac completo. El operador Metal es requerido para afirmar aceleración GPU; si no mejora, el estudio puede terminar con resultado negativo y evidencia del operador CPU.

### 5.3 Selección y controles

Aplicar a todas las matrices MLP el mismo ensayo de tamaño con bloques 128 filas, re_32/re_iv, regla ≤95% de RAW. Publicar tabla de **todas** las capas. No elegir retrospectivamente solo la capa ganadora.

Si hay capas candidatas, usar exclusivamente los 32 prompts/secuencias de calibración para elegir una de estas políticas:

- todas las capas que ahorran tamaño;
- solo capas que ahorran tamaño y cuyo microbenchmark incluye costo total no superior al denso.

Elegir la política con mejor tiempo total de calibración sujeto a no aumentar memoria, desempate por menor memoria, y congelar antes de usar los 96 casos reservados. Si ninguna capa queda seleccionada, la política adaptativa es nativa y debe declararse `NO_ELIGIBLE_LAYERS`, no éxito.

Brazos:

| Brazo | Función |
|---|---|
| `dense_native` | Referencia MLX real |
| `compressed_decode_dense` | Almacena compacto pero reconstruye para calcular: demuestra costo de una integración ingenua |
| `grammar_direct_cpu` | Producto directo compilado, con transfers incluidos |
| `grammar_direct_metal` | Producto directo GPU, si implementado y validado |

Mantener microbenchmark CPU denso como diagnóstico del algoritmo, no como referencia del producto final. Medir también la política adaptativa elegida; no mezclar sus cifras con las de conversión forzada de todas las capas.

### 5.4 Pruebas de corrección

- Reutilizar fixtures del codec y productos derecho/izquierdo contra NumPy FP64 como oráculo CPU en matrices pequeñas y MLX FP32 como comparación de ejecución.
- Incluir la misma cifra en columnas distintas: una gramática de valores sin columnas produciría resultados erróneos.
- Verificar `<Wx,y> ≈ <x,W.T y>` y filas vacías, terminales sueltos, reglas compartidas y límites de bloques.
- Toda matriz convertida se reconstruye una vez fuera del benchmark y se contrasta bit a bit con el checkpoint.
- Comparar logits y NLL en teacher forcing con prefijos idénticos. No usar secuencias ya divergentes para atribuir error a una capa aislada.
- Probar prompt 1/128/512 y generación incremental; offsets y ventanas conservados. Confirmar el caso batch 1 y el caso de cuatro secuencias independientes.
- Prefill principal de longitud ≤512 para `sequence_len=1024` evita cambiar semántica por un prompt que exceda la ventana corta. Probar esos límites explícitamente como test, no añadir una reparación del KV cache solo a un brazo.

### 5.5 Carga de trabajo y medición

Preparar 32 prompts por longitud 128 y 512 tokens: 64 prompts reservados procedentes de validación, disjuntos de los usados para calibrar. Registrar tokens y hashes. Para latencia, consumir exactamente 128 pasos de decode por prompt.

El benchmark de rendimiento usa una continuación fija de tokens válida para alimentar cada paso, con caché y cómputo de logits completos. Así todos los brazos procesan el mismo contexto y longitud. Esto es **teacher-forced decode**, no velocidad de generación libre; etiquetarlo así. Medir generación greedy aparte con EOS natural y registrar longitud efectiva, concordancia y divergencias.

Condición principal: batch 1, prompt 512, decode 128. Secundarias: prompt 128 y batch 4 con igual longitud. Si el motor no soporta prompts diferentes en batch sin modificación, implementarlo en el harness o ejecutar cuatro secuencias independientes y etiquetar esa modalidad; no llamarla batching paralelo.

Medir TTFT desde ids ya preparados, prefill, tiempo por token de decode (p50/p95), tokens/s, carga/conversión, pico MLX, RSS y bytes residentes de pesos. Separar tokenización e impresión. Deshabilitar ejecución de herramientas y penalidades en el benchmark sintético de latencia; mantener generación de usuario ordinaria sin cambios.

Cinco pares de procesos con mismos prompts y orden AB/BA repartido 3/2, aleatorizado con semilla 20260913. Para cada proceso medir todos los prompts. Calcular primero su latencia media por token y luego los cocientes pareados y su media geométrica, con el bootstrap de §2.7. Intervalos pareados por proceso; no tratar 128 tokens correlacionados como 128 ensayos independientes. Para diferencias NLL teacher-forced usar bootstrap pareado de 10.000 remuestreos por documento reservado, agrupando prompts del mismo documento; usar el percentil 95 como límite superior unilateral y reportar la limitación de un solo checkpoint principal.

### 5.6 Puerta negativa y éxito

Puerta previa: si ninguna matriz MLP entrenada logra ocupar ≤95% de RAW, no simular mejora cambiando precisión. Ejecutar aun así conversión forzada de una capa representativa —bloque central, `c_fc`—, microbenchmark y al menos 8 prompts de inferencia completa con esa capa, para documentar tamaño y tiempo logrados. Luego `NEGATIVE_SCREENING_NO_COMPRESSIBILITY`. Eso refuta esta representación en esa muestra y configuración, no toda compresión neuronal.

Si hay candidatas, confirmación y criterios conjuntos:

1. Corrección local y modelo, y límite superior unilateral IC95 de delta NLL <0,01 nats/token.
2. Límite superior IC95 del cociente de latencia media de decode directo/nativo ≤0,90 en la condición principal; p95 no peor en más de 5%.
3. Pico MLX y RSS no empeoran más de 5%; bytes totales residentes de pesos y estructuras del modelo disminuyen al menos 10%.
4. TTFT no empeora más de 10% y tiempo completo prefill+128 decode no empeora.

La exigencia simultánea impide declarar éxito por un kernel rápido que agranda el modelo o hace prohibitivo el prefill. Una mejora únicamente de memoria se reporta como tal, pero no satisface SUCCESS de E3.

Calcular punto de amortización de conversión: `tokens_break_even = conversion_seconds / (seconds_per_token_native - seconds_per_token_direct)` si el denominador es positivo. Si no, `never`. Aclarar que la fórmula supone régimen de decode estable; reportar también solicitudes completas porque la longitud del prompt importa.

## 6. Informe antes/después obligatorio

No llenar la tabla con cifras del README, del paper ni de otro Mac. Encabezado de cada tabla: SHA, hardware, dtype, perfil, corpus/checkpoint y modo de medición. Valores desconocidos: `null`.

| Medida | Actual nativo | Control intermedio | Intervención | Cambio relativo | IC / observaciones |
|---|---:|---:|---:|---:|---|
| Parámetros aprendibles | null | null | null | null | E1/E2 deben coincidir |
| Tokens procesados | null | null | null | null | Mismo presupuesto |
| NLL final | null | null | null | null | Diferencia absoluta también |
| Tiempo completo/paso o solicitud | null | null | null | null | Costos incluidos |
| Tokens/s | null | null | null | null | Definir tokens y modalidad |
| Pico MLX, bytes | null | null | null | null | No sumar a RSS |
| RSS máximo, bytes | null | null | null | null | Host y auxiliares |
| Bytes residentes del objeto | null | null | null | null | Índices y capacidad incluidos |
| Pack/unpack o conversión | null | null | null | null | Incluido en total pertinente |
| Porcentaje RAW/fallback | null | null | null | null | No ocultar capas omitidas |

Para reducción: `100*(1 - experimental/base)`. Para speedup: `base_time/experimental_time`. No usar ambas convenciones sin etiqueta.

`summary.json` debe contener como mínimo:

```json
{
  "experiment": "E1",
  "protocol_version": "1.0",
  "status": "not_measured",
  "repo_sha": "b54b9fc139f455a9a5e60dc9a688ca9dbdb22944",
  "hardware": null,
  "profile": null,
  "correctness_pass": null,
  "baseline": null,
  "controls": [],
  "intervention": null,
  "paired_ratios": null,
  "confidence_intervals": null,
  "quality_noninferiority": null,
  "algorithm_fidelity": null,
  "decision": null,
  "failure_reason": null,
  "limitations": [],
  "raw_files": []
}
```

El informe narrativo explica: hipótesis, qué proviene del artículo, adaptación, configuración, corrección, resultados antes/después, incertidumbre, veredicto, límites y comando de reproducción. Añadir curvas de NLL frente a tokens y frente a tiempo real, y memoria frente a paso, generadas desde CSV. Mostrar trazas negativas y OOM; no descartarlas silenciosamente.

## 7. Reglas de ejecución, revisión y cierre

1. Preflight y fijación de entorno. Si no hay Apple Silicon/Metal, ejecutar solo codec CPU y validación de formatos. No producir benchmarks de entrenamiento Mac en Linux ni usar los mocks como sustituto.
2. Registrar línea base antes de editar el cálculo; aislar cambios de instrumentación. Crear rama o worktree reversible respetando el checkout actual.
3. Implementar y validar infraestructura y fuentes. Cerrar fuente→función en `source_map.md`.
4. E1: bitmap y cinta; controles; smoke; piloto; confirmación si procede.
5. E2: codec, state store y staging; controles; piloto; confirmación si procede.
6. E3: reutilizar codec, operador directo, conversión y motor; piloto; confirmación si procede.
7. Hacer una revisión final de las condiciones de aceptación contra datos, no contra expectativas. Si una implementación no reproduce correctamente el algoritmo, corregirla dentro del presupuesto; no etiquetar ese defecto como fracaso de la literatura.
8. Entregar código ejecutable, instrucciones, resultados crudos y reportes. Resumir los tres veredictos sin extrapolar de un checkpoint, una forma de matriz o un chip a todos los LLM.

Checklist de completitud por experimento:

- [ ] Fuentes leídas, fijadas y atribuidas; adaptación explícita.
- [ ] Línea base actual real disponible o bloqueo documentado.
- [ ] Algoritmo correcto y memoria físicamente compacta.
- [ ] Controles que separan representación de reprogramación/recomputación.
- [ ] Tests MLX reales y replay verificable.
- [ ] Rendimiento antes y después, incluidos índices, construcción, copias y fallback.
- [ ] Criterios registrados antes de confirmación y aplicados sin cambios oportunistas.
- [ ] Resultado SUCCESS, negativo, inconcluso o bloqueo con alcance preciso.
- [ ] Comandos y artefactos suficientes para repetir.

No queda autorizada una migración de arquitectura ni una búsqueda ilimitada de variantes si las hipótesis fracasan. Un cierre negativo con evidencia es una entrega completa de la tarea científica.
