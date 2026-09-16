# Estructuras de datos compactas para entrenar y ejecutar nanochat-mlx

## Especificación de investigación, implementación y evaluación, versión 2.0

**Fecha:** 15 de septiembre de 2026. **Estado:** protocolo nuevo informado por evidencia histórica y un censo estructural ejecutado durante esta revisión. **Se implementó el ejecutor v2 de exploración; el estado de cada corrida se consulta en su registro, sin equiparar pilotos a confirmación.**

**Pregunta central:** ¿en qué objetos, operaciones y regímenes de uso una representación compacta aporta una mejora verificable al entrenamiento o la inferencia de este GPT, después de contabilizar construcción, índices, acceso, aritmética, memoria y calidad?

**Conclusión de partida:** no hay evidencia que justifique sustituir generalmente pesos o estados FP32 de nanochat por gramáticas. Sí hay una oportunidad investigable en activaciones con ceros exactos. Los productos sobre matrices congeladas, otros formatos y datos auxiliares requieren experimentos diferentes. El objetivo es una estrategia seleccionada por evidencia, que conserve operaciones densas donde sean superiores.

Repositorio: <https://github.com/vtomasv/nanochat-mlx>. HEAD revisado: **1f80132dbda17fd787234361bd2cfcbdd77326f7**. Referencia matemática histórica: **b54b9fc139f455a9a5e60dc9a688ca9dbdb22944**. El diff entre ambos no modifica gpt.py, optim.py, train.py, dataloader.py, engine.py ni eval.py. Los módulos experimentales sí evolucionaron.

Este documento sustituye el diseño futuro de v1; **no cambia sus criterios ni reclasifica sus resultados**. La [especificación v1 conservada en Git](https://github.com/vtomasv/nanochat-mlx/blob/1f80132dbda17fd787234361bd2cfcbdd77326f7/ESPECIFICACIONES_NAVARRO_NANOCHAT_MLX.md), los manifiestos, CSV y reportes históricos son la autoridad para aquella campaña. Los CLI históricos siguen siendo v1. La implementación adicional `scripts.navarro_v2` cubre exploración V2-A, parte de V2-F/S y confirmación condicionada; no acredita por sí sola todos los experimentos de esta especificación.

### Lectura y aplicación

- §1–3: evidencia, artículo y correspondencia con el GPT.
- §4–5: modelos de costo y cartera priorizada.
- §6–12: experimentos, controles, pruebas y puertas de avance.
- §13–17: datos, corrección, estadística, medición y selección final.
- §18–20: entregables, ejecución, salvaguardas y referencias.

**DEBE**, **NO DEBE** y **PUEDE** expresan obligación, prohibición y opción. Una opción solo entra en confirmación si quedó fijada antes de observar sus datos reservados. Salvo atribución expresa, las derivaciones para GPT, los umbrales y las políticas siguientes son diseño propio del estudio.

## 1. Evidencia y revisión del planteamiento anterior

### 1.1 Resultados históricos

Resultados de ingeniería en M3 Max, 128 GiB, MLX 0.32.2, FP32, depth 8 y 125.829.648 parámetros. Son pilotos y verificaciones, no confirmaciones estadísticas nuevas.

| Evidencia | Observación documentada | Inferencia permitida |
|---|---|---|
| E1 v1, bitmap de activaciones | Payload P/denso ≈0,4939; pico MLX −13,95%; RSS +46,81%; tiempo/paso ×3,227 | La representación ahorra payload; esta integración CPU pierde frente al nativo y a recomputación |
| E2 v1, estados gramaticales | 68/68 tensores registrados RAW; tiempo/paso ×27,709 | Intentar comprimir estos estados con aquella integración cuesta tiempo sin ahorro |
| E3 v1, pesos MLP | 0/16 matrices elegibles en paso 1000; decode forzado ×2,43 | La política no selecciona capas; forzarlas no demuestra utilidad |
| Control inicial E3 | 8/16 elegibles en paso 0 | Las proyecciones inicialmente nulas producen una impresión engañosa |
| E3 externo SFT | Fallo de tolerancia de logits y de un producto local; reconstrucción exacta | Fidelidad del codec no garantiza fidelidad numérica del operador |
| Constructor de autores | Cero reglas en 14 bloques entrenados; fixture repetitivo sí produce reglas | Control positivo y evidencia dirigida de poca repetición, no censo completo |
| Híbrido posterior | 21,32→9,71 s/paso frente al codec anterior; nativo ≈0,613 s/paso; estados RAW | Optimizar el constructor reduce desperdicio, pero no mejora frente a MLX |

Fuentes: [informe histórico](experiments/navarro/results/informe_final.md), [evaluación MM RePair](docs/navarro/mm-repair-assessment.md), [seguimiento híbrido](docs/navarro/hybrid-repair.md), [comparación de checkpoints híbridos](experiments/navarro/verification/hybrid-repair-v2/comparison.json).

Limitaciones que deben guiar v2:

1. Hay 43 documentos idénticos entre shards de entrenamiento y validación; el replay antiguo no preserva toda la procedencia documental.
2. Mil pasos representan aprendizaje temprano. El checkpoint SFT externo no tiene procedencia suficiente para reemplazar una referencia confirmatoria.
3. Hay variación entre procesos nativos y discrepancias por tensor del híbrido. No se demostró que toda diferencia experimental se explique por ruido nativo.
4. Los tiempos del constructor externo con archivos/procesos y los del codec en memoria tienen fronteras diferentes.
5. Menor pico MLX acompañado de mayor RSS no prueba ahorro físico de memoria unificada.
6. Los 97 tests históricos y los 118 del seguimiento pertenecen a revisiones concretas; no certifican código futuro.
7. Mejorar nuestro codec anterior no equivale a mejorar el sistema nativo.

### 1.2 Estudio nuevo ejecutado para esta revisión

Se realizó un censo CPU de **todas las matrices FP32** de pesos y estados del checkpoint nativo semilla 17, paso 1000. Se verificaron los hashes históricos de ambos archivos antes de leer tensores. Se usaron bloques consecutivos de hasta 128 filas, incluidos bloques finales y matrices pequeñas; vectores y contadores excluidos están enumerados en el JSON.

Para un bloque con N celdas y d patrones FP32 no nulos distintos, el formato actual necesita al menos **64 + 4d bytes**, correspondientes a cabecera y diccionario. Se omiten reglas, símbolos, delimitadores e índices. Si esa cota supera **0,95 × 4N**, el formato no puede pasar la política de selección aunque el constructor fuese instantáneo.

| Objeto | Matrices | Bloques | Descartados por esta cota | Bytes densos examinados |
|---|---:|---:|---:|---:|
| Pesos | 58 | 1.828 | 1.828 | 503.318.528 |
| Estados FP32 matriciales | 64 | 3.364 | 3.264 | 905.971.712 |

Los bloques descartados de estados contienen 879.757.312 bytes densos. Bajo la regla que los deja RAW, aun comprimiendo a cero bytes los restantes, el ahorro máximo de estas matrices sería **2,893512%**, antes de sobrecargas. Los 100 bloques no descartados son momentos de wte y value_embeds; **no se ha demostrado que compriman**. No aparecieron valores IEEE especiales en las matrices examinadas.

La cota descarta el objetivo de ahorrar 20% de estados con **este formato, checkpoint, partición y política**. También descarta seleccionar capas de pesos con esa configuración. No es una imposibilidad para diccionarios globales, otros tamaños de bloque, compresión de bits, otros checkpoints ni todos los compresores. Permutar columnas dentro de los mismos bloques conserva d y N y no rescata estos descartes.

Evidencia: [resumen y hashes](experiments/navarro/verification/spec-v2-census/summary.json), [bloques de pesos](experiments/navarro/verification/spec-v2-census/model.safetensors.csv), [bloques de estados](experiments/navarro/verification/spec-v2-census/optimizer.safetensors.csv), [script reproducible](scripts/navarro_structural_census.py).

Se comprobaron fixtures de valores distintos, ceros y patrones IEEE especiales, ausencia de modificación del archivo leído y agregados contra CSV. Las [cinco pruebas del censo](tests/test_navarro_structural_census.py) pasaron; se conserva su [JUnit](experiments/navarro/verification/spec-v2-census/correctness-junit.xml). Este estudio **no construye gramáticas ni mide velocidad de entrenamiento**. Completa una parte acotada del futuro V2-S, no toda la campaña.

### 1.3 Cambios científicos de v2

- Sustituir una lista fija de implementaciones por un embudo de fidelidad, estructura, operador y sistema.
- Distinguir representación compacta, cálculo directo, planificación de memoria y modificación del modelo.
- Incorporar gradientes de entradas **y de cada peso independiente**, frecuencia de actualización y reconstrucción.
- Conservar controles densos optimizados, recomputación y staging. Las estructuras se centran en Navarro; no se excluyen controles científicos externos pertinentes.
- Separar entrenamiento completo exacto, inferencia exacta y extensiones que cambian precisión o parámetros entrenables.
- Reconocer éxitos limitados a memoria, almacenamiento o capacidad con su etiqueta propia.
- Definir selección, incertidumbre, contaminación de datos, revisiones y preservación de resultados negativos.

## 2. Lectura crítica del artículo indicado

### 2.1 Identidad y versión

**N3:** Ferragina, Manzini, Gagie, Köppl, Navarro, Striani y Tosoni. *Improving Matrix-vector Multiplication via Lossless Grammar-Compressed Matrices*. PVLDB 15(10), 2175–2187, 2022. DOI: [10.14778/3547305.3547321](https://doi.org/10.14778/3547305.3547321). La [ficha institucional](https://research.uniupo.it/en/publications/improving-matrix-vector-multiplication-via-lossless-grammar-compr-2/) confirma la referencia bibliográfica.

Se consultó y descargó exactamente el [PDF indicado por el usuario](https://users.dcc.uchile.cl/~gnavarro/ps/pvldb22.pdf): 13 páginas; SHA-256 **fbf3d67e0550b72d67491212d00a20bd6d047b762c4a8c045516a3c8d647d6ff**. Su primera página contiene marcadores editoriales de volumen, año y DOI; no copiarlos como referencia final. La [copia del coautor Köppl](https://koeppl.github.io/bin/paper/vldb22improving.pdf) sirve de contraste bibliográfico. El preprint N3 utilizado en v1 tiene 26 páginas y otro hash: no intercambiar versiones ni paginaciones.

### 2.2 Aporte y límites de transferencia

N3 representa matrices mediante CSRV y una gramática de símbolos que identifican valor y columna. Las reglas excluyen delimitadores de fila. Los productos derecho e izquierdo recorren la representación comprimida, con espacio auxiliar. Los formatos re_32/re_iv, el bloqueo y el reordenamiento ofrecen compromisos diferentes. [N3, §§2–5](https://users.dcc.uchile.cl/~gnavarro/ps/pvldb22.pdf).

Su ensayo usa ocho matrices de datos, CPU y valores de doble precisión; alterna productos sobre una matriz fija. No evalúa GPT, Metal ni actualizaciones neuronales. Una cota entrópica de la secuencia simbólica no garantiza reducción del archivo completo ni ventaja sobre GEMM optimizado. [N3, §4, tablas 1–3 y ecuación 4](https://users.dcc.uchile.cl/~gnavarro/ps/pvldb22.pdf).

### 2.3 Mapa de comprobaciones propias

| Localizador N3 | Obligación para la adaptación | Riesgo que debe resolver |
|---|---|---|
| §2, figura 1 | Terminales por bits y columna; filas vacías | Fusionar valores próximos o columnas distintas |
| §3.1, teorema 3.4 | Oráculo independiente de producto derecho | Omitir símbolos finales o duplicar contribuciones |
| §3.2, teorema 3.10 | Oráculo independiente de producto transpuesto | Sobrescribir contribuciones compartidas |
| §4, formato práctico | Permitir varios símbolos finales por fila | Suponer que cada fila es una sola regla |
| §4.1 | Declarar alcance de diccionario y bloques | Comparar formatos locales/globales sin contabilizarlos |
| §4.2 | Separar construcción y reutilización | Amortizar un estado que cambia cada paso |
| §5 | Permutación exacta, costo y procedencia | Cambiar la función o ignorar búsqueda |

Esta tabla prescribe nuestra validación; no extiende teoremas a redes neuronales. El [código de autores fijado](https://gitlab.com/manzai/mm-repair/-/tree/edd85fa3193dd1e6e60e1d4886f1ad8ded2a138f) está estudiado por archivo en la [evaluación local](docs/navarro/mm-repair-assessment.md). Nuestro híbrido no adquiere garantías de complejidad de otra implementación por usar ideas similares.

### 2.4 Tres nociones de exactitud

1. **Representación:** decode(encode(A)) conserva bits, dtype, forma y orden en el dominio declarado.
2. **Operador:** los pesos pueden reconstruir exactamente y producir resultados FP32 distintos por reasociación de sumas. Requiere tolerancias y oráculo pequeño de mayor precisión.
3. **Aprendizaje:** diferencias pequeñas pueden amplificarse durante actualizaciones. Requiere gradientes, estados, trayectoria y calidad reservada; no implica identidad binaria de toda trayectoria.

Se prohíbe usar «lossless» para omitir pruebas de producto o calidad. Un defecto de implementación tampoco demuestra imposibilidad científica de la estructura.

## 3. Correspondencia con nanochat: objetos, operaciones y vida útil

### 3.1 Auditoría del código

| Módulo | Hecho auditado | Consecuencia |
|---|---|---|
| [gpt.py](nanochat_mlx/gpt.py) | MLP sin bias, ReLU², RMSNorm, RoPE, QK-norm, value embeddings, lambdas, softcap | Probar todas las rutas del GPT |
| [optim.py](nanochat_mlx/optim.py) | Muon para matrices de bloques; AdamW para embeddings, salida y escalas | Censar familias separadamente y preservar ecuaciones |
| [train.py](nanochat_mlx/train.py) | Acumulación, materialización y escalados LR/schedule | Igualar trabajo y configuración efectiva |
| [dataloader.py](nanochat_mlx/dataloader.py) | BOS best-fit; estado insuficiente para repetir todo el buffer | Replay verificable y procedencia documental |
| [engine.py](nanochat_mlx/engine.py) | KV por concatenación/recorte; offset tras última capa | Control denso preasignado y pruebas de caché |
| [eval.py](nanochat_mlx/eval.py) | BPB necesita longitudes y denominador válido | NLL obligatoria y BPB auditada |
| [activation_tape.py](nanochat_mlx/experiments/activation_tape.py) | Backward segmentado, atención recomputada, reconstrucción P por bloque | Aislar planificación y recomputación |
| [grammar_matrix.py](nanochat_mlx/experiments/grammar_matrix.py) | Codec CPU FP32 little-endian, diccionarios locales, bloques de 128 filas | Metal, otros dtypes y diccionario global son trabajo nuevo |
| [grammar_linear.py](nanochat_mlx/experiments/grammar_linear.py) | Entradas a NumPy y productos CPU por vector | No es GEMM GPU ni VJP de entrenamiento |
| [config.py](nanochat_mlx/experiments/config.py) | Perfiles cerrados v1 | Cambiar solo protocol_version no implementa v2 |

### 3.2 Álgebra del MLP y sus derivadas

Sea m=B×T, d=n_embd y h=4d. Con pesos MLX orientados [salida, entrada]:

~~~text
U : [m,d]          W1 : [h,d]          W2 : [d,h]
Z = U W1ᵀ         P = max(Z,0)        H = P ⊙ P
Y = H W2ᵀ         G = ∂L/∂Y

∂L/∂W2 = Gᵀ H
∂L/∂H  = G W2
∂L/∂Z  = (2P) ⊙ (G W2)
∂L/∂W1 = (∂L/∂Z)ᵀ U
∂L/∂U  = (∂L/∂Z) W1
~~~

Esta derivación propia distingue:

- **Guardar P compacta:** potencial ahorro de memoria; puede reconstruirse para usar productos densos.
- **Operar sobre H compacta:** investigar H W2ᵀ y HᵀG, reutilizando una estructura temporal; sigue haciendo falta P para derivar.
- **Comprimir W:** puede ayudar a forward y gradiente de entrada; no resuelve automáticamente el gradiente de cada peso ni su actualización.

Guardar H y recuperar P=sqrt(H) no es exacto: el cuadrado redondea y puede subdesbordar. Guardar solo el signo pierde la magnitud 2P. Ambas sustituciones están prohibidas en la ruta exacta.

### 3.3 La trampa de entrenar el diccionario

Ejemplo propio: W=[a,a] e y=a·x₁+a·x₂. Los dos pesos actuales tienen igual valor, pero sus gradientes independientes son g·x₁ y g·x₂. Un parámetro compartido de diccionario recibe g(x₁+x₂) y mantiene los pesos atados: **eso cambia el modelo**.

Por tanto:

1. El diccionario representa un estado, no una nueva parametrización aprendible.
2. Cada posición original recibe su gradiente independiente.
3. Después de actualizar, valores iguales pueden separarse y reglas válidas pueden desaparecer.
4. Retener la gramática antigua exige demostrar que representa exactamente el estado nuevo.
5. Muon ejecuta Newton–Schulz sobre cada matriz completa. Sus bloques de almacenamiento no son unidades independientes de actualización.

### 3.4 Matriz de aplicabilidad

| Objeto | Estructura candidata | Mutación/reutilización | Prioridad y condición |
|---|---|---|---|
| P de ReLU² | Bitmap/rank y positivos originales | Nuevo por microbatch; vida forward/backward | Alta: ceros exactos y ahorro de payload observado |
| H de ReLU² | Bitmap o CSRV/gramática | Nuevo por microbatch; múltiples productos posibles | Media: construir puede costar más que reutilizar |
| Pesos de inferencia | Gramática estática | Muchas solicitudes, mismos pesos | Condicional a censo positivo |
| Pesos entrenables | Gramática renovada por actualización | Reutilización entre microbatches | Baja en FP32 actual |
| Momentos y buffer Muon | RAW/bitmap/gramática | Lectura/escritura por paso | Gramática actual desfavorable; censar dispersión |
| Gradientes | Bitmap o filas exactas | Acumulación puede densificarlos | Diagnóstico; no asumir sparsity de lm_head |
| K/V de caché | Bloques sellados y cola mutable | Lectura repetida; append/eviction | Censo primero; atención/softmax siguen presentes |
| Máscaras causales | Fórmula o límites por fila | Estructura conocida | Control implícito antes de comprimir un tensor evitable |
| IDs/límites documentales | Bit packing, bitmap/rank/select | Lectura frecuente, cambios raros | Auxiliar; medir cuello de botella de datos |
| Checkpoints | Codec por tensor/bloque | Escritura/lectura poco frecuentes | Almacenamiento, distinto de cómputo |

## 4. Modelo de viabilidad y descarte

### 4.1 Memoria física

Para N activaciones FP32, k positivas, palabras uint32 y rank muestreado cada s bits:

~~~text
B_dense  = 4N
B_bitmap = 4k + 4 ceil(N/32) + 4(ceil(N/s)+1)
           + cabecera + padding + capacidad sobrante
~~~

Con s=256, ignorando cabeceras/alineación: B_bitmap/B_dense≈p+1/32+1/256, donde p=k/N. Para p=0,5, el cociente es ≈0,53515625. Es una estimación analítica de payload, no de pico global. Un contador más ancho cambia el índice.

En GPU, contar directorio, prefijos locales/globales, temporales de scan, arena y fragmentación. k<N no ahorra memoria residente si permanece una asignación de N valores o una vista retiene el buffer denso.

~~~text
B_grammar = B_valores + B_terminales/IDs + B_reglas + B_C
            + B_filas + B_permutación + B_cabeceras + B_alineación

B_operación = B_grammar + B_scratch + B_entradas + B_salidas
              + B_conversiones_vivas + B_copias_densas_vivas
~~~

Distinguir bits empaquetados, palabras físicas y contenedores Python. Desempaquetar índices permanentemente crea otro brazo de residencia. El pico global depende de buffers simultáneos; no se obtiene sumando máximos individuales.

### 4.2 Cotas conservadoras

1. **Diccionario:** b·d+cabecera limita el formato que guarda cada valor distinto con b bytes. Una muestra no da el total; el conteo parcial puede dar una cota inferior si todos los distintos observados permanecen en el diccionario final.
2. **Singletons:** con t posiciones no nulas y d valores distintos, al menos max(0,2d−t) valores aparecen una sola vez. Bajo sustitución de pares repetidos no pueden quedar absorbidos en reglas reutilizadas. Añadir su almacenamiento mínimo requiere demostrar el ancho real del formato. El híbrido usa una cota conservadora con N≥t; no trasladarla a IDs remapeados o diccionario global sin nueva prueba.
3. **Elegibilidad:** si bloques descartados ocupan fracción q del objeto y se dejan RAW, ahorro máximo≤1−q, ignorando toda sobrecarga restante. Es la cota de §1.2.
4. **Permutación:** conserva bits/cardinalidad; dentro de las mismas filas conserva también cardinalidad por bloque. Cambiar vecindades no vence una cota invariante de diccionario.
5. **Entropía:** diagnosticar bytes, bits, valores y terminales por separado. Baja entropía empírica, correlación o semejanza decimal no prueban reglas reutilizables.

Un descarte demostrado produce REJECTED_BY_BOUND con ecuación, entradas y unidades. No descartado significa UNDECIDED, nunca COMPRESSIBLE.

### 4.3 Tiempo y amortización

~~~text
T_step = T_datos + T_forward + T_backward + T_optimizer
         + T_pack/decode/rebuild + T_sincronización + T_gestión

T_total(Q) = T_construcción + T_carga + Σ T_solicitud(q)
T_solicitud = T_tokenización + T_prefill + T_decode + T_salida
~~~

Son descomposiciones conceptuales; tiempos anidados o solapados no se suman sin barreras y fronteras coherentes.

Una estructura de pesos se invalida al actualizarlos. Acumular a microbatches permite reutilizar pesos antes de la actualización, pero se paga reconstrucción por actualización y el batch efectivo debe igualar al control.

En inferencia, Q_break_even=ceil(ΔT_inicial/ahorro_medio_por_solicitud), solo con ahorro positivo y mezcla de solicitudes fijada; si no, never. La amortización por token es secundaria y debe declarar qué prefill omite. Un formato final pequeño también puede ser inviable si construirlo excede memoria.

### 4.4 Batched, GPU y límites globales

Repetir matvec m veces no garantiza GEMM eficiente. Probar m∈{1,8,32,128,1024,4096} en formas y presupuestos admitidos. Scratch puede crecer con m×n_reglas; usar tiles acotados, respetar DAG y contar lanzamientos, reducción transpuesta y metadata.

Si se toca fracción f del tiempo nativo y esa parte acelera s veces, el ideal es 1/((1−f)+f/s), antes de sobrecargas. Medir f; no deducirlo de FLOPs nominales. El ahorro de memoria depende de qué objetos están vivos en el pico.

Memoria unificada no significa conversiones, coherencia, layout y barreras gratuitas. Menos operaciones CPU pueden perder frente a Metal regular. Perfilar ancho de banda, irregularidad y overhead antes de portar todo el algoritmo.

## 5. Cartera experimental y orden

E1/E2/E3 quedan como IDs históricos. Los nuevos IDs evitan mezclar evidencia. «Obligatorio» exige una conclusión trazable; una cota válida puede cerrar un experimento sin entrenamiento caro.

| ID | Pregunta | Naturaleza | Prioridad/puerta |
|---|---|---|---|
| V2-F | ¿Formato y ambos productos son fieles? | Banco de fidelidad | Obligatorio antes de integración |
| V2-S | ¿Dónde hay estructura real y ahorro posible? | Censo y costo | Obligatorio; §1.2 es una parte inicial |
| V2-A | ¿Guardar P compacta en GPU mejora entrenamiento? | Exacto, activaciones | Primera implementación nueva |
| V2-H | ¿Operar sobre H compacta ayuda al backward/forward? | Exacto, activaciones | Tras censo de H |
| V2-O | ¿Estados/gradientes exactos compactos ayudan? | Estado mutable | Cerrar formato actual por cota donde aplique |
| V2-W | ¿Pesos compactos ayudan a entrenarlos todos? | Actualización completa | Solo con compresión y amortización plausibles |
| V2-I | ¿Pesos estáticos compactos ayudan en inferencia? | Inferencia exacta | Solo candidatos reales |
| V2-Q | ¿La gramática aporta sobre una base cuantizada? | Extensión con pérdida explícita | Condicional y separada |
| V2-FZ | ¿Base congelada compacta facilita adaptación? | Otro conjunto entrenable | Condicional; no preentrenamiento completo |
| V2-K | ¿KV compacta mejora servicio conservando atención? | Caché exacta | Censo y baseline KV correcto |
| V2-D | ¿Índices/checkpoints compactos ayudan al pipeline? | Uso auxiliar | Cuello de botella medido |
| V2-P | ¿Qué combinación ofrece mejor estrategia validada? | Integración | Última etapa |

**Núcleo de entrenamiento:** F+S+A y conclusión explícita de H/O/W; P si existen candidatos. **Núcleo de inferencia:** F+S+I. Las extensiones no rescatan ni ocultan negativos del núcleo.

No exigir 1000 pasos de un formato inelegible por cota. Un ensayo forzado pequeño solo procede para una duda pendiente del operador; E3 v1 ya aporta uno para su implementación CPU.

## 6. V2-F y V2-S: fidelidad y censo

### 6.1 V2-F: banco de corrección

**Hipótesis técnica:** codec y operadores preservan el contrato. Esta etapa no prueba mejoras del GPT.

Brazos: oráculo pequeño NumPy FP64; denso FP32 CPU; CSRV sin gramática; re_32; re_iv; selector RAW/gramática. Para Metal, agregar denso MLX y operador propio. FP64 es diagnóstico numérico, no baseline de velocidad GPU.

Fixtures obligatorios:

- Filas vacías, todo cero, una celda, filas/columnas finales parciales.
- Filas idénticas, pares repetidos, reglas compartidas/anidadas, cadenas profundas y pares solapados.
- Mismos valores en columnas distintas, permutaciones con inversa.
- Finito muy pequeño/grande, cero negativo, NaN con payload e infinitos: RAW exacto o rechazo explícito.
- FP32 casi todo distinto; dispersa con no nulos distintos; densa de alfabeto pequeño.
- Corrupción de formas/IDs, ciclos, referencias futuras, delimitadores, longitudes truncadas, overflow, bit packing entre palabras y expansión desproporcionada.

Pruebas metamórficas: linealidad dentro de tolerancia, vectores base, identidad de adjunto <Ax,y>≈<x,Aᵀy>, partición/unión por bloques, serialización y restauración. El transpuesto **acumula** contribuciones a reglas y columnas compartidas. Comparar errores por elemento y norma.

Constructor híbrido contra oráculo de recuentos en fixtures pequeños; ties y política deterministas. Binarios externos son segunda referencia, no sustituto de pruebas. Constructores con desempates diferentes no necesitan reglas byte-idénticas; sí contenido/operador equivalentes, salvo que se afirme equivalencia exacta de algoritmo.

**Salida:** fidelidad por formato/operador/dtype y límites. Fallos bloquean ese brazo; un fixture repetitivo positivo no acredita utilidad en nanochat.

### 6.2 V2-S: censo completo

**H-S:** existe estructura exacta explotable en suficiente fracción de los objetos vivos. Ejecutar censo fuera del timing de rendimiento.

Universo:

1. Pesos, gradientes antes/después de acumulación, buffer Muon y momentos AdamW.
2. P/H de todos los MLP; K/V cuando se abra V2-K.
3. Checkpoints nativos 0/10/100/500/1000; después, checkpoint maduro de procedencia completa. Paso 0 es diagnóstico.
4. Todas las familias/capas; para activaciones al menos 32 secuencias de calibración estratificadas por longitud/origen.

Registrar forma, dtype, strides, rol, paso/capa, procedencia/hash, celdas, ceros, no finitos, cardinalidad por bits, terminales (bits,columna), frecuencia de pares, reglas/altura, C, bytes completos y construcción. Correlación, baja varianza o cercanía decimal no son igualdad.

Factores de ingeniería: 32/128/512 filas; diccionario local/global; orientación original/transpuesta. Máximo 12 combinaciones por familia antes de congelar candidatos. Un diccionario global paga construcción, IDs más anchos y memoria; transponer paga layout y operador.

Controles de mecanismo:

- Barajar filas dentro del bloque conserva multiconjunto de filas; cambios por desempate se documentan.
- Barajar entradas independientemente por columna conserva marginales y rompe dependencias entre columnas.
- Permutar columnas cambia vecindades; compensar para conservar producto.
- Conservar máscara de ceros y sustituir positivos por valores distintos separa dispersión de repetición.
- Reducir alfabeto artificialmente es fixture sintético, jamás un tensor natural exacto.

Salida: censo, cobertura, bytes vivos, cotas, candidatos y ausencias. Agregar Σbytes_compactos/Σbytes_densos, no media no ponderada de ratios. Distinguir estimaciones muestrales de bytes serializados. §1.2 solo cubre pesos/estados de un checkpoint y una partición.

## 7. V2-A: activaciones positivas compactas en GPU

### 7.1 Hipótesis y brazos

**H-A:** guardar P con bitmap y acceso por rank en GPU reduce memoria sin degradar función, gradientes ni calidad. N1/N2 fundamentan operaciones de bitmap; su uso para P es adaptación propia.

| Brazo | Ejecución | Aísla |
|---|---|---|
| native | Autodiff original | Referencia práctica |
| checkpoint_native | Recomputa segmentos fijados | Alternativa de recomputación |
| tape_dense_gpu | Cinta segmentada, P denso GPU | Planificación |
| tape_recompute_gpu | Misma cinta, P recomputada | Valor de guardar frente a recalcular |
| tape_bitmap_cpu | Histórico revalidado | Costo de residencia CPU |
| tape_bitmap_gpu | P físicamente compacta GPU | Intervención |
| tape_bitmap_gpu_no_rank | Bitmap y recorrido secuencial | Valor incremental de rank si solo se decodifica completo |

El último brazo es ablation diagnóstica salvo selección previa. [Chen et al.](https://arxiv.org/abs/1604.06174) fundamentan el control de recomputación; no se les atribuye el bitmap ni rendimiento en MLX.

### 7.2 Formato y construcción

PackedPositiveGPU contiene forma/dtype/N, bitmap, índice opcional, positivos originales, tamaño lógico, capacidad asignada y propietario. rank(i)=Σ bits anteriores a i; probar 0 y N. Inicialmente uint32 y muestreo 256; calibración 128/256/512 solo piloto.

Dos rutas admisibles:

1. Conteo/scan GPU, obtener total, asignación exacta y escritura; incluir barrera host.
2. Arena por bloques reutilizable, con capacidad/fragmentación contadas. Reserva densa no se declara ahorro residente.

No truncar positivos ni reservar según una esperanza sin fallback exacto. Overflow debe conservar todos los valores y registrar costo. RAW por bloque cuando compacto completo no alcance ≤95% de denso; contar intento.

Los [kernels Metal de MLX](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) permiten operaciones propias, pero no resuelven automáticamente VJP, liberación del grafo o tamaño dinámico. Verificar API, opciones matemáticas y layout en versión fijada.

### 7.3 Integración y backward

Conservar la cinta segmentada como referencia. Guardar entradas/residuales necesarios; materializar y cortar referencias al grafo descartado. No envolver todo el forward en autodiff global esperando liberar sus activaciones.

Reconstruir P de un bloque o tile por vez y aplicar §3.2. Acumular dW1/dW2 entre tiles y entregar gradientes completos al optimizador. La atención recomputada, x0, gates, normalizaciones, lambdas y embeddings con tokens repetidos deben aportar al árbol.

Probar W2 aleatorio no nulo: ceros iniciales ocultan errores. Conservar pesos del forward hasta terminar derivadas. Guardar positivos bit a bit, declarar semántica de ceros, probar subnormales y abortar NaN/Inf sin reemplazarlos por cero.

### 7.4 Métricas y puertas

Medir p, bytes lógicos/asignados, rank/arena, máximo P vivo, pack/unpack, barreras y fallback. Comprobar que no queda copia densa y que backward no reconstruye todos los bloques simultáneamente.

Piloto: corrección aprobada, ≥10% menos payload P y tiempo/paso≤2× nativo. Si falla tiempo, una ronda de optimización guiada por perfil; después cerrar implementación o abrir versión nueva.

Confirmación de memoria útil: ≥10% menos pico MLX contra nativo, ≥5% contra tape_dense_gpu, RSS≤1,05×, tiempo≤1,10× y no inferioridad. Evaluar también tape_recompute_gpu/checkpoint_native: si dominan al compacto, no recomendar bitmap. Ahorro solo de payload se registra como resultado de representación.

## 8. V2-H: productos con activaciones compactas y backward

**H-H:** representar H reduce trabajo/transferencias en Y=H W2ᵀ y dW2=GᵀH lo suficiente para pagar construcción por microbatch. Es hipótesis propia, no conclusión neuronal de N3.

### 8.1 Brazos y álgebra

Brazos: denso nativo; misma segmentación densa; bitmap decode+denso; bitmap con producto disperso; CSRV directo; gramática decode+denso; gramática directa derecha/transpuesta. Cerrar rama gramatical por falta de repetición no cierra necesariamente la dispersa.

Para H:[m,h], forward aplica H a d columnas de W2ᵀ. Para dW2, aplicar Hᵀ a columnas de G y transponer. Confirmación requiere operación batched/tiles, no bucle Python por escalar; comparar con GEMM de iguales formas.

Conservar bits de H ya calculada. Guardar P adicional o recomputarla con control explícito; H no la sustituye. Si se guardan ambas, contar ambas. dH puede ser densa aunque H sea dispersa; el soporte de dZ viene de la derivada y P.

### 8.2 Salvaguardas y decisiones

- Separar ahorro por ceros, diccionario y reglas.
- Validar dW2/dW1/dU y todas las rutas GPT.
- No omitir un gradiente solo porque el peso o activación representada es cero.
- Contar atomics, reducciones entre tiles, scratch y orden; repetir para detectar carreras.
- Contar usos reales de cada gramática antes de destruirla; no amortizar entre lotes diferentes.

Puerta: ≥10% ahorro físico del objeto y construir+forward+backward+liberar≤1,20× control de igual segmentación. Si reglas no mejoran tamaño/trabajo frente a CSRV/bitmap, cerrar aporte gramatical, aunque sparsity resulte útil.

Confirmación: §16 frente a nativo y al mejor control disperso admisible. Veredictos separados para bitmap, diccionario y gramática; no atribuir toda mejora a RePair.

## 9. V2-O y V2-W: estado mutable y entrenamiento completo

### 9.1 V2-O: estados y gradientes exactos

**H-O:** una representación exacta del estado reduce residencia con sobrecarga aceptable. E2 y §1.2 dan evidencia negativa del formato actual; no justifican repetir intentos en cada bloque sin criterio.

Brazos: native, staged_raw con mismas fronteras/residencia, staged_bitmap para ceros exactos, staged_grammar con descarte y selector solo si hay candidatos. Un selector que no comprime debería poder conservar la ruta nativa; mover a CPU y devolver RAW cuesta.

Separar momentos m/v AdamW, Muon y gradientes antes/después de acumulación. Un token ausente no implica momento cero. Softmax puede producir gradientes de lm_head para filas no observadas como targets. No omitir decay de momentos/pesos porque una fila no fue visitada.

Contrato:

1. Calcular/materializar todos los gradientes del paso con iguales pesos antes de modificar parámetros.
2. Recuperar estado por matriz; ejecutar ecuaciones originales, grupos, LR, schedule y contadores.
3. Muon usa matriz completa en Newton–Schulz; bloques solo almacenan.
4. Materializar nuevo estado, codificar y liberar antiguo/temporales sin duplicados permanentes.
5. Checkpoint/restauración conserva representación, RNG/cursor y configuración; exportación densa verificable.

Primero 20 actualizaciones con **gradientes prefijados idénticos** para aislar optimizador; luego integración. NLL próxima no excusa estados incorrectos.

Rama temporal opcional: XOR entre bits del estado anterior/nuevo, con bitmap/rank de cambios. Es propuesta propia de almacenamiento, no producto de N3. Contar referencia, delta, bases y profundidad de cadena. Resta flotante no es sustituto lossless de XOR. Estado denso más delta no demuestra ahorro residente. Comenzar en V2-D si solo reduce archivos.

Puerta: cotas y compresión real en dos checkpoints entrenados, ≥10% ahorro agregado piloto y ciclo≤1,20× staging. La gramática local de 128 filas de §1.2 **se cierra por cota** para meta de 20% sin nuevo entrenamiento.

Confirmación O: ≥20% menos bytes persistentes del estado, RSS≥10% menor que nativo, MLX≤1,05×, tiempo≤1,10× y calidad aprobada. Frente a staged_raw, ≥5% menos del indicador de memoria declarado. Si staging explica todo, la estructura no acredita éxito.

### 9.2 V2-W: pesos compactos durante entrenamiento de todos los parámetros

**H-W:** reutilizar W compacta en forward/dX antes de actualizar compensa reconstrucción, gradiente denso y recompresión de W nueva. Baja prioridad sin una representación/checkpoint con ahorro real.

Brazos: denso nativo; denso con planificación experimental; W compacta decode+denso por uso; W compacta con productos directo/transpuesto, dW independiente por posición, actualización densa idéntica y nueva gramática tras cada paso.

Medir ciclo completo con W densa temporal, gradientes y estados. Forward con W congelada pertenece a I/FZ, no W. GrammarLinear actual no satisface VJP de modelo entrenable por convertir a NumPy.

Prohibido entrenar valores compartidos del diccionario, usar pesos obsoletos o cambiar Muon por actualización de reglas. Esas ideas modificarían arquitectura/optimización y quedan fuera del núcleo exacto.

Puerta: compresión en estados entrenados, amortización conservadora a acumulación fijada, operador batched correcto; después 10 actualizaciones verificadas y 100 pasos piloto. Sin compresión o con costo>2×, cerrar. Si avanza, §16 con idéntico conjunto entrenable.

## 10. V2-I: inferencia con pesos estáticos

### 10.1 Selección y controles

**H-I:** algunas matrices entrenadas permiten una política con mejora de latencia o memoria de servicio. Seleccionar por censo/costo/calibración y congelar antes de prompts reservados.

Examinar lineales MLP, Q/K/V, proyección de atención y lm_head por path. Embeddings se consumen por lookup y necesitan otro operador; no tratarlos como matvec para aparentar ahorro. La política puede dejar todo denso.

Brazos: MLX denso; CPU denso con igual dtype/threading; CSRV CPU; gramática CPU; decode+denso; gramática Metal si F valida; política mixta por capa/fase. CPU/CPU explica mecanismo; MLX/modelo completo decide utilidad. Construcción y producto tienen contrastes separados.

### 10.2 Formatos y permutaciones

Calibrar re_32/re_iv, diccionario local/global y bloqueo dentro de V2-S. Prevenir overflow antes de calcular códigos valor×columnas+columna. Medir anchos físicos y distinguir mejora de packing de mejora del constructor.

Distinguir:

1. **Reordenar recorrido conservando columna original en terminales:** altera vecindades y sumas, sin permutar entrada.
2. **Permutar columnas físicamente:** W'=W[:,π], x'=x[π], entonces W'x'=Wx. El transpuesto debe dispersar salida mediante la inversa. Contar permutaciones/metadata.

No propagar permutaciones por RoPE, cabezas, residuales, normalización, grupos cuantizados o estados sin prueba. Mantenerlas locales al operador. No buscar ordenamiento cuando una cota invariante ya descarta formato.

Primera campaña: una heurística explícita, identidad y permutación aleatoria de control; registrar objetivo, semilla, ties, búsqueda/aplicación. Referir §5 de N3 no autoriza llamar «algoritmo publicado» a cualquier heurística propia.

### 10.3 Prefill, decode y caché

Condición principal: batch 1, prompt de 512 tokens, 128 de continuación fija. Confirmación: ≥64 prompts nuevos de documentos distintos. Secundarias: prompts 128/1024, decode 32/256 y batch 4/8, dentro del dominio validado.

Prefill puede favorecer GEMM. Política densa para prefill y compacta para decode debe contar matriz densa residente o reconstrucción/transición por solicitud. Conservar ambas no permite reclamar memoria de una sola.

Teacher forcing con idénticas continuaciones controla contexto/logits. Greedy libre se mide aparte con EOS natural, longitud efectiva y primera divergencia; no comparar textos de diferente longitud como mismo trabajo.

Probar caché incremental contra prefijos sin caché: batch, offset, ventanas S/L, reinicio y eviction. El motor actual recorta incluso L al tamaño configurado; no es contexto ilimitado. Si chunks/límites exponen un defecto del control, corregir o excluir explícitamente ese régimen antes del codec.

### 10.4 Exactitud, métricas y éxito

Reinvestigar el fallo SFT externo con oráculo FP64 CPU, FP32, sumas por bloques y acumulación FP64 CPU como brazo separado. Acumulación más ancha cambia aritmética; registrar costo/scratch/FMA. No presuponer FP64 en Metal.

Medir TTFT, prefill, media/p95 decode, solicitud, throughput, pesos/KV, picos, construcción/carga y amortización. Definir si TTFT incluye tokenización, sampling y entrega; reportar modelo separado.

**Aceleración:** límite superior ratio decode≤0,90; p95≤1,05; solicitud≤0,95; TTFT≤1,10; MLX/RSS≤1,05; corrección/calidad aprobadas. Ahorro de pesos≥10% es segundo beneficio si existe.

**Memoria:** ≥15% menos pesos/estructuras residentes, ≥10% menos pico primario; otro indicador≤1,05; solicitud≤1,10; calidad aprobada. Solo archivo menor pertenece a almacenamiento.

Sin capas seleccionadas: NEGATIVE_SCREENING_NO_COMPRESSIBILITY para formato/checkpoint. Todo RAW demuestra fallback, no inferencia gramatical mejorada.

## 11. Extensiones condicionales: cuantización y base congelada

Requieren manifiestos y conclusiones separados. Su éxito no reclasifica entrenamiento completo FP32 exacto.

### 11.1 V2-Q: aporte sobre cuantización

**H-Q:** la gramática añade ahorro/velocidad a una representación ya cuantizada. Distinguir pérdida de cuantización y codec posterior exacto.

Factorial: FP32 dense, FP32 grammar, quantized native, quantized grammar. Contraste principal de estructura: **quantized grammar / quantized native**, con iguales códigos, escalas, grupos y checkpoint. Si FP32 grammar no es elegible, registrarlo sin inventar rendimiento.

Propuesta inicial: afín por grupos de 64 a 4 bits en inferencia. Fijar API, redondeo, layout, escalas/bias efectivos; incompatibilidad exige enmienda previa. [quantized_matmul de MLX](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html) define el control a verificar en la versión fijada.

Un código de 4 bits no es un valor real global si cambia escala/bias. Opciones: gramática por grupo compatible, terminal con identidad de grupo/escala o descomposición algebraica demostrada. Contar menor reutilización y metadata; conservar códigos exactos. El baseline ya empaqueta bits y usa kernels especializados, no se mide como FP32.

Probar no inferioridad gramatical frente al mismo cuantizado y calidad del cuantizado frente a FP32 **por separado**. Margen mayor frente a FP32 exige objetivo nuevo preregistrado; no relajar retrospectivamente 0,01 nats. No introducir entrenamiento cuantizado, STE, poda o redondeo de activaciones dentro del núcleo exacto.

### 11.2 V2-FZ: base congelada durante adaptación

**H-FZ:** W₀ compacta sin actualización permite entrenar parámetros adicionales a menor costo. Evita reconstrucción de W₀ por paso; no crea repetición inexistente.

Si se usa LoRA: definir W=W₀+(α/r)BA, rango, matrices, inicialización y conjunto entrenable. [Hu et al.](https://arxiv.org/abs/2106.09685) fundamentan ese control; LoRA no se atribuye a Navarro. Aunque W₀ esté congelada, su transpuesto puede ser necesario para dX hacia adaptadores anteriores; congelar no autoriza cortar ese gradiente.

Brazos: misma base congelada densa, compacta decode+densa y compacta directa; adaptadores idénticos. Comparación adicional con full fine-tuning responde otra pregunta y se separa.

Puerta: W₀ realmente compactable y VJP correcto. Si requiere cuantizar, etiquetar Q+FZ. Medir SFT con datos disjuntos, máscaras de roles, base/adaptadores/estados, forward/backward, calidad y costo de merge/servicio. No generalizar a preentrenar todos los pesos.

## 12. Extensiones exactas de KV y datos

### 12.1 V2-K: KV por bloques inmutables

**H-K:** representar bloques sellados de K/V mientras una cola RAW admite nuevos tokens podría mejorar servicio. Censar K **tras RoPE/QK-norm** y V tras value embedding, exactamente donde se almacenan.

Brazos: KV actual concatenada; KV densa preasignada/circular con misma ventana; bloques RAW con misma política; compactos decode+atención nativa; directo solo si sus componentes son correctos. Mejorar concatenaciones es gestión de buffer, no compresión.

Por cabeza: s=Kq, p=softmax(s+mask), o=Vᵀp, con escala original. Gramática no elimina softmax ni posiciones visibles. Por bloques, softmax necesita normalización global estable de máximo/suma. **Softmax independiente por bloque es incorrecto.** No materializar todos los scores si el control fusionado lo evita.

Contar estructuras/layouts K y V, sellado, cola, índices, scratch, máscaras, referencias vivas, eviction y offset absoluto. No deduplicar KV porque se repite un token textual: contexto/posición cambian valores.

Contextos 128/512/1024 y máximo validado; no exceder dominio para fabricar presión. Probar límites de bloque/ventana, EOS, reset y batch. Sin ahorro o con construcción dominante, cierre por screening; no desarrollar RePair dinámico sin evidencia.

Meta: ≥15% menos KV residente, ≥10% menos pico primario, otro indicador≤1,05, solicitud≤1,10× denso optimizado y calidad aprobada. Informar si KV es irrelevante frente al peso del modelo.

### 12.2 V2-D: tokens, límites y checkpoints

Dos subestudios:

- **D-data:** mismo flujo exacto de IDs/targets, enteros mínimos/bit packing, bitmap de límites y rank/select cuando haya consultas que los necesiten. Comparar int32, uint16 si IDs caben y offsets densos. Targets −1, especiales y padding requieren formato explícito; ancho según máximo ID. Conservar packing, orden, procedencia y denominadores.
- **D-checkpoint:** RAW, compresor genérico de control fijado y codec estructural. Medir conjunto restaurable, construcción, escritura/lectura, restauración y pico. Incrementales cuentan cadena completa y se prueban con archivos faltantes/corruptos. Compresores genéricos son controles de archivo, no operadores de N3.

No deduplicar eliminando ejemplos: cambia frecuencias. Compartir almacenamiento debe preservar cada aparición y orden. Un índice que duplica offsets necesita una consulta útil demostrada.

Puerta: costo de datos/I/O relevante medido. Meta auxiliar: ≥20% ahorro de archivo/índice con reconstrucción≤1,10× control adecuado; además medir pipeline completo. Archivo menor sin mejora de pasos es SUCCESS_STORAGE.

## 13. Datos, perfiles, replay y presupuesto

### 13.1 Confirmación nueva

El conjunto histórico fue observado y tiene contaminación. Sirve para depuración/historia; **no es test reservado v2**.

1. Fijar corpus/revisión/shards/tokenizer/hashes; no sustituir corpus implícitamente.
2. Split por documento antes de packing; eliminar intersecciones exactas train/calibración/test por hash. Registrar normalización y auditoría de duplicados cercanos o su ausencia.
3. Guardar document_id, split, offsets y segmentos de procedencia de cada secuencia. Un pack con varios documentos no recibe uno arbitrario para bootstrap.
4. Calibración/test separados e inmutables; dimensionar antes de confirmación.
5. Mismas entradas/targets/máscaras/orden entre brazos; registrar tokens válidos, BOS/EOS, padding, épocas y documentos.
6. Verificar replay contra dataloader y medir pipeline aparte.

Objetivo inicial: 1.024 secuencias de 1.024 tokens válidos o equivalente documentado, de ≥128 documentos disjuntos; calibración adicional de 128 secuencias de otros documentos. Ajustar por disponibilidad/potencia **antes** del bloqueo del manifiesto. Cortar un documento en muchos segmentos no crea independencia.

Inferencia: ≥64 prompts reservados y calibración separada. SFT: split por conversación/origen y máscaras verificadas. Procedencia incompleta bloquea calidad confirmatoria aunque permita diagnóstico de codec.

### 13.2 Perfiles del núcleo

| Campo | Smoke | Principal | Escala posterior |
|---|---:|---:|---:|
| depth / anchura | 4 / 256 | 8 / 512 | 12 / 768 |
| head_dim | 128 | 128 | 128 |
| secuencia | 128 | 1024 | 1024 |
| ventanas | L | SSSL | SSSL |
| microbatch | 2 | 4 | 4 |
| tokens/actualización | 256 | 8192 | 8192 |
| acumulación | 1 | 2 | 2 |
| dtype | FP32 | FP32 | FP32 |
| validación funcional | 10–20 pasos | piloto 100 | tras puertas |

Fijar vocabulario 32.768 solo si tokenizer lo confirma; otro requiere perfil propio. Registrar árbol/grupos reales; cifras históricas aproximadas no sustituyen conteo.

A igual trabajo: mismos parámetros entrenables, longitud, batch efectivo, orden, LR, weight decay y schedule de 1000 actualizaciones, warmup 0, warmdown 0,5 y final_lr_frac 0, con escalados originales. Checkpoint k significa k actualizaciones terminadas; siguiente step=k. No usar el horizonte piloto para redefinir el schedule de confirmación.

Capacidad puede variar microbatch/secuencia/modelo bajo límite fijo, pero es eje separado (§16). Cambiar microbatch conservando batch efectivo requiere ajustar acumulación y verificar; cambiar modelo/longitud cambia el problema.

### 13.3 Reanudación y recursos

Guardar pesos, estados, grupos, RNG Python/NumPy/MLX, cursor, paso, schedule, formatos/hashes. Restaurar mediante temporales verificados y renombrado atómico; no sobrescribir checkpoints. Comparar resume con continuo y exportación densa.

Piloto: semilla 101/datos de ingeniería. Confirmación propuesta: **211/307/401/503/601**, fijadas antes de generar trayectorias; no reutilizar 17/29/43 como datos no observados. Cinco semillas es mínimo operativo, no garantía de potencia.

Memoria: menor entre 80 GiB, recomendación del dispositivo y margen seguro disponible. Límite MLX no limita RAM CPU; vigilar ambos. Registrar swap; detener comparación si incremento sostenido>1 GiB o presión invalidante, preservando incidente.

Presupuesto orientativo: hasta 24 horas por familia que avance, estimadas con piloto. No iniciar confirmación cuyo diseño completo no cabe. Si falta presupuesto, INCONCLUSIVE_RESOURCE_BUDGET; no truncar semillas/pasos tras resultados. Entrenamiento maduro necesita duración/tokens/schedule propios fijados desde inicio, no prolongación de schedule agotado.

## 14. Corrección escalonada

### 14.1 Puertas

| Puerta | Demuestra | Requisito |
|---|---|---|
| C0 | Representación exacta | Bytes, forma, dtype, dominio y serialización |
| C1 | Operador correcto | Derecho/transpuesto/batched, oráculo/metamórficas |
| C2 | Derivadas correctas | Cada entrada/parámetro, VJP y diferencias finitas |
| C3 | Actualización correcta | 20 pasos de gradientes fijos, estados/contadores |
| C4 | Integración GPT | 10 pasos/resume, rutas/máscaras |
| C5 | Trayectoria | Piloto y confirmación, errores/calidad |
| C6 | Servicio | Caché/ventanas/batch, prefijos/logits/NLL |

MLX real para puertas que lo requieren; mocks y tests omitidos no equivalen a aprobados. Falta de dispositivo no impide C0/C1 CPU, pero bloquea sus integraciones.

### 14.2 Tolerancias iniciales FP32

- Codec: igualdad bit a bit, sin tolerancia.
- Operador: abs(error)≤1e-5+1e-4·abs(referencia) por elemento; máximo, relativo L2 y ULP cuando proceda.
- Gradiente por tensor: atol=1e-5, rtol=1e-3 y relativo L2≤1e-3; norma cero usa absoluto.
- Logits: atol=1e-4, rtol=1e-3; diferencia de NLL de lote≤1e-4 nats/token.
- C3 con mismos gradientes/grafo: esperar bits iguales. Si cambia backend/orden, documentar causa y aplicar criterio preregistrado de brazo sin debilitar C0.

Aplicar condiciones por elemento/tensor, no promedio global. Error máximo absoluto aislado tampoco reemplaza criterio relativo. Registrar todo incumplimiento y peor tensor. Márgenes son decisiones del estudio, no teoremas del backend.

Diferencias finitas: función escalar pequeña, FP64 CPU si procede, pesos no nulos, lejos del quiebre ReLU, epsilon 1e-2/1e-3/1e-4. No demandar FP64 GPU ni aceptar una sola epsilon como demostración.

### 14.3 Variación y diagnóstico

Repeticiones nativo/nativo con mismos bytes/datos/procesos antes de comparar. Caracterizan variación; **no autorizan cualquier diferencia experimental**. Si nativo incumple puerta, investigar nondeterminismo, orden, kernel o prueba y registrar limitación.

Separar codec, redondeo, algoritmo y carreras. Probar cancelación, subnormales y reducciones compartidas. Registrar FMA/fast-math/acumuladores; activar fast en un solo brazo es otro factor.

Modificar tolerancia exige informe causal, nueva versión y nuevos datos confirmatorios. Conservar fallos anteriores. No quitar el checkpoint externo desfavorable.

## 15. Estadística, calidad y prevención de sesgos

### 15.1 Estimandos

**Ejecución:** tiempo/memoria de igual trabajo desde igual estado. **Aprendizaje:** calidad a iguales tokens/actualizaciones y costo hasta calidad fijada. Réplicas de timing no son semillas de aprendizaje.

NLL=Σcross-entropy/Σtokens válidos; no promediar promedios con longitudes distintas. Perplejidad=exp(NLL) con igual tokenizer. BPB secundaria con bytes auditados. Guardar NLL sumada/conteos por unidad.

### 15.2 Piloto y selección

1. Declarar hipótesis, máximo de variantes, endpoint/márgenes/exclusiones antes del piloto.
2. Usar piloto independiente para varianza/tamaño. Dimensionar 80% de potencia con efecto de diseño fijado; documentar supuestos/sensibilidad. Si no cabe, etiquetar exploratorio.
3. Congelar código, capas, umbrales, muestra, semillas, prompts y análisis por hashes antes de confirmación.
4. No agregar semillas/endpoints/candidatos mirando test. Ampliación es nuevo estudio.
5. Registrar todos los intentos, no solo ganador.

### 15.3 Rendimiento

Entrenamiento: ventanas 400–499 y 900–999 desde referencias entrenadas con schedule completo. Mínimo propuesto diez pares de procesos por contraste/ventana; aumentar antes según piloto. AB/BA balanceado, aleatorización semilla 20260915. Mismos pesos/estados/lotes por par. Calentar copia descartable y restaurar.

Inferencia: ≥10 pares de procesos con iguales prompts, orden balanceado. Tokens de solicitud son correlacionados; informar tamaño y definición de p95, distribución entre procesos y no convertir cada token en réplica.

Tiempo: promedio de actualizaciones/solicitudes dentro del proceso; media geométrica de cocientes pareados entre procesos. Memoria: pico por proceso y ratio pareado. Puntos individuales/medianas/p95 se muestran como descriptivos.

IC: bootstrap pareado de bloques independientes proceso/ventana, 10.000 remuestreos y semilla fijada. Ambas ventanas separadas y criterios en ambas. Con muestra insuficiente o IC inestable, inconcluso; muchos pasos correlacionados no compensan pocos procesos.

### 15.4 No inferioridad

Margen δ=0,01 nats/token para NLL_experimental−NLL_control; factor perplejidad exp(δ)≈1,01005. Declarar no inferioridad solo con límite superior unilateral 95% **menor** que δ. Diferencia no significativa no prueba equivalencia.

Entrenamiento: trayectorias por semilla, cintas pareadas, test fijo. Análisis principal: diferencia token-ponderada por semilla y límite unilateral t de diferencias pareadas, explicitando supuesto entre semillas. Sensibilidad: bootstrap jerárquico semillas/documentos si procedencia permite. Documentos no inflan número de semillas.

Inferencia de checkpoint fijo: remuestrear documentos/prompts independientes para NLL pareada; no inventar incertidumbre de entrenamiento. Con varios checkpoints, preregistrar jerarquía.

Cinco semillas pueden ser insuficientes. Si IC cruza margen o conclusión depende fuertemente del método: INCONCLUSIVE_QUALITY. Reportar también curvas NLL/tokens y NLL/tiempo, gradientes y fallos.

### 15.5 Múltiples pruebas y datos faltantes

Una familia tiene un contraste principal. Para afirmar que «alguno funciona» seleccionando entre K familias, usar límites unilaterales ajustados Bonferroni α=0,05/K o alternativa preregistrada. IC95 sin ajuste se muestran como descriptivos. Todos los endpoints obligatorios de un éxito se cumplen conjuntamente.

Distinguir fallos externos documentados de OOM, NaN o crash propio. Estos últimos son resultados de factibilidad/corrección, no exclusiones silenciosas. Repetición por causa externa preserva registro y regla simétrica. Métrica ausente=null con motivo; nunca cero o aprobación implícita.

## 16. Utilidad y estrategia final V2-P

### 16.1 Definir «mejor»

No existe estrategia universal sin carga/restricciones. Elegir dentro de regímenes realmente probados:

| Objetivo | Métrica | Condición mínima |
|---|---|---|
| Acelerar entrenamiento | Tiempo/actualización a igual trabajo | Límite superior ratio≤0,95; MLX/RSS≤1,05; calidad aprobada |
| Ahorrar memoria entrenamiento | Pico primario declarado | ≥10% reducción; otro indicador≤1,05; tiempo≤1,10; calidad |
| Mejorar inferencia | Decode/solicitud/TTFT | §10.4 |
| Capacidad | Mayor configuración viable al mismo límite | Sin swap invalidante, ejecución sostenida, calidad/costo explícitos |
| Almacenamiento | Conjunto restaurable | V2-D; no es mejora de cálculo |

Memoria primaria: A usa pico MLX; O usa RSS con límites de medición; otras familias declaran indicador antes de piloto. Reportar ambos y buffers. Indicador no medible bloquea éxito de memoria.

Son **umbrales nuevos**, no resultados esperados ni reclasificación v1. Tamaños de archivos concretos son deterministas; tiempos/picos requieren IC. Éxito incluye efecto incremental contra control de igual planificación según familia.

### 16.2 Capacidad y tiempo hasta calidad

Si permite mayor microbatch a mismo batch efectivo, medir throughput con acumulación ajustada y verificar. Si permite modelo/secuencia mayor, nuevo contraste a presupuesto de tiempo/memoria/tokens definido; medir calidad. No atribuir mejor NLL a estructura sin controlar el modelo mayor.

Elegir NLL* con piloto independiente antes de confirmar. Medir tiempo acumulado hasta alcanzarla, incluidos costos de preparación según amortización declarada y evaluación común. Brazo que no llega es no alcanzado/censurado, no se elimina ni inventa tiempo.

Mil pasos permiten conclusiones de aprendizaje temprano. Estrategia de entrenamiento final necesita checkpoint maduro y evaluación externa trazable.

### 16.3 Integración y selección

Combinar solo candidatos aprobados. Medir nativo, cada uno solo y combinación; con dos candidatos, diseño 2×2. Ahorros pueden coincidir/desplazar picos y no se suman. Q/FZ conservan etiqueta.

Selector por capa/fase usa tamaño/costo de calibración, con fallback nativo, frecuencia de reevaluación, overhead y prevención de oscilación. En estado mutable, un descarte puntual no demuestra imposibilidad futura: no volver a probar es política de costo, no teorema. Reevaluaciones predefinidas cuentan y no usan NLL test para elegir.

Salida: frontera de Pareto tiempo/memoria/calidad/costo inicial, recomendación por régimen y alternativas dominadas. Puede recomendar denso con recomputación, bitmap por capas o gramática solo para archivo. No afirmar óptimo sobre variantes no estudiadas.

### 16.4 Estados

Separar estado operativo, corrección y veredicto:

- NOT_RUN/RUNNING/COMPLETE/BLOCKED: ejecución; completo no significa hipótesis positiva.
- PASS/FAIL/NOT_TESTED: por puerta.
- SUCCESS_SPEED/MEMORY/CAPACITY/STORAGE: solo beneficio acreditado y dominio.
- NEGATIVE_SCREENING/NEGATIVE_SCREENING_NO_COMPRESSIBILITY/REJECTED_BY_BOUND: cierre previo con alcance.
- FAILURE_CORRECTNESS: implementación inadmisible, no refuta estructura.
- FAILURE_PERFORMANCE: implementación correcta y evidencia suficiente de incumplimiento.
- INCONCLUSIVE/INCONCLUSIVE_QUALITY/INCONCLUSIVE_RESOURCE_BUDGET: evidencia insuficiente.

Todo RAW puede demostrar fallback económico, no compresión. Ningún campo faltante se interpreta como aprobado.

## 17. Medición en Apple Silicon

### 17.1 Tiempo

Reloj monotónico y materialización/sincronización reales. MLX es perezoso: construir grafo no ejecuta cómputo. Evaluar salidas, pérdida, gradientes, parámetros y estados pertinentes con fronteras iguales.

Separar frío, compilación, carga/conversión, calentamiento, estable y finalización. Total inclusivo y estable. Perfilar fases en corridas separadas con barreras, sin usar perfil intrusivo como timing principal ni sumar cronómetros anidados. Incluir intento fallido/fallback.

Entrenamiento: actualización completa con acumulación y ensayo pipeline≥100 pasos con dataloader. Inferencia: modelo y solicitud con misma definición. No incluir impresión/calculadora solo en un brazo ni herramientas externas durante kernel benchmark.

### 17.2 Memoria

Registrar:

1. MLX activa/pico reiniciado/caché y política.
2. RSS muestreado, high-water mark y máximo concurrente padre/hijos, unidades verificadas.
3. Footprint físico/presión cuando disponible, limitaciones y páginas compartidas.
4. Bytes lógicos/asignados, arena, índices, reglas, diccionario, copias y temporales.
5. Swap antes/durante/después y límites/recomendaciones.

[get_peak_memory](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html) es un indicador del allocator. **No sumar RSS y Metal como memoria física disjunta**, ni máximos de hijos en instantes distintos. Muestreo, por ejemplo 50 ms, puede perder picos breves; combinar con contabilidad/high-water marks.

Proceso nuevo por brazo, mismas cachés/límites. No vaciar caché por paso solo en intervención. Contar toda arena reservada. Auditar referencias lazy y buffer subyacente tras liberar handles.

### 17.3 Entorno

Chip/RAM/núcleos/macOS, Python/MLX/NumPy, compilador/flags/binario/hash, uv.lock, Git/diff, energía/carga concurrente. Medir conectado a corriente; registrar throttling/incidencias. Energía solo si se mide: tiempo no equivale a energía.

Usar uv según [CLAUDE.md](CLAUDE.md), fijar lock sin actualizar durante confirmación. Binarios identificados por fuentes **y** flags/compilador. API fallida produce null+error, no cero.

## 18. Implementación y entregables

### 18.1 Estado real y trabajo pendiente

Reutilizar módulos existentes tras revalidar; no reemplazar entrypoints de producción por defecto ni sobrescribir v1.

| Componente | Estado en esta revisión | Pendiente |
|---|---|---|
| Bitmap/cinta CPU, codec/operador CPU, híbrido | Existentes con historia | Revalidar controles v2 |
| Censo de diccionario §1.2 | Ejecutado, nuevo script | Activaciones, pasos y otros formatos |
| P compacta Metal | No implementada | Construcción, residencia, VJP/integración |
| H batched y backward | No implementado | Operadores, derivados, scratch |
| W entrenable con VJP | No implementado | Gradientes por posición y reconstrucción |
| Permutación/selector v2 | No implementado | Exactitud y costo |
| Q/FZ/K/D | Diseños condicionales | Solo tras puertas |
| CLI/manifiestos/análisis v2 | Especificados, pendientes | No presentarlos como flags existentes |

Estructura **propuesta a crear**, no interfaz actual:

~~~text
configs/navarro/v2/<family>/<stage>.json
nanochat_mlx/experiments/v2/
  manifest.py, census.py, decisions.py, analysis.py
  bitmap_metal.py, activation_ops.py, weight_vjp.py
  selectors.py, kv_blocks.py                  # si pasan puertas
scripts/navarro_v2.py                         # futuro
tests/test_navarro_v2_*.py
experiments/navarro/v2/<family>/<run_id>/
~~~

Puede variar organización antes del piloto si preserva contratos y trazabilidad.

### 18.2 Artefactos de ejecución

~~~text
manifest.json        Preregistro, configuración, fuentes y hashes
environment.json     Hardware, software, Git y compilación
correctness.json     C0–C6, errores por tensor y omisiones
events.jsonl         Costos, fallback, OOM, errores e incidencias
steps.csv            Pasos/solicitudes/tokens y réplica
tensors.csv          Censo, capacidad física, vida y representación
quality.csv          NLL sumada, conteos, procedencia
summary.json         Estimandos, IC, decisión y límites
report.md            Explicación y enlaces a datos crudos
commands.sh          Comandos exactos y orden
artifacts.sha256     Integridad de resultados/fuentes
~~~

Archivos grandes/corpus/PDF/checkpoints fuera de Git con hashes/localizadores. IDs nuevos, escritura exclusiva. Resume verifica manifiesto/cursor sin duplicar filas o mezclar código.

Ejemplo futuro, sin resultados:

~~~json
{
  "protocol_version": "2.0",
  "experiment": "V2-A",
  "stage_status": "NOT_RUN",
  "evidence_kind": "planned_confirmatory",
  "correctness_status": "NOT_TESTED",
  "scientific_verdict": null,
  "repo_sha": null,
  "source_diff_sha256": null,
  "manifest_sha256": null,
  "control": "native",
  "matched_execution_control": "tape_dense_gpu",
  "primary_objective": "memory",
  "primary_memory_metric": "mlx_active_peak_bytes",
  "checkpoint_hashes": {},
  "dataset_hashes": {},
  "tokenizer_hash": null,
  "selection_frozen_at": null,
  "sample_size_plan": null,
  "paired_ratios": null,
  "confidence_intervals": null,
  "quality_noninferiority": null,
  "fallback_bytes_fraction": null,
  "limitations": [],
  "raw_files": []
}
~~~

Validador bloquea éxito con puerta/métrica ausente, selección excedida o hashes distintos. Reducción=100(1−experimental/control); speedup=tiempo_control/tiempo_experimental. Siempre declarar denominador/unidad/dtype/alcance.

### 18.3 Comandos disponibles ahora

Los comandos históricos siguientes no ejecutan la campaña v2 completa:

~~~bash
# Censo con directorio NUEVO; requiere checkpoints locales.
uv run --frozen python -m scripts.navarro_structural_census \
  --checkpoint experiments/navarro/artifacts/primary/checkpoints/primary-native-reference-s17/step-1000 \
  --output experiments/navarro/verification/spec-v2-census-repeat

# Pruebas existentes; MLX/Metal real cuando corresponda.
uv run --frozen python -m pytest tests/test_navarro_grammar.py tests/test_navarro_bitmap.py -q
uv run --frozen python -m pytest tests/ -q
~~~

Para historia: [runbook v1](docs/navarro/runbook.md), [seguimiento híbrido](docs/navarro/hybrid-repair.md). Su confirm no acredita nuevos splits/semillas/criterios.

### 18.4 Enmienda de ejecución: autoresearch v2, 15-09-2026

Se agrega un [ejecutor y runbook v2](docs/navarro/autoresearch-v2.md) y un
[programa de agente](research/navarro/program.md). El perfil registra nueve
escenarios de activaciones, hasta tres propuestas adaptativas adicionales y un
presupuesto de cuatro horas de workers. Los cambios del candidato se archivan;
no se altera el evaluador dentro de una campaña. El test reservado queda fuera de
búsqueda y se abre únicamente bajo un plan confirmatorio sellado.

Esta primera ejecución es **exploración de ingeniería**: una trayectoria nativa,
pilotos emparejados por checkpoint/cinta, censo temporal y pruebas de gramáticas
sobre muestras de activaciones. Los ensayos H/O/W/I y las ramas condicionales
requieren sus propias puertas antes de presentarse como ejecutados. El banco
implementado valida roundtrip exacto, gradientes completos y controles de VJP;
no se debe etiquetar como aprobación automática de todas las puertas C0–C6.

Enmiendas de recursos/datos explícitas: 128 documentos de calibración producen
82642 tokens válidos; 1024 documentos reservados producen 641033, por padding y
longitudes efectivas. La reserva es menor que un millón de tokens y se reporta
como tal. Solo se audita duplicación exacta; la procedencia del tokenizer y los
casi duplicados permanecen como limitaciones. Con 24 GiB libres, los pilotos
conservan huellas de estados finales y los checkpoints nativos necesarios para
replay; no se guardan todos los estados completos. Reserva de disco: 3 GiB.

~~~bash
uv run --frozen python -m scripts.navarro_v2 prepare
uv run --frozen python -m scripts.navarro_v2 init
uv run --frozen python -m scripts.navarro_v2 campaign
uv run --frozen python -m scripts.navarro_v2 report
~~~

Los comandos por defecto usan `configs/navarro/v2/campaign.json` y
`experiments/navarro/v2/20260915`. Los resultados pertenecen al hash de fuentes y
perfil de su manifest. Este documento no se modifica para reinterpretar una
corrida ya congelada. `report.md`, `results.tsv`, `paper/` y `ledger.jsonl` de la
corrida son los entregables de ejecución; su ausencia significa que aún no hay
resultado, no que el experimento haya fallado científicamente.

## 19. Salvaguardas y cierre

### 19.1 Riesgos vinculados a comprobaciones

| Riesgo | Salvaguarda | Evidencia |
|---|---|---|
| Confundir datos matriciales y pesos GPT | Censo por objeto/paso | S y fuentes |
| Medir solo ceros iniciales | Checkpoints entrenados | Tabla temporal |
| Cambiar precisión silenciosamente | Hash/dtype y Q separado | C0/manifiesto |
| Atar pesos iguales | Gradiente independiente | §3.3, C2/C3 |
| Perder autodiff por NumPy/stop-gradient | VJP de entradas y parámetros | C2/C4 |
| Ocultar recomputación/staging | Control de igual ejecución | Contraste incremental |
| Cronometrar solo grafo lazy | Materialización/barreras | Eventos |
| Mover memoria Metal a host | RSS/footprint/buffers | Picos separados |
| Reserva densa llamada compacta | Capacidad/arena/referencias | Vida útil |
| Ignorar construcción/descarte | Ciclo completo/amortización | Totales |
| Seleccionar con test | Splits nuevos y selección congelada | Hash/fecha |
| Pseudorreplicación de tokens | Procesos/semillas/documentos | Análisis |
| Ocultar peor tensor | Errores por tensor | C1–C5 |
| Ajustar tolerancias tras fallo | Nueva versión/confirmación | Enmienda |
| Atribuir Q/LoRA a exacto FP32 | Factorial/dominio explícito | Q/FZ |
| Suponer actualizaciones compactas gratis | Ciclo por paso/cola | O/W/K |
| Softmax local por bloque | Normalización global estable | C6 |
| Licencia uniforme inferida de raíz | Inventario por archivo | Procedencia |
| Sobrescribir evidencia | IDs/hashes/escritura exclusiva | Historia intacta |

La [inspección MM RePair](docs/navarro/mm-repair-assessment.md) documentó avisos diferentes entre raíz y archivos brepair. Antes de incorporar código externo identificar archivos/avisos concretos; mantener distinción entre implementación propia y binarios de comparación. No afirmar licencia uniforme solo por archivo raíz.

### 19.2 Orden operativo

1. Leer instrucciones, fijar HEAD/diff/entorno y verificar historia.
2. Fuentes y C0/C1; ampliar S a activaciones/vida útil.
3. Datos/procedencia e instrumentación para confirmación.
4. A GPU y controles, derivados/residencia antes de rendimiento.
5. Decisiones explícitas H/O/W/I; implementar ramas viables pendientes.
6. Pilotos; congelar código/candidatos/endpoints/muestra/presupuesto.
7. Confirmar o cerrar negativo/inconcluso con alcance.
8. P e interacciones si hay candidatos; después escala/calidad madura.
9. Q/FZ/K/D separados si pasan puertas, sin rescate estadístico del núcleo.
10. Entregar código/pruebas/datos/informe/recomendación limitada a evidencia.

### 19.3 Completitud por familia

- [ ] Pregunta/objeto/dominio de exactitud/atribución claros.
- [ ] Baseline práctico y control de mecanismo.
- [ ] Cotas y construcción real diferenciadas.
- [ ] Corrección pertinente aprobada o fallo explicado.
- [ ] Datos, semillas, selección y presupuesto congelados.
- [ ] Tiempo inclusivo, memoria y fallback visibles.
- [ ] Calidad con unidades estadísticas válidas y potencia evaluada.
- [ ] Veredicto reproducible, incluidos negativos/fallos.
- [ ] Alcance sin extrapolar a otros modelos/chips/precisiones.
- [ ] Historia intacta y comandos suficientes para repetir.

**Un cierre negativo por cota o ensayo correcto completa la pregunta. La investigación no exige encontrar una victoria de las estructuras compactas: exige establecer dónde aportan, dónde no y qué evidencia falta.**

## 20. Referencias y trazabilidad

### Estructuras

- **N1:** González, Grabowski, Mäkinen y Navarro. [Practical Implementation of Rank and Select Queries](https://users.dcc.uchile.cl/~gnavarro/ps/wea05.pdf), 2005, §1.3. Base de rank muestreado; aplicación a P propia.
- **N2:** Navarro y Providel. [Fast, Small, Simple Rank/Select on Bitmaps](https://users.dcc.uchile.cl/~gnavarro/ps/sea12.1.pdf), SEA 2012. Elección de operaciones; no exigir select sin uso.
- **N3:** Ferragina et al. [PDF solicitado](https://users.dcc.uchile.cl/~gnavarro/ps/pvldb22.pdf), PVLDB 2022; hash/identificación §2.1. [Código fijado](https://gitlab.com/manzai/mm-repair/-/tree/edd85fa3193dd1e6e60e1d4886f1ad8ded2a138f). Hipótesis GPT son adaptaciones.
- **N4:** Navarro. [Practical Adaptive Dynamic Bitvectors](https://users.dcc.uchile.cl/~gnavarro/ps/spe25.pdf), 2025. Contraste estático/dinámico; no hay justificación actual para árbol dinámico en P inmutable de vida breve.

Hashes/versiones anteriores: [mapa histórico](docs/navarro/source_map.md). El PDF nuevo de N3 no reemplaza el hash del preprint histórico.

### Controles y APIs

- **B1:** Chen et al. [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174), 2016. Control de recomputación.
- **B2:** Dao et al. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), 2022. Motiva medir tráfico y no materializar scores innecesarios. No se afirma identidad con kernel MLX ni se trasladan aceleraciones.
- **B3:** Hu et al. [LoRA](https://arxiv.org/abs/2106.09685), 2021. Solo control de adaptación V2-FZ.
- **M1:** MLX, [Custom Metal Kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html).
- **M2:** MLX, [get_peak_memory](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html).
- **M3:** MLX, [quantized_matmul](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html).

APIs consultadas el 15-09-2026; confirmar contra lock local antes de implementar.

### Evidencia local

- **L1:** [Informe v1](experiments/navarro/results/informe_final.md), [auditoría](experiments/navarro/results/audit.json), [variabilidad nativa](experiments/navarro/results/native-variability.json).
- **L2:** [Estudio MM RePair](docs/navarro/mm-repair-assessment.md), [seguimiento híbrido](docs/navarro/hybrid-repair.md).
- **L3:** [Censo nuevo](experiments/navarro/verification/spec-v2-census/summary.json), [script](scripts/navarro_structural_census.py), CSV enlazados en §1.2.

Toda afirmación de resultado apunta a evidencia histórica o run nuevo; toda propuesta se identifica como tal. La literatura proporciona métodos y condiciones, no mediciones de este nanochat.
