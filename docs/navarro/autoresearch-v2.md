# Autoresearch Navarro v2: implementación y ejecución

## Qué se ha implementado

Un controlador de experimentos MLX inspirado en el ciclo proponer, probar,
conservar/descartar y registrar de
[karpathy/autoresearch](https://github.com/karpathy/autoresearch). Es una
implementación independiente. No utiliza su entrenamiento PyTorch/CUDA ni su
métrica de cinco minutos como comparación causal de representaciones.

La búsqueda automática inicial recorre nueve escenarios. Las propuestas de
código nuevas las hace un agente externo siguiendo
[`program.md`](../../research/navarro/program.md); `submit` las somete al mismo
evaluador. El sistema no llama por sí mismo a una API de modelos. Puede ejecutarse
sin credenciales ni servicios externos una vez preparados los datos locales.

| Elemento | Implementación |
|---|---|
| CLI | `scripts/navarro_v2.py` |
| Perfil primario y smoke | `configs/navarro/v2/{campaign,smoke}.json` |
| Controlador, presupuestos, procesos | `nanochat_mlx/experiments/v2/campaign.py` |
| Candidato editable | `research/navarro/candidate.py` |
| Codec de activaciones | `nanochat_mlx/experiments/v2/bitmap_metal.py` |
| Datos por documento | `nanochat_mlx/experiments/v2/data.py` |
| Evaluador fijo | `nanochat_mlx/experiments/v2/worker.py` |
| Confirmación y estadística | `nanochat_mlx/experiments/v2/confirmation.py` |
| Censo estructural | `nanochat_mlx/experiments/v2/screening.py` |
| Informes y trazabilidad | `reporting.py`, `records.py` del mismo paquete |

## Hipótesis y escenarios ejecutables

V2-A estudia conservar P=ReLU(Z) durante backward en un bitmap con valores FP32
exactos. La ejecución sigue usando multiplicaciones densas. El control
`tape-dense` aísla la representación en la misma implementación manual de VJP;
`tape-recompute` y `checkpoint-native` identifican beneficios alcanzables por
recomputación. `native` es el entrenamiento real de referencia.

El codec Metal genera máscaras uint32, cuenta positivos, calcula prefijos,
compacta valores y reconstruye P. Variantes: índice rank cada 128, 256 o 512 bits,
y bitmap sin índice persistente que recalcula prefijos al decodificar. Conserva
los valores como patrones uint32 para no perder subnormales por comparación o
copia flotante con flush-to-zero. El almacenamiento registrado incluye arrays y
64 bytes de cabecera lógica, no el tamaño exacto de objetos Python. Picos MLX y
RSS se miden por separado. La asignación de k valores implica sincronización
GPU→CPU para conocer k; su costo está incluido en el paso.

`bitmap-cpu` conserva el codec histórico como control. RAW se selecciona cuando
el payload compacto no ahorra al menos 5%. La preselección exige beneficio global
y marginal frente a tape denso; la mera compresión de P no basta.

V2-S aplica cotas de diccionario local a pesos y estados en pasos 100, 500 y 1000,
con bloques de 32, 128 y 512 filas. Además prueba gramáticas exactas en P/H de
cuatro documentos de calibración, primeras 128 posiciones y todas las capas.
Esto informa V2-H/O/W/I; no equivale a haber implementado sus operadores directos.
Cuantización, LoRA congelado, KV, índices de datos y entrenamiento integrado
siguen siendo ramas condicionales de la especificación.

## Ejecución

Requiere Apple Silicon, Metal real y dependencias de `uv.lock`. En un agente con
sandbox, los comandos MLX pueden necesitar acceso fuera del sandbox al dispositivo.

```bash
uv run --frozen python -m scripts.navarro_v2 preflight
uv run --frozen python -m scripts.navarro_v2 prepare
uv run --frozen python -m scripts.navarro_v2 init
uv run --frozen python -m scripts.navarro_v2 campaign
uv run --frozen python -m scripts.navarro_v2 status
```

`prepare` e `init` exigen destinos nuevos. Para otra campaña usar `--campaign` y,
si cambia el conjunto de datos, `--data`. El ID final del directorio de campaña
tiene que ser único porque también identifica su directorio de checkpoints.
`campaign --dry-run` muestra escenarios y presupuesto sin entrenar.

`campaign` ejecuta la puerta de pruebas, una trayectoria nativa de 1000 updates,
nueve pilotos de 100 updates desde el mismo checkpoint 100 y el censo. Cada
update procesa 8192 tokens, T=1024, B=4 y acumulación=2; la planificación de LR
mantiene su horizonte de 1000 updates. El calentamiento se descarta y se restaura
el estado inicial antes de medir. Se informa tiempo por update materializado,
tiempo total y evaluaciones por separado. Los pilotos se ejecutan secuencialmente
en orden registrado; no ofrecen IC ni neutralizan deriva térmica mediante ese
orden. La confirmación alterna aleatoriamente brazos dentro de pares de procesos.

Repetir `campaign` omite ensayos terminados y verificados. Un ensayo interrumpido
sin cierre bloquea el avance para evitar omitirlo silenciosamente. Ctrl-C termina
el grupo del worker y conserva su fallo. Para corregir el arnés, crear una campaña
nueva: sus fuentes están congeladas. Los informes pueden regenerarse con `report`.

```bash
uv run --frozen python -m scripts.navarro_v2 seal --trial bitmap-metal-r256
uv run --frozen python -m scripts.navarro_v2 confirm
```

Estos últimos comandos solo proceden si existe un candidato admisible y recursos
registrados suficientes. La confirmación usa cinco semillas nuevas, trayectorias
completas de 1000 pasos y diez pares de procesos por ventana 400 y 900 con controles
nativo/denso/recompute. El endpoint es memoria con tiempo ≤1,10 del nativo,
memoria MLX ≤0,90, RSS ≤1,05, ahorro adicional frente al tape denso y no inferioridad
de NLL con margen 0,01. La media geométrica usa bootstrap de pares de procesos;
calidad usa diferencias por semilla e intervalo t unilateral, supuesto explícito
con n=5. No confundir 100 pasos dentro de un proceso con 100 réplicas independientes.

## Datos, almacenamiento y limitaciones registradas

Preparación primaria del 15-09-2026: dos shards locales, 106496 ocurrencias y
106431 documentos distintos; 65 repeticiones exactas conservadas dentro de su
partición. Cada hash de documento asigna siempre el mismo split 90/5/5. Hay cero
documentos idénticos compartidos entre splits. Se registran segmentos, recortes y
ocurrencias del empaquetado BOS largest-fit/shortest-crop. Calibración: 128
documentos, 82642 tokens válidos. Reservado: 1024 documentos, 641033 tokens válidos.
Estos son tamaños efectivos, inferiores a un millón por las longitudes y padding;
no se cuenta padding como evidencia. No se ha auditado duplicación cercana ni la
procedencia completa de entrenamiento del tokenizer existente. Estas limitaciones
impiden afirmar ausencia total de contaminación.

La búsqueda no abre tokens reservados; la preparación sí los construye. Hashes del
manifest identifican todas las particiones. Al abrir reservado, el worker verifica
también sus hashes y el plan sellado. Los hashes encadenados detectan modificación
accidental del registro; no constituyen certificación externa contra un atacante
que reescriba toda la cadena.

Con 24 GiB libres al preparar el lanzamiento, se registra política de disco:
checkpoints nativos completos en 10/100/400/500/900/1000, estado inicial y reserva
mínima de 3 GiB. Cada piloto conserva hashes por tensor de pesos/optimizador al
final, no un checkpoint completo adicional. Es reproducible por replay pero no
reiniciable directamente desde esos hashes. La confirmación conserva checkpoints
400/900 de la primera semilla nativa y estados iniciales, además de huellas finales.
No se borran checkpoints históricos ni descartes para fabricar espacio.

El límite de cuatro horas incluye el tiempo total de workers terminados y las
estimaciones de confirmación. Preparación, censo y pruebas se registran aparte;
no deben confundirse con tiempo de entrenamiento. Un worker individual tiene
timeout; la trayectoria nativa permite hasta 1800 s. El guard de swap detiene
aumento sostenido superior a 1 GiB. No se alteran configuraciones del sistema.

## Evidencia para el paper

Cada ensayo conserva request, candidato, snapshot de fuentes (incluye cambios sin
commit), hashes, entorno, consola, errores de gradientes, tiempos por paso,
contabilidad de activaciones, NLL por documento y huellas finales. `ledger.jsonl`
conserva eventos; `results.tsv` incluye fallos con métricas null; `report.md` y
`paper/results.md` se derivan de resultados verificados. `paper/outline.md` es una
estructura de manuscrito, no un artículo concluido.

La prueba de desarrollo inicial con comparaciones float perdió un subnormal:
20 fallos y 11 aprobados. Se conserva su código y registro en
`experiments/navarro/v2/development/001-float-comparison-subnormal/`; el archivo
indica que el log original solo estuvo en consola. La corrección por patrones
enteros queda distinguida por snapshot y JUnit. No se borra ese intento negativo.

La atribución científica es precisa: el artículo de
[Ferragina et al., PVLDB 2022](https://users.dcc.uchile.cl/~gnavarro/ps/pvldb22.pdf)
motiva productos sobre matrices comprimidas mediante gramáticas. El bitmap de P
es una hipótesis propia para este GPT; no es un resultado demostrado en el artículo.
El paper futuro debe separar compresión, costo de acceso, reconstrucción y efecto
global, y publicar los casos negativos con el mismo detalle que los positivos.

## Resultado de la ejecución del 15-09-2026

La [campaña registrada](../../experiments/navarro/v2/20260915/README.md) completó
una referencia de 1000 actualizaciones y diez pilotos de 100: los nueve escenarios
iniciales y una propuesta adaptativa SIMD. Ningún candidato pasó la preselección.
El intento de sellar confirmación se rechazó y no se evaluó el reservado.

La propuesta SIMD midió 0,922 s/update y 5,058 GiB de pico MLX. Su ahorro de memoria
frente al nativo fue 25,53%, pero solo 2,50% frente al mismo tape con P densa. La
comparación temporal es descriptiva y sin réplicas independientes. Las 128 pruebas
previas y 33 pruebas adicionales del candidato pasaron sin omisiones.

Los [estudios complementarios](../../experiments/navarro/v2/studies/) añaden:

- 512 muestras de P/H en 32 documentos: compresión exacta por ceros, cero reglas.
- Productos CSR Metal directos y derivadas en 32 casos: 28 fallan el criterio de
  forward por elemento; 11,01× tiempo local denso incluyendo construcción.
- Diagnóstico FP64 de discrepancias, sin relajar tolerancias ni integrar el operador.
- Cotas globales por matriz en ambas orientaciones y censos tempranos adicionales.
- Censo de 870 matrices de gradientes en cinco checkpoints; 98 muestras con
  roundtrip exacto, sin extrapolar sus ratios a todos los gradientes.
- Seis pilotos separados de acumulación por filas, con controles nuevos. Filas
  más bitmap: 4,761 GiB, −29,91% frente al nativo y −8,23% frente al tape denso;
  0,927 s/update, +13,50% frente a su nativo. Ambos candidatos fallan el máximo
  de +10% de tiempo. [Informe](../../experiments/navarro/v2/20260915/verification/closeout-r2/report.md).

La extensión comprueba en cada paso los ceros de filas omitidas y reconstruye
gradientes antes del optimizador. Pasaron la prueba pequeña de diez updates de
pesos/estados y la puerta inicial de gradientes del modelo completo. Las huellas
finales tras cien updates difieren de los controles: no afirmar trayectoria bit
a bit idéntica ni equivalencia final de pesos a tamaño completo. El primer intento
falló por comparar estados por posición; se conserva el fallo y la revisión que
compara rutas de parámetros, con las mismas tolerancias.

El [anexo de procedencia](../../experiments/navarro/v2/20260915/verification/corpus-provenance-addendum.json)
identifica después de ejecutar una revisión pública con los mismos bytes de los
dos shards. No modifica el manifest previo ni resuelve la procedencia del tokenizer
o la duplicación cercana. La
[auditoría ampliada](../../experiments/navarro/v2/20260915/verification/closeout-r2/audit.json)
verifica los hashes, fuentes, peticiones y aritmética de las decisiones.

El informe generado `report.md` corresponde a la cuadrícula de entrenamiento y
su censo inicial; el [manuscrito](../../experiments/navarro/v2/20260915/paper/manuscript.md)
integra también los estudios complementarios. Hay figuras SVG/PDF/PNG, tablas y
fuentes reproducibles. No se ha completado ni validado en ejecución una campaña
confirmatoria `confirm`: esa ruta requiere un candidato admisible, justificar
potencia/muestra y revisar su plan antes de abrir el reservado. Los mínimos
operativos de semillas/procesos no garantizan potencia por sí mismos.

### Continuar sin alterar resultados

`research/navarro/candidate.py` conserva la propuesta SIMD. Quedan dos propuestas
dentro del presupuesto de 12 pilotos. Utilizar un ID nuevo y una hipótesis; cada
ensayo copia y verifica su código. Las fuentes protegidas de la campaña continúan
intactas. Si cambia el arnés, perfil o hipótesis fuera de su contrato, iniciar otra
campaña con manifest nuevo, preservando la anterior.

```bash
uv run --frozen python -m scripts.navarro_v2 status
uv run --frozen python -m scripts.navarro_v2 submit \
  --trial candidate-02 --hypothesis 'Describir el cambio concreto y su predicción'
```

Para reproducir estudios ya terminados, utilizar directorios de salida nuevos
cuando el CLI lo permita, o una copia del paquete de evidencia con nueva ruta
registrada. Los scripts complementarios son fuentes de un estudio concreto y
rechazan sobrescribir su destino; no borrarlos para repetir un resultado.

La auditoría de cierre puede repetirse sin Metal y sin abrir arrays reservados:

```bash
uv run --frozen python -m experiments.navarro.v2.studies.closeout \
  --output experiments/navarro/v2/20260915/verification/closeout-new
```

La extensión de filas es una enmienda exploratoria independiente, con dos variantes
y cuatro controles, fuera de la superficie P-only original. No consume ni amplía
silenciosamente su presupuesto de doce propuestas. Su conclusión negativa cumple
la regla de parada registrada; continuar optimizando exige otro plan y destinos
nuevos, conservando todos los intentos y controles anteriores.

El [paquete de evidencia](../../experiments/navarro/v2/EVIDENCE.md) agrupa código,
pruebas, fuentes congeladas, resultados y figuras con SHA256 por archivo. Permite
auditar las decisiones sin GPU; para repetir exactamente el entrenamiento siguen
siendo necesarios los datos, el tokenizer identificado y Metal. No incluye esos
payloads ni checkpoints grandes.
