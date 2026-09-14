# Resultados de los experimentos Navarro

Adaptaciones de estructuras compactas al entrenamiento e inferencia de nanochat-mlx. No son reproducciones de experimentos neuronales de Navarro.

Referencia auditada: `b54b9fc139f455a9a5e60dc9a688ca9dbdb22944`. Apple M3 Max, 128 GiB, MLX 0.32.2, FP32. Perfil principal: depth 8, secuencia 1024, batch 8192 tokens, acumulación 2. Corpus y tokenizer locales fijados por hash.

Cada piloto usa semilla 101 y procesos separados. El checkpoint principal de inferencia usa semilla 17, paso 1000. Los cocientes siguientes son descriptivos de ingeniería; no son intervalos de confirmación. NLL se evalúa en 96 secuencias reservadas.

Referencia nativa completa: **1000 actualizaciones**, **8,192,000 tokens procesados**, **125,829,648 parámetros**; NLL final **5.296038** y **0.5991 s/paso**. [Mediciones y checkpoints](baseline/primary-native-reference-s17/report.md).

Verificación final: **97 pruebas aprobadas, 1 omitidas, 0 fallidas**. [JUnit y razones de omisión](correctness-junit.xml).

## E1: entrenamiento

| Brazo | Pasos | NLL final | s/paso (media) | Tokens/s | Pico MLX (bytes) | RSS máximo (bytes) |
|---|---:|---:|---:|---:|---:|---:|
| [native](E1/primary-pilot-native-s101/report.md) | 100 | 6.6302656 | 0.6228 | 13153.4 | 6161390588 | 1550008320 |
| [tape_dense](E1/primary-pilot-tape_dense-s101/report.md) | 100 | 6.6302693 | 0.8039 | 10189.8 | 5570193744 | 1549893632 |
| [tape_recompute](E1/primary-pilot-tape_recompute-s101/report.md) | 100 | 6.6302703 | 0.8287 | 9885.3 | 5301758288 | 1549320192 |
| [tape_bitmap](E1/primary-pilot-tape_bitmap-s101/report.md) | 100 | 6.6302928 | 2.0098 | 4076.1 | 5301758288 | 2275508224 |

Cociente de tiempo intervención/nativo: **3.227×**. Reducción del pico MLX: **13.95%**. Cambio de RSS: **+46.81%**. Diferencia NLL final: **+0.0000272 nats/token**. IC y no inferioridad confirmatoria: `null`.
Payload P residente durante pasos 50–99: 0.4939 del denso, incluidos contenedores CPU. La recomputación tiene el mismo pico MLX que bitmap en este piloto y menor tiempo.

![Curvas E1](E1/pilot-curves.png)

Veredicto: **NEGATIVE_SCREENING**. [Puerta de decisión y datos](E1/primary-pilot-decision-s101-20260914T155331700057Z/report.md).

Medición con dataloader original (`pipeline`), separada de replay:

| Brazo | Pasos | s/paso | NLL final |
|---|---:|---:|---:|
| [native](E1/primary-pipeline-native-s101/report.md) | 100 | 0.6325874691191712 | 6.630256315072377 |
| [tape_dense](E1/primary-pipeline-tape_dense-s101/report.md) | 100 | 0.8037867283518426 | 6.630266944567363 |
| [tape_recompute](E1/primary-pipeline-tape_recompute-s101/report.md) | 100 | 0.8288412025297294 | 6.63026720782121 |
| [tape_bitmap](E1/primary-pipeline-tape_bitmap-s101/report.md) | 100 | 2.0088832382508555 | 6.630272189776103 |

## E2: entrenamiento

| Brazo | Pasos | NLL final | s/paso (media) | Tokens/s | Pico MLX (bytes) | RSS máximo (bytes) |
|---|---:|---:|---:|---:|---:|---:|
| [native](E2/primary-pilot-native-s101/report.md) | 100 | 6.6302537 | 0.6290 | 13024.8 | 6161390588 | 1548943360 |
| [staged_raw](E2/primary-pilot-staged_raw-s101/report.md) | 100 | 6.6302632 | 1.4644 | 5594.2 | 5255418752 | 3236151296 |
| [staged_grammar](E2/primary-pilot-staged_grammar-s101/report.md) | 100 | 6.6302538 | 17.4278 | 470.1 | 5255418748 | 2224291840 |

Cociente de tiempo intervención/nativo: **27.709×**. Reducción del pico MLX: **14.70%**. Cambio de RSS: **+43.60%**. Diferencia NLL final: **+0.0000000 nats/token**. IC y no inferioridad confirmatoria: `null`.
Estado residente tras la última actualización: 1.000418 del payload denso. RAW en 68/68 tensores registrados. Los intentos de construcción y sus copias están incluidos en s/paso.

![Curvas E2](E2/pilot-curves.png)

Veredicto: **NEGATIVE_SCREENING**. [Puerta de decisión y datos](E2/primary-pilot-decision-s101-20260914T155331854039Z/report.md).

Medición con dataloader original (`pipeline`), separada de replay:

| Brazo | Pasos | s/paso | NLL final |
|---|---:|---:|---:|
| [native](E2/primary-pipeline-native-s101/report.md) | 100 | 0.6112409541595843 | 6.630252773563067 |
| [staged_raw](E2/primary-pipeline-staged_raw-s101/report.md) | 100 | 1.473905864561384 | 6.630266651511192 |
| [staged_grammar](E2/primary-pipeline-staged_grammar-s101/report.md) | 100 | 18.221508423800696 | 6.63025838136673 |

## E3: inferencia

Checkpoint nativo principal: paso 1000, semilla 17. Condición medida: ocho prompts reservados de 512 tokens, 128 pasos de decode con continuación fija por prompt, batch 1. Conversión forzada de `blocks.4.mlp.c_fc.weight`. Cada brazo es un proceso nuevo.

| Brazo | s/token (media) | p95 s/token | TTFT (s) | Conversión (s) | Pesos residentes (bytes) | Pico MLX (bytes) | RSS (bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|
| [dense_native](E3/primary-E3-dense_native/report.md) | 0.002011 | 0.003061 | 0.011803 | 0.000000 | 503318592 | 843407996 | 626737152 |
| [compressed_decode_dense](E3/primary-E3-compressed_decode_dense/report.md) | 0.006891 | 0.008609 | 0.020377 | 0.086348 | 506602389 | 843637372 | 681590784 |
| [grammar_direct_cpu](E3/primary-E3-grammar_direct_cpu/report.md) | 0.004879 | 0.006834 | 0.659186 | 0.084927 | 506602389 | 690178356 | 663896064 |

Matrices MLP elegibles: **0/16**. [Todas las capas](E3/primary-E3-screen/tensors.csv). [Corrección del modelo entrenado y microbenchmark](E3/primary-E3-correctness/report.md).

[Concordancia y primera divergencia de generación greedy](E3/greedy-comparison.json).

Veredicto: **NEGATIVE_SCREENING_NO_COMPRESSIBILITY**. [Decisión](E3/primary-pilot-decision-s101-20260914T155332003003Z/report.md).

Validación externa: checkpoint SFT depth 4, paso 257033; 0/8 matrices elegibles. Veredicto: **FAILURE_CORRECTNESS**. [Datos y límites de procedencia](E3/external-E3-sft-d4/report.md). No reemplaza el checkpoint principal.
El producto directo incumple la tolerancia de logits (máximo absoluto 0,000329), aunque pasa NLL. La reconstrucción seguida del cálculo denso coincide exactamente con el nativo; uno de ocho vectores locales también incumple la tolerancia frente al oráculo FP64. Esto localiza la discrepancia en el producto CPU y limita su uso con este checkpoint; no se ampliaron tolerancias ni se modificó el operador medido. [Investigación separada](E3/external-E3-sft-d4-reduction-diagnostic/correctness.json).

Control diagnóstico inicial: 8/16 matrices elegibles en paso 0, frente a 0/16 en paso 1000. Las proyecciones inicialmente nulas no predicen compresión de pesos aprendidos. [Paso cero](E3/primary-E3-step0-diagnostic/tensors.csv).

Perfiles de fases con barreras explícitas, excluidos de los cocientes principales: [native](E1/primary-detailed-native/report.md), [tape_bitmap](E1/primary-detailed-tape_bitmap/report.md), [tape_dense](E1/primary-detailed-tape_dense/report.md), [tape_recompute](E1/primary-detailed-tape_recompute/report.md), [staged_grammar](E2/primary-detailed-staged_grammar/report.md), [staged_raw](E2/primary-detailed-staged_raw/report.md).

La generación greedy libre se registra aparte con EOS natural y longitud efectiva en los eventos de cada brazo. No se usa para sustituir el benchmark de decode con contexto fijo.

## Límites y trazabilidad

- Los intervalos confirmatorios permanecen `null`: las puertas negativas detienen el escalado; no se ajustan umbrales ni precisión.
- El corpus tiene 43 documentos idénticos entre shards de entrenamiento y validación. El packing no conserva procedencia por documento; no se reporta bootstrap por documento.
- MLX mostró variación numérica entre repeticiones del propio CLI: [diagnóstico nativo/nativo y nativo/harness](native-variability.json). No se afirma identidad bit a bit de trayectorias largas.
- La construcción RePair es una implementación propia compilada con recuentos repetidos. Sus tiempos no son una evaluación de la implementación lineal de los autores.
- RSS y Metal se reportan por separado; no se suman como memorias físicas disjuntas.
- Las corridas fallidas y diagnósticas se conservan en el [índice completo](report.md). Los archivos grandes están en `../artifacts/` y quedan fuera de git.
- Cuatro índices del smoke E1 todavía describían el marcador inicial de corrección. Se conservaron los índices anteriores y se documentó la [reconciliación de metadatos](audit-metadata-reconciliation.json), sin cambios en CSV. El XML histórico de esa puerta no se conservó; la suite completa final aporta evidencia separada.
- [Fuentes y adaptación](../../../docs/navarro/source_map.md), [implementación](../../../docs/navarro/implementation.md), [reproducción](../../../docs/navarro/runbook.md), [tests](correctness-junit.xml), [auditoría de artefactos](audit.json).
