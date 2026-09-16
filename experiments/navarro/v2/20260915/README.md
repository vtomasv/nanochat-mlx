# Campaña Navarro v2 — evidencia del 15-09-2026

**Estado:** ciclo de exploración terminado; ningún candidato preseleccionado.
Confirmación y evaluación reservada no ejecutadas. La campaña conserva espacio
para dos propuestas nuevas bajo su presupuesto original.

## Leer primero

- [Manuscrito con resultados y límites](paper/manuscript.md).
- [Informe automático de los pilotos](report.md) y [tabla TSV](results.tsv).
- [Figuras de entrenamiento](paper/figures/training.pdf) y
  [mecanismos](paper/figures/mechanisms.pdf), y
  [memoria/tiempo de gradientes](paper/combined-figures-r2/combined.pdf),
  también disponibles en SVG/PNG.
- [Runbook y comandos](../../../../docs/navarro/autoresearch-v2.md).
- [Contenido y límites del paquete de evidencia](../EVIDENCE.md).

## Evidencia

| Parte | Resultado | Artefacto |
|---|---|---|
| Referencia | 1000 updates nativos | [summary](trials/reference/summary.json) |
| Pilotos | Diez ventanas de 100 updates | [decisiones](decisions.json) |
| Propuesta adaptativa | Constructor SIMD, gradientes válidos | [ensayo](trials/bitmap-metal-simd/summary.json) |
| Puerta previa | 128 pruebas, cero fallos/omisiones | [JUnit](gate/junit.xml) |
| Validación SIMD posterior | 33 pruebas, incluido reinicio | [registro](verification/candidate-validation.json) |
| Censo inicial | 18 combinaciones checkpoint/objeto/bloque | [summary](screening/summary.json) |
| Censo ampliado y H directa | 512 activaciones; 28/32 operadores fallan tolerancia | [summary](mechanisms/summary.json) |
| Gradientes | 870 observaciones antes/después de acumular; 98 roundtrips exactos | [summary](gradient-census/summary.json) |
| Filas de gradientes + bitmap | Seis pilotos adicionales; −29,91% memoria vs. nativo, +13,50% tiempo; no preseleccionado | [informe y tabla](verification/closeout-r2/report.md) |
| Intento previo de filas | Fallo del verificador antes de pilotos, conservado | [fallo](combined-gradients/failure.json) |
| Oráculo FP64 | Sensibilidad al orden de reducción; bloqueo conservado | [summary](direct-diagnosis/summary.json) |
| Diccionarios globales | Pesos/estados descartados en ambas orientaciones probadas | [summary](dictionary-extension/summary.json) |
| Integridad y reservado | Hashes verificados; `seal` rechazado | [auditoría](verification/evidence-audit.json) |
| Cierre ampliado | Once ensayos principales y seis adicionales verificados; decisiones recalculadas | [auditoría r2](verification/closeout-r2/audit.json) |
| Procedencia | Revisión pública con shards idénticos por SHA256 | [anexo](verification/corpus-provenance-addendum.json) |

[Manifest](manifest.json), [fuentes iniciales](sources.zip), [ledger](ledger.jsonl).
Los hashes por ensayo también incluyen snapshots particulares y cambios del
candidato. Los [scripts complementarios](../studies/) tienen planes y hashes
propios. Ningún resultado anterior se reescribe como si perteneciera a otra fuente.

## Cobertura del protocolo

| Experimento | Alcance realmente cubierto y decisión |
|---|---|
| V2-F | Banco existente más codec Metal, VJP, integración y reinicio SIMD. El producto H nuevo falla su puerta; no se aprueban automáticamente todos los formatos/operadores posibles. |
| V2-S | Pesos/estados temporales, tres bloques locales, diccionario por matriz en dos orientaciones, 32 documentos P/H. Gradientes de dos microbatches antes/después de acumulación en cinco checkpoints; vida completa de buffers pendiente. |
| V2-A | Cuadrícula y propuesta adaptativa ejecutadas; ahorro marginal inferior al criterio registrado. Sin confirmación. |
| V2-H | Prototipo CSR directo, derivadas y diagnóstico FP64 ejecutados. Fallo numérico y costo elevado impiden integración GPT. Gramática con cero reglas en la muestra. |
| V2-O | Estados: las cotas tardías no permiten la meta de los formatos estudiados. Gradientes: seis pilotos completados; filas compactas mejoran memoria, pero ambos candidatos exceden el límite de tiempo. Sin confirmación. |
| V2-W | Pesos de los formatos examinados descartados estructuralmente. |
| V2-I | Sin pesos elegibles para la política gramatical estudiada; no se inventan tiempos de servicio gramatical. |
| V2-Q/FZ/K/D | No ejecutados: extensiones separadas y condicionales. |
| V2-P | Sin candidato que integrar como mejora confirmada; nativo sigue siendo la ruta por defecto. |

Las ramas no ejecutadas y el censo parcial no equivalen a fracaso universal de
estructuras compactas. El manuscrito limita cada conclusión a los objetos,
formatos, pasos y hardware realmente examinados.
