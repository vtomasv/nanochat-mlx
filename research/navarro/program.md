# Programa del agente de investigación Navarro v2

## Objetivo y límites

Investigar si estructuras compactas mejoran entrenamiento real de nanochat-mlx,
conservando resultados negativos. Leer primero la especificación v2 y
`docs/navarro/autoresearch-v2.md`. Este programa adapta el ciclo de
[karpathy/autoresearch](https://github.com/karpathy/autoresearch) a MLX. El agente
externo propone código; el controlador ejecuta y registra escenarios. El
controlador no contiene un LLM ni promete inventar optimizaciones por sí solo.

Solo puedes editar `research/navarro/candidate.py` dentro de una campaña abierta.
El contrato es `pack_positive(p, sample_bits=256, strategy='rank')`. Debe devolver
un objeto con `unpack('mlx')`, `resident_bytes`, `mode` y `pack_seconds`. Dominio:
FP32 finito no negativo; conservar subnormales y bits positivos exactamente,
canonizar ceros con signo como indica el protocolo. La salida de `unpack` tiene
la forma original. No modificar pesos, optimizador, pérdidas, datos o gradientes.

No editar el evaluador, pruebas, escenarios registrados, umbrales, presupuesto,
estado de RNG, checkpoints ni informes de intentos anteriores. No abrir
`reserved_*` ni `documents.sqlite`. No consultar resultados reservados para
proponer otra versión. La guarda Python detecta errores accidentales; no es una
barrera de seguridad contra código nativo hostil. No introducir llamadas de red,
lecturas externas o procesos que sobrevivan al ensayo.

## Ciclo acotado

1. Leer `results.tsv`, `report.md`, `ledger.jsonl` y los resultados de calibración.
   Distinguir payload de pico MLX y RSS; nunca sumar MLX y RSS.
2. Formular una hipótesis falsable con una sola modificación principal. Explicar
   qué costo cambia: construcción, índice, reconstrucción, sincronización o bytes.
3. Editar el candidato y ejecutar `submit` con ID nuevo e hipótesis explícita:

   ```bash
   uv run --frozen python -m scripts.navarro_v2 submit \
     --campaign experiments/navarro/v2/20260915 \
     --trial candidate-01 \
     --hypothesis 'Una pasada menos de construcción reducirá tiempo manteniendo representación exacta'
   ```

4. El worker copia el código antes de importarlo, verifica hashes, ejecuta
   roundtrip y todos los gradientes sobre pesos entrenados, restaura el mismo
   checkpoint después del calentamiento y mide una ventana fija. Archiva incluso
   crashes y fallos de corrección; no reutiliza IDs.
5. Leer `correctness.json`, `steps.csv`, `tensors.jsonl`, `quality.json` y
   `summary.json`. Una sola observación favorable solo permite preseleccionar.
6. Mantener el candidato como fuente de la siguiente propuesta o reemplazarlo
   explícitamente por otra versión. **Descartar no borra evidencia.** El arnés no
   modifica Git ni ejecuta reset/rebase. Los snapshots son la autoridad aunque
   el árbol de trabajo tenga cambios sin commit.
7. Detenerse al agotar 12 pilotos totales, el presupuesto de cuatro horas, una
   violación de fidelidad no resuelta o los límites de memoria/disco. Los nueve
   escenarios iniciales dejan hasta tres propuestas adaptativas. Una nueva
   hipótesis fuera del contrato exige nueva campaña y enmienda escrita.
8. Solo si hay preselección admisible, congelar con `seal --trial ID`. Tras ello
   ejecutar `confirm`; no ajustar el candidato ni detener por un resultado
   intermedio favorable. Si falta presupuesto, registrar confirmación pendiente.

## Redacción científica

No convertir `CANDIDATE_FOR_CONFIRMATION` en éxito. No atribuir mejoras de la
recomputación al bitmap. No extrapolar cotas de diccionarios locales a toda
estructura compacta. Publicar todos los intentos, tolerancias, denominadores,
hardware, checkpoints, semillas, tiempos excluidos e incluidos, y límites de
contaminación del corpus. Si no hay mejora, explicar el mecanismo medido sin
afirmar imposibilidad universal. No fabricar IC con pasos de un mismo proceso.
