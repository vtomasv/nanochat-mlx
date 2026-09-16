# Paquete de evidencia Navarro v2

La campaña y su extensión terminaron como exploración: no hay candidato admisible
ni evaluación reservada. Leer el [manuscrito](20260915/paper/manuscript.md), la
[auditoría ampliada](20260915/verification/closeout-r2/audit.json) y el
[runbook](../../../docs/navarro/autoresearch-v2.md).

El ZIP generado por `studies/package_evidence.py` conserva código, configuración,
dependencias fijadas, especificación, pruebas, fuentes por ensayo, medidas,
intentos fallidos, tablas y figuras. `EVIDENCE-MANIFEST.json` enumera SHA256 de
cada miembro. El empaquetador relee el ZIP y verifica todos sus bytes antes de
declararlo completo; el archivo `.sha256` identifica el contenedor final.

## Reproducibilidad y límites

- La auditoría aritmética y de hashes funciona con Python sin Metal. Extraer el
  paquete, situarse en su raíz y ejecutar:

  ```bash
  python3 -m experiments.navarro.v2.studies.closeout --output /tmp/navarro-audit-new
  ```

- Las tablas se pueden revisar sin instalar MLX. Los experimentos requieren
  Apple Silicon, Metal y el entorno registrado. Los manifests originales incluyen
  rutas absolutas de la máquina de ejecución; no se reescriben para aparentar una
  ejecución nueva. Al repetir entrenamiento se registran destinos y manifest nuevos.
- Los datos, tokenizadores y checkpoints grandes **no están en el ZIP**. Se
  incluyen el manifest de datos y URLs de shards con revisión fijada. Reproducir
  exactamente los tokens requiere además el tokenizer local con los SHA256
  registrados; su historial de entrenamiento no se ha recuperado. El paquete
  permite auditar resultados, pero no es una reproducción autónoma del entrenamiento.
- Los hashes finales de pilotos no permiten reanudar desde esos estados ni medir
  distancias entre pesos. La equivalencia numérica final de la extensión de filas
  a tamaño completo queda pendiente; no se infiere de su NLL cercana.
- Las pruebas previas cuentan 128 aprobadas y la validación posterior del candidato
  SIMD, 33. La extensión conserva su autoprueba de diez updates y los fallos de
  desarrollo. No se presenta la ruta confirmatoria como validada de extremo a extremo.
- No hay intervalos confirmatorios: los pasos de un mismo proceso no son réplicas
  independientes. No mezclar los controles temporales de estudios distintos.

## Volver a empaquetar

```bash
python3 -m experiments.navarro.v2.studies.package_evidence \
  --output /tmp/navarro-v2-evidence-new.zip
```

Los destinos deben ser nuevos. La creación de un paquete no publica ni envía los
archivos a ningún servicio externo. La cadena de hashes detecta cambios respecto
al registro conservado; no sustituye una certificación independiente.
