# Declaración de uso de asistentes de IA

**INF-8239 Ciencia de Datos II · Unidad 02 · Ejercicio 04** · Alexis De La Cruz · 100805370

El mandato exige declarar las herramientas utilizadas, los prompts relevantes, las partes
verificadas y las correcciones realizadas.

## Herramienta

**Claude Opus 5 (Anthropic)**, a través de Claude Code, con acceso al proyecto local y al aula
virtual para leer el mandato, la rúbrica y el capítulo LAB07.

## Instrucciones relevantes

Las instrucciones se recogen **resumidas en su contenido, no transcritas literalmente**.

| Instrucción dada al asistente | Qué determinó |
|---|---|
| Inicio del Ejercicio 04 | Leer el mandato, la rúbrica y el plazo; instalar el proyecto base de visión |
| Solicitud de diagnóstico del bloqueo encontrado al importar TensorFlow | Identificar Smart App Control como causa y evaluar alternativas sin desactivar la protección |

## Qué hizo el asistente

- Diagnosticó el bloqueo de TensorFlow y propuso replicar las arquitecturas en PyTorch.
- Escribió `scripts/train_torch.py`: descarga con huella SHA-256, partición, las tres
  configuraciones, entrenamiento con parada temprana, métricas, curvas, matriz de confusión y
  errores por clase.
- Escribió `scripts/train_variant.py` y las pruebas de `tests/test_variant.py`.
- Redactó los borradores del README, el Model Card, esta declaración y la conclusión.

## Qué verifiqué

- **El bloqueo fue real pero transitorio.** El error exacto fue
  `ImportError: DLL load failed while importing _pywrap_tfe: An Application Control policy has
  blocked this file`. Se decidió no desactivar Smart App Control porque Windows no permite volver a
  activarlo sin reinstalar el sistema. **Al terminar el ejercicio comprobé que TensorFlow volvía a
  importar y a entrenar**: esa protección bloquea un archivo desconocido mientras su servicio de
  reputación lo evalúa, y después lo permite. Por eso ejecuté también el script original del
  proyecto base y comparé ambos marcos.
- **Todas las cifras provienen de ejecutar `scripts/train_torch.py`** en este equipo. No hay ningún
  número escrito a mano en el informe.
- **Las huellas SHA-256** de los cuatro archivos originales de Fashion-MNIST las calcula el propio
  script al descargarlos.
- **Las pruebas** se ejecutan con `uv run pytest -q` y pasan.

## Correcciones durante el trabajo

- El primer intento usó TensorFlow, como el proyecto base. Falló por la política de seguridad, y en
  lugar de forzarla se cambió de biblioteca conservando partición, semilla, tamaño de lote, épocas y
  parada temprana, para que la comparación siguiera siendo válida.
- **Corrección posterior:** al volver a probar, TensorFlow ya funcionaba. En lugar de dejar escrito
  que estaba bloqueado, se ejecutó el script original y se informan los dos resultados. El texto se
  corrigió para decir que el bloqueo fue temporal, no permanente.
- Las pruebas no podían importar desde `scripts/` porque `pyproject.toml` solo pone `src` en la
  ruta. Se añadió un `tests/conftest.py` en lugar de modificar la configuración del proyecto base.
- La CNN del proyecto base quedó por debajo de la línea base densa. No se ocultó ni se reemplazó:
  se añadió una tercera configuración que cambia una sola cosa —conservar la posición espacial— para
  comprobar si esa era la causa, y el resultado se informa tal cual.

## Responsabilidad

El código, los datos, las referencias y las conclusiones son responsabilidad mía. Puedo explicar
cualquier fragmento del proyecto y justificar cada decisión metodológica.
