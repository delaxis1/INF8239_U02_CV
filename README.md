# INF8239_U02_CV · Clasificador visual con CNN

**INF-8239 Ciencia de Datos II · Unidad 02 · Ejercicio 04** · Alexis De La Cruz · Matrícula 100805370
Maestría en Ciencias de Datos e Inteligencia Artificial (UASD)

Construido sobre el proyecto base del curso (autor académico: Edwin Ramón José Nolasco), siguiendo
el laboratorio LAB07.

> ¿Una CNN pequeña supera a una línea base densa en Fashion-MNIST, y cuánto cuesta esa ventaja?

## Resultado principal

**No, la CNN del proyecto base no supera a la línea base**, y además tarda diez veces más en
entrenar. La causa resultó ser una sola decisión de arquitectura.

| Configuración | F1 macro | Parámetros | Entrenamiento | Inferencia | Tamaño |
|---|---|---|---|---|---|
| Línea base densa | 0.8548 | 50 890 | 4.1 s | 0.000 ms | 201.0 KiB |
| CNN con promedio global *(proyecto base)* | 0.7646 | 19 466 | 43.1 s | 0.257 ms | 78.7 KiB |
| **CNN con posición espacial** | **0.9007** | 50 186 | 56.0 s | 0.146 ms | 198.7 KiB |

Conjunto de prueba de 10 000 imágenes · 6 épocas máximo con parada temprana · semilla 42.

La CNN del proyecto base termina en `GlobalAveragePooling2D`: resume cada uno de los 64 mapas de
7×7 en un solo número, de modo que conserva **qué** rasgos aparecen y descarta **dónde**. Al
sustituir ese promedio por un aplanado —sin tocar las capas convolucionales— el F1 macro sube
13.6 puntos. El costo está casi todo en la última capa.

## Verificación en los dos marcos

El laboratorio usa TensorFlow. Durante el desarrollo, Smart App Control de Windows bloqueó
temporalmente la carga de sus DLL, así que las arquitecturas se replicaron en PyTorch conservando
partición, semilla, tamaño de lote, épocas y parada temprana. Más tarde TensorFlow volvió a
funcionar y **se ejecutó también el script original**, con resultados equivalentes:

| Configuración | PyTorch | TensorFlow |
|---|---|---|
| Línea base densa | 0.8548 | 0.8601 |
| CNN con promedio global | 0.7646 | 0.7761 |

Que el hallazgo se repita en dos bibliotecas distintas refuerza que es un efecto de la
arquitectura y no un accidente de implementación. Las métricas de TensorFlow están en
`reports/cv_metrics_tensorflow.json` y sus figuras llevan el sufijo `_tensorflow`.

## Datos

| | |
|---|---|
| Conjunto | Fashion-MNIST · 70 000 imágenes de 28×28 en gris, 10 clases |
| Fuente | [Zalando Research](https://github.com/zalandoresearch/fashion-mnist) · licencia MIT |
| Partición | 54 000 entrenamiento · 6 000 validación · 10 000 prueba, semilla 42 |
| Trazabilidad | El script descarga los cuatro archivos IDX originales y calcula su SHA-256 |
| Preprocesamiento | Valores de 0 a 255 llevados a [0, 1]; sin aumento de datos |

Los datos no se versionan: se descargan al ejecutar el script.

## Instalación y ejecución

Requiere [uv](https://docs.astral.sh/uv/).

```bash
uv python install 3.12
uv sync --extra cpu
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv run pytest -q
```

```bash
uv run python scripts/train_torch.py --epochs 6    # las tres configuraciones en PyTorch
uv run python scripts/analyze_errors.py            # errores por clase, sin reentrenar
uv run python scripts/train_cv.py --epochs 6       # versión en TensorFlow del proyecto base
```

El entrenamiento completo tarda unos dos minutos en CPU. `analyze_errors.py` reutiliza los pesos
guardados, así que el análisis se repite en segundos.

## Qué genera

| Archivo | Contenido |
|---|---|
| `reports/cv_metrics.json` | Métricas de las tres configuraciones, huellas SHA-256 y partición |
| `reports/curvas_aprendizaje.png` | Pérdida y exactitud en validación, época a época |
| `reports/confusion_cnn.png` | Matriz de confusión del modelo elegido |
| `reports/errores_por_clase.csv` | Errores, tasa y confusión principal de cada clase |
| `reports/cnn_errors.png` | Dieciséis errores reales, con etiqueta y predicción |
| `reports/conclusion.md` | Conclusión interpretada |
| `models/*.pt` | Pesos de las tres configuraciones |

## Errores por clase

| Clase | Errores | Tasa | Se confunde con |
|---|---|---|---|
| Camisa | 270 | 27.0 % | Camiseta |
| Camiseta | 211 | 21.1 % | Camisa |
| Abrigo | 160 | 16.0 % | Camisa |
| Pulóver | 136 | 13.6 % | Abrigo |
| Vestido | 72 | 7.2 % | Camisa |
| Zapatilla | 45 | 4.5 % | Botín |
| Botín | 34 | 3.4 % | Zapatilla |
| Bolso | 27 | 2.7 % | Camisa |
| Pantalón | 21 | 2.1 % | Vestido |
| Sandalia | 17 | 1.7 % | Zapatilla |

Las cuatro prendas de torso concentran los errores. A 28×28 píxeles y en escala de grises, lo que
distingue una camisa de una camiseta —cuello, botones— ocupa unos pocos píxeles.

## Pruebas

```bash
uv run pytest -q
```

| Archivo | Qué comprueba |
|---|---|
| `tests/test_data.py` | Normalización y validación de forma y rango de las imágenes |
| `tests/test_model_contract.py` | Contrato de los modelos del proyecto base |
| `tests/test_variant.py` | Que la variante cambie el promedio global por aplanado, que comparta las capas convolucionales con la base, que por eso tenga más parámetros y que ambas salgan en 10 clases |

## Model Card

[`MODEL_CARD.md`](MODEL_CARD.md) documenta el modelo elegido: uso previsto, usos fuera de alcance,
particiones, métricas por clase, costo, limitaciones y plan de monitoreo.

## Estructura

```
INF8239_U02_CV/
├── MODEL_CARD.md              ficha del modelo elegido
├── docs/USO_DE_IA.md          declaración de uso de IA
├── models/                    pesos entrenados (no versionados)
├── reports/                   métricas, curvas, matriz, errores y conclusión
├── scripts/
│   ├── train_cv.py            versión TensorFlow del proyecto base, sin modificar
│   ├── train_torch.py         las tres configuraciones en PyTorch
│   ├── train_variant.py       variante aislada en TensorFlow
│   ├── analyze_errors.py      errores por clase reutilizando los pesos
│   └── check_runtime.py       comprobación del entorno
├── src/inf8239_u02_cv/        normalización, validación y modelos
├── tests/
├── pyproject.toml
└── uv.lock
```

## Entorno

Windows 11 · Intel Core i9-13980HX con 16 GB de RAM · Python 3.12 gestionado por uv ·
PyTorch 2.14.1 y TensorFlow 2.21, ambos en CPU.

## Declaración de uso de asistentes de IA

Este proyecto se desarrolló con la asistencia de Claude Opus 5 (Anthropic). El detalle está en
[`docs/USO_DE_IA.md`](docs/USO_DE_IA.md), como exige el mandato.

## Licencia y uso

Código académico para INF-8239. Fashion-MNIST se usa bajo licencia MIT, citando: Xiao, H.,
Rasul, K., y Vollgraf, R. (2017). *Fashion-MNIST: a novel image dataset for benchmarking machine
learning algorithms*. arXiv:1708.07747.
