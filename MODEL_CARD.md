# Model Card · Fashion-MNIST CNN

**INF-8239 Ciencia de Datos II · Unidad 02 · Ejercicio 04** · Alexis De La Cruz · 100805370

## Modelo y versión

**`cnn_flat`**, versión 1.0, entrenado el 10 de octubre de 2026.

Red convolucional pequeña para clasificar imágenes de prendas de vestir en 10 categorías. Dos
bloques convolucionales (32 y 64 filtros de 3×3, cada uno seguido de *max pooling* 2×2), después
**aplanado** de la rejilla resultante de 7×7×64, *dropout* de 0.25 y una capa densa de 10 salidas.

La diferencia con la CNN del proyecto base está en una sola decisión: esa base resume cada mapa de
características con un **promedio global**, que conserva qué rasgos aparecen y descarta dónde. Esta
versión conserva la posición. Los pesos están en `models/cnn_flat.pt`.

## Uso previsto

Ejercicio académico de clasificación de imágenes sobre el conjunto Fashion-MNIST. Sirve para
comparar arquitecturas y para ilustrar el efecto de descartar la información espacial.

## Usos fuera de alcance

- **Fotografías reales de ropa.** Fashion-MNIST son recortes centrados, de 28×28, en escala de
  grises y sin fondo. Nada de lo medido aquí anticipa el rendimiento con fotos de catálogo o de
  teléfono.
- **Decisiones automáticas sobre inventario, precios o devoluciones.** Dos de cada diez camisas se
  clasifican mal; ninguna decisión sin revisión humana debería apoyarse en eso.
- **Cualquier dominio distinto de las diez clases entrenadas.** El modelo siempre devuelve una de
  las diez, también ante una imagen que no sea una prenda.

## Dataset y particiones

| | |
|---|---|
| Conjunto | Fashion-MNIST · 70 000 imágenes de 28×28 en escala de grises, 10 clases |
| Fuente | Repositorio oficial de Zalando Research · licencia MIT |
| Obtención | `scripts/train_torch.py` descarga los cuatro archivos IDX originales y calcula su SHA-256 |
| Entrenamiento | 54 000 imágenes |
| Validación | 6 000 imágenes (las últimas del bloque de entrenamiento) |
| Prueba | 10 000 imágenes, usadas una sola vez al final |
| Semilla | 42 |

Huellas SHA-256 de los archivos originales, primeros 16 caracteres:

| Archivo | Huella |
|---|---|
| `train-images-idx3-ubyte.gz` | `3aede38d61863908` |
| `train-labels-idx1-ubyte.gz` | `a04f17134ac03560` |
| `t10k-images-idx3-ubyte.gz` | `346e55b948d973a9` |
| `t10k-labels-idx1-ubyte.gz` | `67da17c76eaffca5` |

## Preprocesamiento

Los valores de 0 a 255 se dividen entre 255 para quedar en el intervalo [0, 1]. Nada más: sin
aumento de datos, sin recortes y sin estandarización por canal, para que la comparación entre las
tres configuraciones mida solo el efecto de la arquitectura. La forma se reordena a
(N, 1, 28, 28), que es lo que espera PyTorch.

## Métricas globales y por clase

Exactitud 0.901 · **F1 macro 0.9007** · 10 000 imágenes de prueba.

| Clase | F1 | Errores | Tasa de error | Se confunde sobre todo con |
|---|---|---|---|---|
| Camisa | 0.722 | 270 | 27.0 % | Camiseta |
| Camiseta | 0.839 | 211 | 21.1 % | Camisa |
| Abrigo | 0.838 | 160 | 16.0 % | Camisa |
| Pulóver | 0.850 | 136 | 13.6 % | Abrigo |
| Vestido | 0.897 | 72 | 7.2 % | Camisa |
| Zapatilla | 0.955 | 45 | 4.5 % | Botín |
| Botín | 0.965 | 34 | 3.4 % | Zapatilla |
| Bolso | 0.979 | 27 | 2.7 % | Camisa |
| Pantalón | 0.984 | 21 | 2.1 % | Vestido |
| Sandalia | 0.979 | 17 | 1.7 % | Zapatilla |

Las cuatro prendas de torso concentran los errores. A 28×28 píxeles y en gris, lo que distingue una
camisa de una camiseta —botones, cuello, mangas— ocupa unos pocos píxeles y se pierde. El calzado y
los bolsos, de silueta inconfundible, rondan el 2 %.

## Comparación de costo

- **Hardware:** Intel Core i9-13980HX, 16 GB de RAM, solo CPU. Windows 11, Python 3.12, PyTorch 2.14.
- **Parámetros:** 50 186, frente a 19 466 de la CNN con promedio global y 50 890 de la línea base densa.
- **Tiempo de entrenamiento:** 56.0 s con 6 épocas, frente a 43.1 s de la CNN base y 4.1 s de la densa.
- **Tiempo de inferencia:** 0.146 ms por imagen, frente a 0.257 de la CNN base.
- **Tamaño del modelo:** 198.7 KiB, frente a 78.7 y 201.0 KiB.

| Configuración | F1 macro | Parámetros | Entrenamiento | Inferencia | Tamaño |
|---|---|---|---|---|---|
| Línea base densa | 0.8548 | 50 890 | 4.1 s | 0.0002 ms | 201.0 KiB |
| CNN con promedio global | 0.7646 | 19 466 | 43.1 s | 0.257 ms | 78.7 KiB |
| **CNN con posición espacial** | **0.9007** | 50 186 | 56.0 s | 0.146 ms | 198.7 KiB |

Todo el aumento de parámetros está en la última capa; el extractor de rasgos es idéntico. La
inferencia de la línea base es tan rápida que su medición queda cerca del ruido: conviene leerla
como «dos órdenes de magnitud por debajo» y no como una cifra exacta.

## Limitaciones y riesgos

- **El entrenamiento se detuvo a las 6 épocas** y la variante aún mejoraba. 0.9007 es un piso, no
  un techo.
- **Sin aumento de datos ni regularización más allá del dropout.** Un modelo entrenado más tiempo
  requeriría volver a medir.
- **El par camisa-camiseta es el riesgo principal.** Cualquier uso debería tratarlas como una sola
  categoría o exigir revisión.
- **Sesgo del conjunto:** imágenes centradas y normalizadas por Zalando, que no representan la
  variedad de fondos, poses, iluminación ni tallas de una fotografía real.
- **No se midió consumo energético**, solo tiempo y tamaño.

## Supervisión y monitoreo

Si este modelo se pusiera en uso, vigilaría tres cosas: la proporción de predicciones en el par
camisa-camiseta, que es donde se concentra el error; la distribución de las confianzas máximas, cuya
caída indicaría imágenes fuera del dominio; y el rendimiento sobre una muestra etiquetada a mano
cada cierto tiempo, porque un catálogo de ropa cambia de estilo con cada temporada. Reentrenaría al
detectar una caída sostenida o al incorporar categorías nuevas.
