"""Entrena una tercera configuración: la misma CNN pero conservando la posición espacial.

La CNN del proyecto base termina en `GlobalAveragePooling2D`, que promedia cada mapa de
características sobre toda la imagen. Eso deja 64 números por imagen y descarta **dónde** estaba
cada rasgo. En Fashion-MNIST la posición importa: un bolso y una camiseta comparten texturas, pero
no la distribución de bordes en el marco de 28x28.

Esta variante sustituye ese promedio por `Flatten`, que conserva la rejilla de 7x7x64 antes de la
capa final. Entrena con la misma partición, la misma semilla y el mismo presupuesto de épocas, de
modo que la comparación mida el efecto de esa única decisión.

Uso:
    uv run python scripts/train_variant.py --epochs 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score

from inf8239_u02_cv.data import normalize_images, validate_images

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
CLASES = ["Camiseta", "Pantalón", "Pulóver", "Vestido", "Abrigo",
          "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botín"]

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_cnn_spatial(tf_module):
    """Misma arquitectura que la CNN base, pero con Flatten en lugar del promedio global."""
    return tf_module.keras.Sequential([
        tf_module.keras.layers.Input((28, 28, 1)),
        tf_module.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf_module.keras.layers.MaxPooling2D(),
        tf_module.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf_module.keras.layers.MaxPooling2D(),
        tf_module.keras.layers.Flatten(),
        tf_module.keras.layers.Dropout(0.25),
        tf_module.keras.layers.Dense(10, activation="softmax"),
    ], name="cnn_spatial")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    reports = ROOT / "reports"
    models_dir = ROOT / "models"
    reports.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    (x_all, y_all), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_all, x_test = normalize_images(x_all), normalize_images(x_test)
    x_train, y_train = x_all[:-6000], y_all[:-6000]
    x_valid, y_valid = x_all[-6000:], y_all[-6000:]
    validate_images(x_train, y_train)
    validate_images(x_test, y_test)

    modelo = build_cnn_spatial(tf)
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    inicio = perf_counter()
    historia = modelo.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                          epochs=args.epochs, batch_size=128, verbose=2,
                          callbacks=[tf.keras.callbacks.EarlyStopping(patience=2,
                                                                      restore_best_weights=True)])
    segundos_entrenamiento = perf_counter() - inicio

    inicio = perf_counter()
    probabilidades = modelo.predict(x_test, verbose=0)
    segundos_inferencia = perf_counter() - inicio
    predicciones = probabilidades.argmax(axis=1)

    medidas = {
        "f1_macro": float(f1_score(y_test, predicciones, average="macro")),
        "parameters": int(modelo.count_params()),
        "train_seconds": segundos_entrenamiento,
        "inference_ms_per_image": 1000 * segundos_inferencia / len(x_test),
    }
    print("cnn_spatial", json.dumps(medidas, indent=2))
    print(classification_report(y_test, predicciones, target_names=CLASES, digits=3))

    # Se añade al archivo existente para que las tres configuraciones queden juntas.
    archivo = reports / "cv_metrics.json"
    todas = json.loads(archivo.read_text(encoding="utf-8")) if archivo.exists() else {}
    todas["cnn_spatial"] = medidas
    archivo.write_text(json.dumps(todas, indent=2), encoding="utf-8")

    # Curvas de aprendizaje: el criterio de la rúbrica pide curvas, no solo cifras finales.
    figura, ejes = plt.subplots(1, 2, figsize=(9, 3.4))
    for eje, clave, titulo in ((ejes[0], "loss", "Pérdida"), (ejes[1], "accuracy", "Exactitud")):
        eje.plot(historia.history[clave], label="entrenamiento")
        eje.plot(historia.history[f"val_{clave}"], label="validación")
        eje.set_xlabel("Época")
        eje.set_title(titulo)
        eje.legend()
    figura.suptitle("CNN con posición espacial · curvas de aprendizaje")
    figura.tight_layout()
    figura.savefig(reports / "curvas_cnn_spatial.png", dpi=170)

    ConfusionMatrixDisplay.from_predictions(y_test, predicciones, display_labels=CLASES,
                                            cmap="Blues", xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(reports / "confusion_cnn_spatial.png", dpi=170, bbox_inches="tight")

    # Errores por clase, para saber cuál confunde con cuál.
    errores = []
    for real in range(10):
        mascara = y_test == real
        fallos = predicciones[mascara] != real
        confundida = (np.bincount(predicciones[mascara][fallos], minlength=10).argmax()
                      if fallos.any() else real)
        errores.append({"clase": CLASES[real], "casos": int(mascara.sum()),
                        "errores": int(fallos.sum()),
                        "tasa_error": float(fallos.mean()),
                        "se_confunde_con": CLASES[confundida]})
    import pandas as pd

    tabla = pd.DataFrame(errores).sort_values("tasa_error", ascending=False)
    tabla.to_csv(reports / "errores_por_clase.csv", index=False, encoding="utf-8")
    print()
    print(tabla.to_string(index=False))
    modelo.save(models_dir / "cnn_spatial.keras")


if __name__ == "__main__":
    main()
