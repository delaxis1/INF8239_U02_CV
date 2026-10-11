"""Analiza los errores del modelo elegido reutilizando el modelo ya entrenado.

Carga los pesos guardados por `train_torch.py` en lugar de volver a entrenar, de modo que el
análisis se puede repetir en segundos. Produce la tabla de errores por clase y una rejilla con
dieciséis fallos reales.

Uso:
    uv run python scripts/analyze_errors.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import classification_report  # noqa: E402

from scripts.train_torch import (CLASES, REPORTS, build_cnn_flat, build_cnn_gap,  # noqa: E402
                                 build_dense, cargar)

MODELOS = Path(__file__).resolve().parents[1] / "models"
CONSTRUCTORES = {"dense": build_dense, "cnn_gap": build_cnn_gap, "cnn_flat": build_cnn_flat}


def main() -> None:
    datos = json.loads((REPORTS / "cv_metrics.json").read_text(encoding="utf-8"))
    ganador = datos["ganador"]

    _, _, x_prueba, y_prueba, _ = cargar()
    modelo = CONSTRUCTORES[ganador]()
    modelo.load_state_dict(torch.load(MODELOS / f"{ganador}.pt"))
    modelo.eval()
    with torch.no_grad():
        predicciones = modelo(x_prueba).argmax(1).numpy()
    y_real = y_prueba.numpy()

    print(f"Modelo analizado: {ganador}\n")
    print(classification_report(y_real, predicciones, target_names=CLASES, digits=3))

    filas = []
    for real in range(10):
        mascara = y_real == real
        fallos = predicciones[mascara] != real
        confundida = (np.bincount(predicciones[mascara][fallos], minlength=10).argmax()
                      if fallos.any() else real)
        filas.append({"clase": CLASES[real], "casos": int(mascara.sum()),
                      "errores": int(fallos.sum()), "tasa_error": round(float(fallos.mean()), 4),
                      "se_confunde_con": CLASES[confundida]})
    filas.sort(key=lambda f: f["tasa_error"], reverse=True)

    destino = REPORTS / "errores_por_clase.csv"
    with destino.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)

    ancho = max(len(f["clase"]) for f in filas)
    print(f"\n{'clase':<{ancho}}  casos  errores  tasa    se confunde con")
    for f in filas:
        print(f"{f['clase']:<{ancho}}  {f['casos']:>5}  {f['errores']:>7}  "
              f"{f['tasa_error']:.3f}   {f['se_confunde_con']}")

    fallos = np.where(predicciones != y_real)[0][:16]
    figura, ejes = plt.subplots(4, 4, figsize=(7.5, 8))
    for eje, indice in zip(ejes.ravel(), fallos):
        eje.imshow(x_prueba[indice].squeeze().numpy(), cmap="gray")
        eje.set_title(f"{CLASES[y_real[indice]]}\n→ {CLASES[predicciones[indice]]}", fontsize=8)
        eje.axis("off")
    figura.suptitle("Errores del modelo elegido: etiqueta real → predicción")
    figura.tight_layout()
    figura.savefig(REPORTS / "cnn_errors.png", dpi=160)
    print(f"\nGuardado: {destino.name} y cnn_errors.png")


if __name__ == "__main__":
    main()
