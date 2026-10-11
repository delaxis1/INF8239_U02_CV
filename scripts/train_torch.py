"""Entrena las tres configuraciones de visión en PyTorch.

Por qué PyTorch y no TensorFlow: el proyecto base usa TensorFlow, pero en este equipo Smart App
Control bloquea la carga de `_pywrap_tfe.pyd`, el núcleo nativo de la biblioteca, con el error
«An Application Control policy has blocked this file». Esa protección de Windows 11 solo puede
desactivarse de forma permanente, así que se replicaron las mismas arquitecturas en PyTorch, que el
sistema sí admite. La partición, la semilla, el tamaño de lote y el número de épocas son los mismos
que en `scripts/train_cv.py`.

Las tres configuraciones:

1. `dense`   — línea base: aplanar y una capa oculta de 64 unidades.
2. `cnn_gap` — la CNN del proyecto base: dos bloques convolucionales y promedio global.
3. `cnn_flat`— la misma CNN pero con aplanado, que conserva **dónde** estaba cada rasgo.

Uso:
    uv run python scripts/train_torch.py --epochs 6
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import urllib.request
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score  # noqa: E402
from torch import nn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CRUDOS = ROOT / "data/raw"
REPORTS = ROOT / "reports"
MODELOS = ROOT / "models"
SEMILLA = 42
BASE_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion"
ARCHIVOS = {
    "x_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}
CLASES = ["Camiseta", "Pantalón", "Pulóver", "Vestido", "Abrigo",
          "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botín"]


# --- datos --------------------------------------------------------------------
def descargar() -> dict[str, str]:
    """Descarga los cuatro archivos originales y devuelve su huella SHA-256."""
    CRUDOS.mkdir(parents=True, exist_ok=True)
    huellas = {}
    for clave, nombre in ARCHIVOS.items():
        destino = CRUDOS / nombre
        if not destino.exists():
            urllib.request.urlretrieve(f"{BASE_URL}/{nombre}", destino)
        huellas[clave] = hashlib.sha256(destino.read_bytes()).hexdigest()[:16]
    return huellas


def leer_idx(nombre: str) -> np.ndarray:
    """Lee el formato IDX de Fashion-MNIST: cabecera de enteros grandes y luego los bytes."""
    with gzip.open(CRUDOS / nombre, "rb") as archivo:
        datos = archivo.read()
    dimensiones = datos[3]
    forma = [int.from_bytes(datos[4 + 4 * i:8 + 4 * i], "big") for i in range(dimensiones)]
    return np.frombuffer(datos, dtype=np.uint8, offset=4 + 4 * dimensiones).reshape(forma)


def cargar():
    huellas = descargar()
    x_todo, y_todo = leer_idx(ARCHIVOS["x_train"]), leer_idx(ARCHIVOS["y_train"])
    x_prueba, y_prueba = leer_idx(ARCHIVOS["x_test"]), leer_idx(ARCHIVOS["y_test"])
    # Normalizar a [0, 1] y pasar a (N, canal, alto, ancho), que es lo que espera PyTorch.
    normalizar = lambda a: torch.tensor(a.astype("float32") / 255.0).unsqueeze(1)
    return (normalizar(x_todo), torch.tensor(y_todo.astype("int64")),
            normalizar(x_prueba), torch.tensor(y_prueba.astype("int64")), huellas)


# --- modelos ------------------------------------------------------------------
def build_dense() -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 10))


def _bloques() -> list[nn.Module]:
    return [nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]


def build_cnn_gap() -> nn.Module:
    """La CNN del proyecto base: resume cada mapa con su promedio y pierde la posición."""
    return nn.Sequential(*_bloques(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Dropout(0.25), nn.Linear(64, 10))


def build_cnn_flat() -> nn.Module:
    """La variante: conserva la rejilla 7x7x64 completa antes de decidir."""
    return nn.Sequential(*_bloques(), nn.Flatten(), nn.Dropout(0.25), nn.Linear(7 * 7 * 64, 10))


# --- entrenamiento ------------------------------------------------------------
def entrenar(modelo, x_ent, y_ent, x_val, y_val, epocas: int, lote: int = 128):
    optimizador = torch.optim.Adam(modelo.parameters())
    criterio = nn.CrossEntropyLoss()
    historia = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    mejor, mejor_estado, sin_mejorar = float("inf"), None, 0

    for epoca in range(epocas):
        modelo.train()
        orden = torch.randperm(len(x_ent))
        perdida_total, aciertos = 0.0, 0
        for inicio in range(0, len(orden), lote):
            indices = orden[inicio:inicio + lote]
            entrada, objetivo = x_ent[indices], y_ent[indices]
            optimizador.zero_grad()
            salida = modelo(entrada)
            perdida = criterio(salida, objetivo)
            perdida.backward()
            optimizador.step()
            perdida_total += perdida.item() * len(indices)
            aciertos += (salida.argmax(1) == objetivo).sum().item()
        historia["loss"].append(perdida_total / len(x_ent))
        historia["accuracy"].append(aciertos / len(x_ent))

        modelo.eval()
        with torch.no_grad():
            salida = modelo(x_val)
            perdida_val = criterio(salida, y_val).item()
            historia["val_loss"].append(perdida_val)
            historia["val_accuracy"].append((salida.argmax(1) == y_val).float().mean().item())
        print(f"  época {epoca + 1}/{epocas} · pérdida {historia['loss'][-1]:.4f} · "
              f"validación {perdida_val:.4f} · exactitud val "
              f"{historia['val_accuracy'][-1]:.4f}", flush=True)

        # Parada temprana con paciencia 2, igual que el script de TensorFlow.
        if perdida_val < mejor:
            mejor, mejor_estado, sin_mejorar = perdida_val, {k: v.clone() for k, v in
                                                             modelo.state_dict().items()}, 0
        else:
            sin_mejorar += 1
            if sin_mejorar >= 2:
                print("  parada temprana: la validación no mejoró en dos épocas", flush=True)
                break
    if mejor_estado:
        modelo.load_state_dict(mejor_estado)
    return historia


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()
    REPORTS.mkdir(exist_ok=True)
    MODELOS.mkdir(exist_ok=True)

    random.seed(SEMILLA)
    np.random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)
    torch.use_deterministic_algorithms(False)

    x_todo, y_todo, x_prueba, y_prueba, huellas = cargar()
    # Misma partición que el script base: las últimas 6 000 filas quedan para validación.
    x_ent, y_ent = x_todo[:-6000], y_todo[:-6000]
    x_val, y_val = x_todo[-6000:], y_todo[-6000:]
    print(f"Entrenamiento {len(x_ent)} · validación {len(x_val)} · prueba {len(x_prueba)}")
    print(f"Huellas SHA-256 (16 primeros): {huellas}\n")

    medidas, predicciones, historias = {}, {}, {}
    for nombre, constructor in (("dense", build_dense), ("cnn_gap", build_cnn_gap),
                                ("cnn_flat", build_cnn_flat)):
        print(f"[{nombre}]", flush=True)
        torch.manual_seed(SEMILLA)
        modelo = constructor()
        inicio = perf_counter()
        historias[nombre] = entrenar(modelo, x_ent, y_ent, x_val, y_val, args.epochs)
        segundos = perf_counter() - inicio

        modelo.eval()
        inicio = perf_counter()
        with torch.no_grad():
            salida = modelo(x_prueba)
        ms_por_imagen = 1000 * (perf_counter() - inicio) / len(x_prueba)
        pred = salida.argmax(1).numpy()
        predicciones[nombre] = pred
        medidas[nombre] = {
            "f1_macro": float(f1_score(y_prueba.numpy(), pred, average="macro")),
            "parameters": int(sum(p.numel() for p in modelo.parameters())),
            "train_seconds": segundos,
            "inference_ms_per_image": ms_por_imagen,
        }
        torch.save(modelo.state_dict(), MODELOS / f"{nombre}.pt")
        tamano = (MODELOS / f"{nombre}.pt").stat().st_size / 1024
        medidas[nombre]["size_kb"] = round(tamano, 1)
        print(f"  F1 macro {medidas[nombre]['f1_macro']:.4f} · "
              f"{medidas[nombre]['parameters']:,} parámetros · {segundos:.1f} s · "
              f"{tamano:.0f} KiB\n", flush=True)

    ganador = max(medidas, key=lambda k: medidas[k]["f1_macro"])
    print(f"Mejor configuración: {ganador}\n")
    y_real = y_prueba.numpy()
    print(classification_report(y_real, predicciones[ganador], target_names=CLASES, digits=3))

    (REPORTS / "cv_metrics.json").write_text(
        json.dumps({"modelos": medidas, "ganador": ganador, "huellas_sha256": huellas,
                    "particion": {"entrenamiento": len(x_ent), "validacion": len(x_val),
                                  "prueba": len(x_prueba)}, "semilla": SEMILLA,
                    "epocas_max": args.epochs}, indent=2, ensure_ascii=False), encoding="utf-8")

    # Curvas de aprendizaje de las tres configuraciones.
    figura, ejes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for nombre, historia in historias.items():
        ejes[0].plot(historia["val_loss"], marker="o", label=nombre)
        ejes[1].plot(historia["val_accuracy"], marker="o", label=nombre)
    ejes[0].set_title("Pérdida en validación")
    ejes[1].set_title("Exactitud en validación")
    for eje in ejes:
        eje.set_xlabel("Época")
        eje.legend()
    figura.tight_layout()
    figura.savefig(REPORTS / "curvas_aprendizaje.png", dpi=160)

    ConfusionMatrixDisplay.from_predictions(y_real, predicciones[ganador], display_labels=CLASES,
                                            cmap="Blues", xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    plt.savefig(REPORTS / "confusion_cnn.png", dpi=160, bbox_inches="tight")

    # Errores por clase: cuál falla más y con cuál se confunde.
    filas = []
    for real in range(10):
        mascara = y_real == real
        fallos = predicciones[ganador][mascara] != real
        confundida = (np.bincount(predicciones[ganador][mascara][fallos], minlength=10).argmax()
                      if fallos.any() else real)
        filas.append({"clase": CLASES[real], "casos": int(mascara.sum()),
                      "errores": int(fallos.sum()), "tasa_error": round(float(fallos.mean()), 4),
                      "se_confunde_con": CLASES[confundida]})
    import pandas as pd

    tabla = pd.DataFrame(filas).sort_values("tasa_error", ascending=False)
    tabla.to_csv(REPORTS / "errores_por_clase.csv", index=False, encoding="utf-8")
    print()
    print(tabla.to_string(index=False))

    # Dieciséis errores reales del ganador, para mirarlos.
    fallos = np.where(predicciones[ganador] != y_real)[0][:16]
    figura, ejes = plt.subplots(4, 4, figsize=(7.5, 8))
    for eje, indice in zip(ejes.ravel(), fallos):
        eje.imshow(x_prueba[indice].squeeze().numpy(), cmap="gray")
        eje.set_title(f"{CLASES[y_real[indice]]}\n→ {CLASES[predicciones[ganador][indice]]}",
                      fontsize=8)
        eje.axis("off")
    figura.suptitle("Errores del modelo elegido: etiqueta real → predicción")
    figura.tight_layout()
    figura.savefig(REPORTS / "cnn_errors.png", dpi=160)


if __name__ == "__main__":
    main()
