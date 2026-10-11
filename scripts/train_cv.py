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
from inf8239_u02_cv.models import build_cnn, build_dense

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def compile_model(model):
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    reports = ROOT / "reports"
    models_dir = ROOT / "models"
    reports.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    (x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train_all = normalize_images(x_train_all)
    x_test = normalize_images(x_test)
    x_train, y_train = x_train_all[:-6000], y_train_all[:-6000]
    x_valid, y_valid = x_train_all[-6000:], y_train_all[-6000:]
    validate_images(x_train, y_train)
    validate_images(x_test, y_test)
    metrics = {}
    trained = {}
    for name, factory in {"dense": build_dense, "cnn": build_cnn}.items():
        model = compile_model(factory(tf))
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)]
        if name == "cnn":
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(models_dir / "best_cnn.keras", save_best_only=True))
        start = perf_counter()
        model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=args.epochs,
                  batch_size=128, callbacks=callbacks, verbose=2)
        train_seconds = perf_counter() - start
        start = perf_counter()
        probabilities = model.predict(x_test, verbose=0)
        inference_seconds = perf_counter() - start
        predictions = probabilities.argmax(axis=1)
        metrics[name] = {
            "f1_macro": float(f1_score(y_test, predictions, average="macro")),
            "parameters": int(model.count_params()),
            "train_seconds": train_seconds,
            "inference_ms_per_image": 1000 * inference_seconds / len(x_test),
        }
        trained[name] = (model, predictions)
        print(name, json.dumps(metrics[name], indent=2))
    _, predictions = trained["cnn"]
    print(classification_report(y_test, predictions, digits=3))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap="Blues")
    plt.savefig(reports / "confusion_cnn.png", dpi=170, bbox_inches="tight")
    wrong = np.where(predictions != y_test)[0][:16]
    figure, axes = plt.subplots(4, 4, figsize=(8, 8))
    for axis, index in zip(axes.ravel(), wrong):
        axis.imshow(x_test[index].squeeze(), cmap="gray")
        axis.set_title(f"R:{y_test[index]} P:{predictions[index]}")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(reports / "cnn_errors.png", dpi=170)
    (reports / "cv_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
