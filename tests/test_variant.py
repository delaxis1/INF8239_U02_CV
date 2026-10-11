"""Pruebas de la variante que conserva la posición espacial.

Comprueban la diferencia que define el experimento: dónde termina el mapa de características.
No entrenan nada, así que corren en segundos.
"""

from __future__ import annotations

import pytest

tf = pytest.importorskip("tensorflow", reason="La variante necesita TensorFlow")

from inf8239_u02_cv.models import build_cnn  # noqa: E402
from scripts.train_variant import build_cnn_spatial  # noqa: E402


def test_la_variante_sustituye_el_promedio_global_por_flatten():
    capas = [type(c).__name__ for c in build_cnn_spatial(tf).layers]
    assert "Flatten" in capas
    assert "GlobalAveragePooling2D" not in capas, "ese promedio es justo lo que se quiere evitar"


def test_la_base_si_usa_el_promedio_global():
    capas = [type(c).__name__ for c in build_cnn(tf).layers]
    assert "GlobalAveragePooling2D" in capas, "la comparación pierde sentido si cambia la base"


def test_ambas_comparten_las_capas_convolucionales():
    convolucionales = lambda modelo: [(c.filters, c.kernel_size) for c in modelo.layers
                                      if type(c).__name__ == "Conv2D"]
    assert convolucionales(build_cnn(tf)) == convolucionales(build_cnn_spatial(tf)), \
        "solo debe cambiar la forma de resumir el mapa, no el extractor de rasgos"


def test_conservar_la_posicion_cuesta_mas_parametros():
    base = build_cnn(tf).count_params()
    variante = build_cnn_spatial(tf).count_params()
    assert variante > base, "Flatten conecta 7x7x64 a la salida; el promedio global solo 64"


def test_la_salida_tiene_diez_clases():
    for modelo in (build_cnn(tf), build_cnn_spatial(tf)):
        assert modelo.output_shape == (None, 10)
