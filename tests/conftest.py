"""Permite importar los scripts desde las pruebas.

`pyproject.toml` pone `src` en la ruta, pero no la raíz del proyecto, y la variante del
experimento vive en `scripts/train_variant.py`. Añadir la raíz aquí evita tocar la configuración
del proyecto base.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
