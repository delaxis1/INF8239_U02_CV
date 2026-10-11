# INF-8239 · Unidad 02 · Proyecto de visión

Autor académico: Edwin Ramón José Nolasco

## CPU
```bash
uv python install 3.12
uv sync --extra cpu
uv run python scripts/check_runtime.py
uv run python scripts/train_cv.py --epochs 8
```

## GPU
La ruta GPU se utiliza solamente en WSL2/Linux con NVIDIA correctamente configurada:
```bash
uv sync --extra gpu
```

Complete `MODEL_CARD.md` después de analizar métricas, errores y costo.
