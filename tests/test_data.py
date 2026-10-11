import numpy as np
import pytest

from inf8239_u02_cv.data import normalize_images, validate_images


def test_normalize_adds_channel_and_scales():
    raw = np.array([[[0] * 28] * 28, [[255] * 28] * 28], dtype=np.uint8)
    result = normalize_images(raw)
    assert result.shape == (2, 28, 28, 1)
    assert result.min() == 0
    assert result.max() == 1


def test_validate_rejects_wrong_shape():
    with pytest.raises(ValueError, match="Forma inesperada"):
        validate_images(np.zeros((2, 28, 28)), np.zeros(2))
