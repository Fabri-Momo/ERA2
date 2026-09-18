import os
import sys

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# A single QApplication for the whole session: QImage/QPixmap need it.
from qt import QtWidgets  # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def rgb_image(rng):
    """Small 8-bit RGB image with real inter-channel structure."""
    img = rng.random((120, 160, 3)).astype(np.float32)
    img[..., 1] = 0.5 * img[..., 0] + 0.5 * img[..., 1]
    img[..., 2] = np.clip(0.3 * img[..., 0] + 0.7 * img[..., 2], 0, 1)
    return img


@pytest.fixture
def tiff16_path(tmp_path):
    """uint16 TIFF holding 256 distinct levels in the narrow range
    30000..30255 — the historical failure case."""
    import cv2
    levels = 30000 + np.arange(256, dtype=np.uint16)
    idx = np.random.default_rng(7).integers(0, 256, size=(128, 96))
    plane = levels[idx]
    bgr = np.dstack([plane, plane, plane]).astype(np.uint16)
    path = str(tmp_path / 'narrow16.tif')
    cv2.imwrite(path, bgr)
    return path
