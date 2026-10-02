"""Non-ASCII paths: cv2.imread/imwrite cannot handle them on Windows."""
import shutil

import cv2
import numpy as np

from imagedata import load_image
from main import imwrite_unicode


def test_load_image_unicode_path(rgb_image, tmp_path):
    src = tmp_path / 'src.png'
    cv2.imwrite(str(src), cv2.cvtColor(
        (rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    dst = tmp_path / 'anaïs été.png'
    shutil.copy(src, dst)
    rgb, valid = load_image(str(dst))
    assert rgb.shape == rgb_image.shape
    assert valid.all()


def test_imwrite_unicode_roundtrip(tmp_path):
    arr = (np.arange(60, dtype=np.uint8).reshape(10, 6) * 3)
    path = str(tmp_path / 'ï.tif')
    imwrite_unicode(path, arr)
    rgb, _ = load_image(path)
    assert (rgb[..., 0] == arr).all()
