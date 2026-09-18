"""Loading: bit depth preservation, alpha/valid masks, error handling."""
import cv2
import numpy as np
import pytest

from imagedata import load_image, source_to_unit_float, to_display_uint8


def test_tiff16_preserves_levels(tiff16_path):
    rgb, valid = load_image(tiff16_path)
    assert rgb.dtype == np.uint16
    assert rgb.shape == (128, 96, 3)
    # Historical bug: cv2.imread() without flags gave 2 levels.
    assert len(np.unique(rgb[..., 0])) == 256
    assert valid.all()


def test_imread_default_would_lose_levels(tiff16_path):
    """Document the bug this fixes: plain imread destroys the levels."""
    legacy = cv2.imread(tiff16_path)
    assert legacy.dtype == np.uint8
    assert len(np.unique(legacy[..., 0])) <= 4


def test_source_float_keeps_sub8bit(tiff16_path):
    rgb, _ = load_image(tiff16_path)
    f = source_to_unit_float(rgb)
    assert f.dtype == np.float32
    assert len(np.unique(f[..., 0])) == 256
    assert 0 <= f.min() and f.max() <= 1


def test_display_uint8(tiff16_path):
    rgb, _ = load_image(tiff16_path)
    disp = to_display_uint8(rgb)
    assert disp.dtype == np.uint8
    # narrow range -> narrow display, but no crash
    assert disp.max() - disp.min() <= 2


def test_uint8_passthrough(tmp_path):
    img = np.random.default_rng(0).integers(0, 256, (50, 60, 3), dtype=np.uint8)
    p = str(tmp_path / 'i.png')
    cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    rgb, valid = load_image(p)
    assert rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb, img)
    assert valid.all()


def test_grayscale(tmp_path):
    gray = np.random.default_rng(0).integers(0, 256, (40, 40), dtype=np.uint8)
    p = str(tmp_path / 'g.png')
    cv2.imwrite(p, gray)
    rgb, valid = load_image(p)
    assert rgb.shape == (40, 40, 3)
    np.testing.assert_array_equal(rgb[..., 0], gray)
    np.testing.assert_array_equal(rgb[..., 1], gray)


def test_rgba_alpha_mask(tmp_path):
    h, w = 32, 32
    bgra = np.random.default_rng(0).integers(0, 256, (h, w, 4), dtype=np.uint8)
    bgra[:, :16, 3] = 0
    bgra[:, 16:, 3] = 255
    p = str(tmp_path / 'a.png')
    cv2.imwrite(p, bgra)
    rgb, valid = load_image(p)
    assert rgb.shape == (h, w, 3)
    assert valid[:, 16:].all()
    assert not valid[:, :16].any()


def test_rgba_uint16(tmp_path):
    h, w = 24, 24
    bgra = np.random.default_rng(1).integers(0, 65536, (h, w, 4),
                                             dtype=np.uint16)
    bgra[:12, :, 3] = 0
    p = str(tmp_path / 'a16.tif')
    cv2.imwrite(p, bgra)
    rgb, valid = load_image(p)
    assert rgb.dtype == np.uint16
    assert valid.shape == (h, w)
    assert not valid[:12].any() and valid[12:].all()


def test_unreadable_raises(tmp_path):
    with pytest.raises(IOError, match='Unable to read image'):
        load_image(str(tmp_path / 'does_not_exist.tif'))
