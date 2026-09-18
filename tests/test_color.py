"""Colour spaces: native/display encodings, parity with OpenCV 8-bit path."""
import cv2
import numpy as np
import pytest

from imagedata import color_interpreters_dict


def _rand01(seed=0, n=64):
    return np.random.default_rng(seed).random((8, 8, 3)).astype(np.float32)


EXPECTED = {'RGB', 'XYZ', 'LAB', 'LUV', 'YCrCb', 'YUV', 'HLS', 'HSV', 'CMY'}


def test_all_spaces_present():
    assert set(color_interpreters_dict) == EXPECTED


def test_cmy_renamed():
    assert 'CMY(K)' not in color_interpreters_dict
    assert 'CMY' in color_interpreters_dict


def test_cmy_definition():
    x = _rand01()
    cmy = color_interpreters_dict['CMY']
    np.testing.assert_allclose(cmy.interpret(x), 1.0 - x, atol=1e-6)


@pytest.mark.parametrize('cs', sorted(EXPECTED))
def test_native_finite_and_shaped(cs):
    x = _rand01()
    native = color_interpreters_dict[cs].interpret(x)
    assert native.dtype == np.float32
    assert native.shape == x.shape
    assert np.isfinite(native).all()


@pytest.mark.parametrize('cs', sorted(EXPECTED - {'CMY', 'RGB'}))
def test_display_matches_opencv_8bit(cs):
    """to_display(native float) must match OpenCV's own 8-bit conversion
    (within rounding), i.e. the display looks like the historical output."""
    rng = np.random.default_rng(2)
    x8 = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    xf = x8.astype(np.float32) / 255.0
    interp = color_interpreters_dict[cs]
    legacy = cv2.cvtColor(x8, getattr(cv2, f'COLOR_RGB2{cs}'))
    ours = interp.to_display(interp.interpret(xf))
    diff = np.abs(ours.astype(int) - legacy.astype(int))
    if cs in ('HSV', 'HLS'):
        # Hue is circular on [0, 180]: 0 and 180 are the same colour.
        h = diff[..., 0]
        diff[..., 0] = np.minimum(h, 180 - h)
    assert diff.max() <= 2


def test_native_ranges():
    """Spot-check native conventions: LAB L in [0,100], HSV H in [0,360]."""
    white = np.ones((2, 2, 3), np.float32)
    red = np.zeros((2, 2, 3), np.float32)
    red[..., 0] = 1
    lab = color_interpreters_dict['LAB'].interpret(white)
    assert np.isclose(lab[..., 0], 100).all()
    hsv = color_interpreters_dict['HSV'].interpret(red)
    assert np.isclose(hsv[..., 0], 0).all() and np.isclose(hsv[..., 1], 1).all()
    green = np.zeros((2, 2, 3), np.float32)
    green[..., 1] = 1
    assert np.isclose(color_interpreters_dict['HSV'].interpret(green)[..., 0],
                      120).all()
