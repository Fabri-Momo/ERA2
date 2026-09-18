"""Contrast boost: exact tail fractions, degenerate channels, no warnings."""
import numpy as np
import pytest
import warnings

from processors.utils import contrast_stretch, reshape_scale, NEUTRAL_VALUE


def _data(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3))


@pytest.mark.parametrize('boost,tail', [(0, 0.0), (1, 0.005), (20, 0.10)])
def test_percentiles_exact(boost, tail):
    a = _data()
    subset = a.copy()
    out = contrast_stretch(a.copy(), subset, boost)
    assert out.dtype == np.float32
    for ch in range(3):
        low = np.quantile(subset[:, ch], tail)
        high = np.quantile(subset[:, ch], 1 - tail)
        expected = np.clip((np.clip(a[:, ch], low, high) - low) / (high - low),
                           0, 1)
        np.testing.assert_allclose(out[:, ch], expected, atol=1e-6)


def test_output_in_unit_range():
    out = contrast_stretch(_data(), _data(seed=2), 5)
    assert out.min() >= 0 and out.max() <= 1


def test_uniform_channel_neutral_no_warnings():
    a = np.ones((100, 3))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = contrast_stretch(a, a, 0)
    np.testing.assert_allclose(out, NEUTRAL_VALUE)
    assert not any(issubclass(w.category, RuntimeWarning) for w in rec)
    assert np.isfinite(out).all()


def test_one_constant_channel():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(500, 3))
    a[:, 1] = 7.0
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = contrast_stretch(a, a, 0)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[:, 1], NEUTRAL_VALUE)
    assert not any(issubclass(w.category, RuntimeWarning) for w in rec)


def test_very_low_variance_channel():
    a = np.random.default_rng(4).normal(size=(500, 3)) * 1e-15
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = contrast_stretch(a, a, 0)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, NEUTRAL_VALUE)


def test_reshape_scale_dtype():
    a = _data()
    out = reshape_scale(a, a, 40, 50, 0)
    assert out.shape == (40, 50, 3)
    assert out.dtype == np.float32
