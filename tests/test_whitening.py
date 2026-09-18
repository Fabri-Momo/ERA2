"""Whitening numerics: covariance ~ identity, degenerate data, PCA signs."""
import numpy as np
import pytest

from processors.utils import (
    estimate_whitening, apply_whitening, whiten, contrast_stretch,
    reshape_scale, check_subset, prepare_whitening_data,
    InsufficientSelectionError, NEUTRAL_VALUE, WHITENING_EPSILON,
)
from processors.zca import Processor as ZCA
from processors.pca import Processor as PCA_
from processors.cholesky import Processor as Chol


def _correlated(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    A = np.array([[1.0, 0.8, 0.3], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    return rng.random((n, 3)) @ A


@pytest.mark.parametrize('method', ['zca', 'pca', 'cholesky'])
def test_covariance_near_identity(method):
    x = _correlated()
    W, mu = estimate_whitening(x, method=method)
    out = apply_whitening(x, W, mu, dtype=np.float64)
    cov = np.cov(out.T)
    # With epsilon regularisation the output covariance approaches I.
    assert np.allclose(cov, np.eye(3), atol=2e-2)


@pytest.mark.parametrize('method', ['zca', 'pca', 'cholesky'])
def test_output_finite(method):
    x = _correlated()
    a, b = whiten(x, x, method=method)
    assert np.isfinite(a).all() and np.isfinite(b).all()


def test_pca_sign_convention_deterministic():
    x = _correlated()
    W1, _ = estimate_whitening(x, method='pca')
    W2, _ = estimate_whitening(x, method='pca')
    np.testing.assert_array_equal(W1, W2)
    # Convention: largest-|component| of each eigenvector is positive.
    _, mu = estimate_whitening(x, method='pca')
    x2 = _correlated(seed=9)  # different data, same rule
    from processors.utils import _centred_covariance
    Sigma, _ = _centred_covariance(x)
    U, _, _ = np.linalg.svd(Sigma)
    from processors.utils import _fix_eigenvector_signs
    Uf = _fix_eigenvector_signs(U)
    for k in range(3):
        i = np.argmax(np.abs(Uf[:, k]))
        assert Uf[i, k] > 0


def test_pca_sign_flip_invariance():
    """If BLAS returns -u instead of u, the convention gives identical W."""
    rng = np.random.default_rng(3)
    U = rng.normal(size=(3, 3))
    U, _ = np.linalg.qr(U)
    from processors.utils import _fix_eigenvector_signs
    np.testing.assert_array_equal(_fix_eigenvector_signs(U),
                                  _fix_eigenvector_signs(-U))


def test_selection_too_small(rgb_image):
    proc = ZCA()
    mask = np.zeros(rgb_image.shape[:2], dtype=bool)
    mask[0, 0] = True
    mask[0, 1] = True
    with pytest.raises(InsufficientSelectionError):
        proc.process(rgb_image, mask=mask)


def test_selection_constant_region(rgb_image):
    proc = ZCA()
    rgb_image[10:40, 10:40] = 0.5  # constant block
    mask = np.zeros(rgb_image.shape[:2], dtype=bool)
    mask[10:40, 10:40] = True
    with pytest.raises(InsufficientSelectionError):
        proc.process(rgb_image, mask=mask)


def test_full_image_constant_is_neutral():
    proc = ZCA()
    flat = np.full((20, 20, 3), 0.4, dtype=np.float32)
    with warnings_caught() as rec:
        out = proc.process(flat)
    assert out.shape == (20, 20, 3)
    np.testing.assert_allclose(out, NEUTRAL_VALUE)
    assert no_runtime_warnings(rec)


def test_single_channel_constant(rgb_image):
    rgb_image[..., 1] = 0.3  # one constant channel
    proc = ZCA()
    with warnings_caught() as rec:
        out = proc.process(rgb_image)
    assert np.isfinite(out).all()
    assert no_runtime_warnings(rec)


def test_alpha_excluded_from_stats():
    """Alpha=0 pixels must not leak into whitening statistics."""
    rng = np.random.default_rng(5)
    img = rng.random((60, 60, 3), dtype=np.float32)
    valid = np.ones((60, 60), dtype=bool)
    img[40:] = 1e30  # absurd values that would dominate stats
    valid[40:] = False
    X, subset, _ = prepare_whitening_data(img, None, valid)
    assert subset.shape[0] == 60 * 40
    assert subset.max() < 2.0


def test_prepare_subset_with_mask_and_valid(rgb_image):
    h, w = rgb_image.shape[:2]
    mask = np.zeros((h, w), bool)
    mask[:60, :80] = True
    valid = np.ones((h, w), bool)
    valid[:60, :40] = False  # half the selection is transparent
    X, subset, user_sel = prepare_whitening_data(rgb_image, mask, valid)
    assert user_sel
    assert subset.shape[0] == 60 * 40


# helpers ------------------------------------------------------------------
import contextlib
import warnings as _warnings


@contextlib.contextmanager
def warnings_caught():
    with _warnings.catch_warnings(record=True) as rec:
        _warnings.simplefilter('always')
        yield rec


def no_runtime_warnings(rec):
    return not any(issubclass(w.category, RuntimeWarning) for w in rec)
