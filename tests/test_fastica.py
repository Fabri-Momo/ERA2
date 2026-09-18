"""FastICA: reproducibility, subsampling, degenerate handling."""
import numpy as np
import pytest

from processors.fastICA import Processor as ICA
from processors.utils import InsufficientSelectionError
from processors.utils import MAX_ICA_TRAINING_PIXELS


def _image(h=120, w=160, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.random((h, w, 3)).astype(np.float32)
    img[..., 1] = 0.6 * img[..., 0] + 0.4 * img[..., 1]
    img[..., 2] = np.clip(0.3 * img[..., 0] + 0.7 * img[..., 2], 0, 1)
    return img


def test_reproducible():
    img = _image()
    ica = ICA()
    np.testing.assert_array_equal(ica.process(img), ica.process(img))


def test_reproducible_across_instances():
    img = _image()
    np.testing.assert_array_equal(ICA().process(img), ICA().process(img))


def test_params_explicit():
    """Guard against silent changes of sklearn defaults."""
    from sklearn.decomposition import FastICA
    import inspect
    ica = ICA()
    # Re-run the constructor call used inside process()
    src = inspect.getsource(ICA.process)
    for token in ["whiten='arbitrary-variance'", 'random_state=',
                  'max_iter=500', 'tol=1e-4', "algorithm='parallel'"]:
        assert token in src


def test_subsampled_fit_consistent():
    """ICA fitted on a deterministic subsample stays close to the full fit."""
    rng = np.random.default_rng(11)
    n = MAX_ICA_TRAINING_PIXELS + 50_000
    # synthesise pixels with independent-component-like structure
    s = rng.random((n, 3))
    A = np.array([[1.0, 0.9, 0.1], [0.0, 1.0, 0.4], [0.0, 0.0, 1.0]])
    X = s @ A
    img = X.reshape(150, -1, 3).astype(np.float32)

    from sklearn.decomposition import FastICA
    full = FastICA(n_components=3, algorithm='parallel',
                   whiten='arbitrary-variance', fun='logcosh',
                   random_state=0, max_iter=500, tol=1e-4).fit(X)
    rng0 = np.random.default_rng(0)
    idx = rng0.choice(n, MAX_ICA_TRAINING_PIXELS, replace=False)
    sub = FastICA(n_components=3, algorithm='parallel',
                  whiten='arbitrary-variance', fun='logcosh',
                  random_state=0, max_iter=500, tol=1e-4).fit(X[idx])
    # Components may differ in sign/order; compare via correlation matrix.
    C = np.corrcoef(full.components_.T, sub.components_.T)[:3, 3:]
    matched = np.abs(C).max(axis=1)
    assert (matched > 0.98).all()


def test_tiny_selection_raises():
    img = _image()
    mask = np.zeros(img.shape[:2], bool)
    mask[0, 0] = True
    with pytest.raises(InsufficientSelectionError):
        ICA().process(img, mask=mask)


def test_constant_selection_raises():
    img = _image()
    img[10:50, 10:50] = 0.5
    mask = np.zeros(img.shape[:2], bool)
    mask[10:50, 10:50] = True
    with pytest.raises(InsufficientSelectionError):
        ICA().process(img, mask=mask)


def test_degenerate_full_image_neutral():
    flat = np.full((30, 30, 3), 0.2, np.float32)
    out = ICA().process(flat)
    np.testing.assert_allclose(out, 0.5)


def test_rank_deficient_full_image_neutral():
    img = np.random.default_rng(0).random((40, 40, 1)).astype(np.float32)
    img = np.dstack([img, img, img])  # rank-1 covariance
    out = ICA().process(img)
    assert np.isfinite(out).all()
