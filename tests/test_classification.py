"""Supervised classification robustness: t-test NaN, supervision checks."""
import numpy as np
import pytest

import main as era_main
from imagedata import ImageData


def test_overlap_detected():
    m = np.ones((20, 20), bool)
    assert 'overlap' in era_main.check_supervision(m, m.copy())


def test_too_small_detected():
    a = np.zeros((20, 20), bool)
    b = np.zeros((20, 20), bool)
    a[:4, :4] = True   # 16 px < MIN_SUPERVISION_PIXELS
    b[10:, 10:] = True
    assert 'too small' in era_main.check_supervision(a, b)


def test_supervision_ok():
    a = np.zeros((40, 40), bool)
    b = np.zeros((40, 40), bool)
    a[:20, :20] = True
    b[20:, 20:] = True
    assert era_main.check_supervision(a, b) is None


def test_ttest_nan_channels():
    rng = np.random.default_rng(0)
    case = rng.random((60, 9))
    notc = rng.random((60, 9))
    case[:, 3] = 5.0
    notc[:, 3] = 5.0   # identical constants -> NaN statistic
    case[:, 4] = 0.0
    notc[:, 4] = 1.0   # constant but different -> Inf statistic
    X, y, selected = era_main.prepare_train_data(case, notc, 5)
    assert np.isfinite(X).all()
    assert selected.sum() == 5
    assert selected[4]              # perfectly discriminative channel kept
    assert not selected[3]          # NaN channel ranked last


def test_n_best_capped():
    rng = np.random.default_rng(0)
    case = rng.random((60, 9))
    notc = rng.random((60, 9))
    _, _, selected = era_main.prepare_train_data(case, notc, 500)
    assert selected.sum() == 9      # cannot exceed real channel count


class _Worker:
    def report(self, s, t): pass
    def isInterruptionRequested(self): return False


def _loaded_data(rgb_image):
    data = ImageData()
    data.source_float = rgb_image
    data.raw_display = (rgb_image * 255).astype(np.uint8)
    data.raw_source = data.raw_display
    data.valid_mask = np.ones(rgb_image.shape[:2], bool)
    data._compute_whitened(mask=None, contrast_boost=0)
    return data


def test_end_to_end_lr(rgb_image):
    """Full classification path on a small image (LR)."""
    data = _loaded_data(rgb_image)
    h, w = rgb_image.shape[:2]
    case = np.zeros((h, w), bool)
    notc = np.zeros((h, w), bool)
    case[10:30, 10:30] = True
    notc[60:100, 60:100] = True

    out = era_main.run_classification(
        data, case, notc, 'LR',
        dict(blur=3, n_best=10, pca_pct=95, n_segments=100, compactness=10),
        _Worker())
    assert set(out) == {'with confident learning', 'without confident learning',
                        'white background', 'black foreground'}
    for key, (arr, is_color) in out.items():
        assert arr.shape[:2] == (h, w)
        assert np.isfinite(arr).all()


def test_end_to_end_superpixels(rgb_image):
    data = _loaded_data(rgb_image)
    h, w = rgb_image.shape[:2]
    # Scatter supervision over many superpixels: CleanLearning needs
    # >= cv_n_folds labelled superpixels per class.
    case = np.zeros((h, w), bool)
    notc = np.zeros((h, w), bool)
    case[::15, 10:60:15] = True
    notc[::15, 100:150:15] = True
    out = era_main.run_classification(
        data, case, notc, 'Superpixels',
        dict(blur=3, n_best=10, pca_pct=95, n_segments=40, compactness=10),
        _Worker())
    for key, (arr, is_color) in out.items():
        assert arr.shape[:2] == (h, w)
        assert np.isfinite(arr).all()


def test_superpixels_supervision_gap(rgb_image):
    """Supervision covering only one class must error, not crash."""
    data = _loaded_data(rgb_image)
    h, w = rgb_image.shape[:2]
    case = np.zeros((h, w), bool)
    notc = np.zeros((h, w), bool)
    case[10:30, 10:30] = True
    notc[11:12, 11:12] = True  # exclude inside same superpixels as include
    with pytest.raises(Exception):
        era_main.run_classification(
            data, case, notc, 'Superpixels',
            dict(blur=3, n_best=10, pca_pct=95, n_segments=60, compactness=10),
            _Worker())
