"""End-to-end pipeline: float everywhere, all colour spaces, 16-bit data."""
import cv2
import numpy as np
import pytest

from imagedata import ImageData, color_interpreters_dict, processors_dict


def test_pipeline_stays_float(rgb_image, tmp_path):
    p = str(tmp_path / 'img.png')
    cv2.imwrite(p, cv2.cvtColor(
        (rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    data = ImageData()
    data.load(p)

    # whitened results: float32 pseudo-RGB in [0,1]
    for name, w in data.whitened.items():
        assert w.dtype == np.float32, name
        assert np.isfinite(w).all(), name
        assert 0.0 <= w.min() and w.max() <= 1.0, name

    # native colour data: float32, finite
    for proc, cs, native in data.iter_natives():
        assert native.dtype == np.float32, (proc, cs)
        assert np.isfinite(native).all(), (proc, cs)
        assert native.shape == rgb_image.shape

    # display data: uint8 — the ONLY quantisation, for display
    for proc in data.processor_names:
        for cs in data.colorspace_names:
            bundle = data.get_bundle(proc, cs)
            assert bundle.display.dtype == np.uint8
            assert bundle.display.shape == rgb_image.shape


def test_16bit_end_to_end(tiff16_path):
    """256 levels survive the whole scientific pipeline (was: 2)."""
    data = ImageData()
    data.load(tiff16_path)
    assert data.raw_source.dtype == np.uint16
    assert len(np.unique(data.raw_source[..., 0])) == 256
    for name, w in data.whitened.items():
        if name == 'FastICA':
            continue  # rank-deficient test image -> neutral by design
        # sub-8-bit detail still present after whitening+stretch
        assert len(np.unique(w[..., 0])) > 100, name


def test_update_with_selection(rgb_image, tmp_path):
    p = str(tmp_path / 'img.png')
    cv2.imwrite(p, cv2.cvtColor(
        (rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    data = ImageData()
    data.load(p)
    mask = np.zeros(rgb_image.shape[:2], bool)
    mask[10:80, 10:80] = True
    data.update(mask=mask, contrast_boost=5)
    for w in data.whitened.values():
        assert np.isfinite(w).all()


def test_reset_clears_everything(rgb_image, tmp_path):
    p = str(tmp_path / 'img.png')
    cv2.imwrite(p, cv2.cvtColor(
        (rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    data = ImageData()
    data.load(p)
    data.reset()
    assert data.raw_source is None
    assert data.raw_display is None
    assert data.reference is None
    assert data.whitened == {}
    assert data.image_path is None
    assert data.valid_mask is None


def test_thumbnail_small(tiff16_path):
    data = ImageData()
    data.load(tiff16_path)
    thumb = data.get_thumbnail('ZCA', 'LAB', max_dim=64)
    assert max(thumb.width(), thumb.height()) <= 64


def test_bundle_cache_bounded(tiff16_path):
    from imagedata import BUNDLE_CACHE_SIZE
    data = ImageData()
    data.load(tiff16_path)
    for proc in data.processor_names:
        for cs in data.colorspace_names:
            data.get_bundle(proc, cs)
    assert len(data._bundles) <= BUNDLE_CACHE_SIZE
