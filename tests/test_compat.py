"""Compatibility with historical ERA outputs (baseline_outputs.npz).

Differences caused by removing the intermediate 8-bit quantisation are
expected.  PCA eigenvector signs and ICA component order/sign are
canonicalised in the new code, so channels may legitimately be sign-flipped
or reordered versus a historical run.  What must NOT happen: RGB/BGR
permutation, axis flips, wrong percentiles, wrong whitening formula.

Strategy: match each historical pseudo-RGB channel (the `__RGB` buffers) to
the new whitened channels allowing sign flips and permutations, then verify
every colour-space display agrees once that canonicalisation is applied.
"""
import os

import cv2
import numpy as np
import pytest

from imagedata import ImageData, color_interpreters_dict

BASELINE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'baseline_outputs.npz')


def _baseline_image(tmp_path):
    """Regenerate the exact 8-bit image used by baseline_check.py."""
    rng = np.random.default_rng(1234)
    h, w = 120, 160
    img8 = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    img8[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    p = str(tmp_path / 'test8.png')
    cv2.imwrite(p, cv2.cvtColor(img8, cv2.COLOR_RGB2BGR))
    return p


def _corr(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 1.0 if np.abs(a - a.mean()).max() < 1e-9 and \
            np.abs(b - b.mean()).max() < 1e-9 else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _match_channels(old, new):
    """Return (perm, flips): old[:, :, i] ~= new[:, :, perm[i]], inverted
    (1 - x) when flips[i] is True — a sign flip before min/max stretching
    maps the stretched channel x to 1 - x."""
    n = old.shape[2]
    corr_signed = np.array(
        [[_corr(old[..., i], new[..., j]) for j in range(n)]
         for i in range(n)])
    # inverted comparison: 1 - x has correlation -corr(x)
    corr = np.abs(corr_signed)
    perm = np.zeros(n, dtype=int)
    flips = np.zeros(n, dtype=bool)
    remaining = set(range(n))
    for i in np.argsort(-corr.max(axis=1)):
        j = max(remaining, key=lambda j: corr[i, j])
        remaining.discard(j)
        perm[i] = j
        flips[i] = corr_signed[i, j] < 0
    return perm, flips, corr


@pytest.mark.skipif(not os.path.isfile(BASELINE),
                    reason='baseline_outputs.npz missing (run baseline_check.py)')
def test_historical_compat(tmp_path):
    saved = np.load(BASELINE)
    data = ImageData()
    data.load(_baseline_image(tmp_path))

    # reference display = original image, unchanged
    np.testing.assert_array_equal(data.raw_display, saved['reference'])

    failures = []
    for proc in data.processor_names:
        old_rgb = saved[f'{proc}__RGB']
        new_rgb = data.whitened[proc]
        perm, flips, corr = _match_channels(old_rgb, new_rgb)

        # every historical channel must match a new channel almost exactly
        for i in range(3):
            if corr[i, perm[i]] < 0.95:
                failures.append((proc, 'RGB', i,
                                 round(float(corr[i, perm[i]]), 3)))

        # apply the matched canonicalisation, then check every colour space
        corrected = new_rgb[..., perm].astype(np.float32).copy()
        for i in range(3):
            if flips[i]:
                corrected[..., i] = 1.0 - corrected[..., i]
        for cs in data.colorspace_names:
            key = f'{proc}__{cs}'
            if key not in saved.files and f'{proc}__CMY(K)' in saved.files \
                    and cs == 'CMY':
                key = f'{proc}__CMY(K)'
            if key not in saved.files:
                continue
            interp = color_interpreters_dict[cs]
            new_disp = interp.to_display(interp.interpret(corrected))
            old_disp = saved[key]
            for ch in range(3):
                c = _corr(old_disp[..., ch], new_disp[..., ch])
                # Hue-like first channels of HLS/HSV are numerically unstable
                # near grey: use a looser bound there.
                threshold = 0.8 if (cs in ('HLS', 'HSV') and ch == 0) else 0.95
                if c < threshold:
                    failures.append((proc, cs, ch, round(c, 3)))

    assert not failures, f'channels diverged vs historical: {failures}'
