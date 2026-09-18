"""Baseline characterization of the HISTORICAL ERA1 behaviour.

This script was run BEFORE the stabilization changes. It documents the
pre-stabilization pipeline (uint8 load, eager processing, historical
FastICA defaults) and is kept as a frozen record: it is NOT expected to
run against the current code. Its products are kept in the repository:

- baseline_report.txt  — measured historical behaviour;
- baseline_outputs.npz — reference outputs of the historical pipeline on
  small 8-bit images, used by tests/test_compat.py.
"""
import os
import sys
import warnings
import numpy as np
import cv2

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'baseline_outputs.npz')
REPORT = os.path.join(HERE, 'baseline_report.txt')

lines = []


def log(msg=''):
    print(msg)
    lines.append(str(msg))


def make_test_images(tmp):
    rng = np.random.default_rng(1234)
    # 16-bit TIFF: 256 distinct levels in a very narrow range
    levels = (30000 + np.arange(256, dtype=np.uint16))
    img16 = levels[np.random.default_rng(7).integers(0, 256, size=(128, 96))]
    bgr16 = np.dstack([img16, 30000 + (img16 - 30000), img16]).astype(np.uint16)
    p16 = os.path.join(tmp, 'test16.tif')
    cv2.imwrite(p16, bgr16)

    # 8-bit reference image (colour noise + gradient)
    h, w = 120, 160
    img8 = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    img8[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    p8 = os.path.join(tmp, 'test8.png')
    cv2.imwrite(p8, cv2.cvtColor(img8, cv2.COLOR_RGB2BGR))

    # 16-bit RGBA TIFF
    alpha = np.full((64, 64), 65535, dtype=np.uint16)
    alpha[:, :32] = 0
    bgra16 = np.dstack([img16[:64, :64]] * 3 + [alpha]).astype(np.uint16)
    p16a = os.path.join(tmp, 'test16a.tif')
    cv2.imwrite(p16a, bgra16)
    return p16, p8, p16a


def main():
    import tempfile
    tmp = tempfile.mkdtemp(prefix='era_baseline_')
    p16, p8, p16a = make_test_images(tmp)

    log('# ERA1 baseline report')
    log(f'python {sys.version.split()[0]}, cv2 {cv2.__version__}, numpy {np.__version__}')
    import sklearn
    log(f'sklearn {sklearn.__version__}')
    log()

    # ---- 1. 16-bit loading --------------------------------------------------
    log('## 1. 16-bit TIFF loading (current: cv2.imread without flags)')
    loaded = cv2.imread(p16)
    log(f'  cv2.imread(path) dtype={loaded.dtype}, unique levels={len(np.unique(loaded[..., 0]))}')
    unchanged = cv2.imread(p16, cv2.IMREAD_UNCHANGED)
    log(f'  IMREAD_UNCHANGED   dtype={unchanged.dtype}, unique levels={len(np.unique(unchanged[..., 0]))}')

    loaded_a = cv2.imread(p16a)
    log(f'  RGBA via imread(): shape={loaded_a.shape} dtype={loaded_a.dtype}')
    unchanged_a = cv2.imread(p16a, cv2.IMREAD_UNCHANGED)
    log(f'  RGBA UNCHANGED:    shape={unchanged_a.shape} dtype={unchanged_a.dtype}')
    log()

    # ---- 2. new_range on degenerate data ------------------------------------
    log('## 2. new_range() on degenerate data')
    from processors.utils import new_range
    a = np.ones((50, 3), dtype=np.float64) * 7
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = new_range(a.copy(), a.copy(), 0)
    log(f'  uniform input -> warnings: {[str(w.message) for w in rec]}')
    log(f'  uniform output unique: {np.unique(out)}, any NaN in float stage: {np.isnan(a).any()}')

    rng = np.random.default_rng(0)
    b = rng.normal(0, 1, (200, 3))
    b[:, 1] = 5.0  # constant channel
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out2 = new_range(b.copy(), b.copy(), 0)
    log(f'  constant channel -> warnings: {[str(w.message) for w in rec]}')
    log(f'  constant channel output col1 unique: {np.unique(out2[:, 1])}')
    log()

    # ---- 3. FastICA reproducibility -----------------------------------------
    log('## 3. FastICA defaults / reproducibility')
    from sklearn.decomposition import FastICA
    import inspect
    sig = inspect.signature(FastICA.__init__)
    log(f'  FastICA signature: {sig}')
    X = rng.normal(size=(2000, 3)) @ np.array([[1, .8, .2], [0, 1, .5], [0, 0, 1]])
    r1 = FastICA(n_components=3).fit_transform(X)
    r2 = FastICA(n_components=3).fit_transform(X)
    log(f'  two default runs identical: {np.allclose(r1, r2)}')
    log()

    # ---- 4. Import discovery -------------------------------------------------
    log('## 4. Plugin discovery')
    from processors import get_processors
    from color_interpreters import get_color_interpreters
    procs = get_processors()
    interps = get_color_interpreters()
    log(f'  processors: {[p.name for p in procs]}')
    log(f'  interpreters: {[i.name for i in interps]}')
    log()

    # ---- 5. Memory footprint of eager pipeline -------------------------------
    log('## 5. Memory estimate')
    H, W = 4000, 5000  # 20 MP
    n = len(procs) * len(interps)
    log(f'  combos = {n}; per combo uint8 buffer = {H*W*3/1e6:.0f} MB + 3 channel copies = {H*W/1e6:.0f} MB each')
    log(f'  eager total ~= {n * H*W*3*2/1e9:.1f} GB (buffers + channel copies), + QImages')
    log()

    # ---- 6. SVG contours -------------------------------------------------------
    log('## 6. findContours on current mask convention (figure=0 black, bg=255)')
    mask = np.full((100, 100), 255, np.uint8)
    cv2.rectangle(mask, (30, 30), (60, 60), 0, -1)  # figure = black square
    contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    log(f'  contours found on raw mask: {len(contours)}')
    for c in contours:
        x, y, wc, hc = cv2.boundingRect(c)
        log(f'    bounding rect: x={x} y={y} w={wc} h={hc}')
    fg = 255 - mask
    contours2, _ = cv2.findContours(fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    log(f'  contours on inverted (fg=255): {len(contours2)}')
    for c in contours2:
        x, y, wc, hc = cv2.boundingRect(c)
        log(f'    bounding rect: x={x} y={y} w={wc} h={hc}')
    log()

    # ---- 7. Historical pipeline outputs (compat reference) --------------------
    log('## 7. Saving historical pipeline outputs for compat tests')
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    from qt import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from imagedata import ImageData
    data = ImageData()
    data.load(p8)
    outs = {}
    for proc, d in data.processed.items():
        for cs, bundle in d.items():
            outs[f'{proc}__{cs}'] = bundle.buffer
    outs['reference'] = data.reference.buffer
    np.savez_compressed(OUT, **outs)
    log(f'  saved {len(outs)} arrays to {OUT}')

    # keep=5 (contrast boost) variant too
    data.update(mask=None, keep=5)
    outs5 = {}
    for proc, d in data.processed.items():
        for cs, bundle in d.items():
            outs5[f'{proc}__{cs}'] = bundle.buffer
    np.savez_compressed(os.path.join(HERE, 'baseline_outputs_boost5.npz'), **outs5)
    log(f'  saved keep=5 variant ({len(outs5)} arrays)')
    log()

    with open(REPORT, 'w') as f:
        f.write('\n'.join(lines))
    print(f'\nReport written to {REPORT}')


if __name__ == '__main__':
    main()
