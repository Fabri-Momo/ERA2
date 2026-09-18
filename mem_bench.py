"""One-off memory benchmark for the stabilization report (not part of the suite)."""
import os, threading
import numpy as np
import psutil

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

if __name__ == '__main__':
    from qt import QtWidgets
    app = QtWidgets.QApplication([])
    import imagedata, processors, color_interpreters
    import main as era_main

    proc = psutil.Process()
    rss = lambda: proc.memory_info().rss / 1e6

    class W:
        def report(self, s, t): pass
        def isInterruptionRequested(self): return False

    d = imagedata.ImageData()
    r0 = rss()
    d.load(os.path.join(os.path.dirname(__file__), 'Yury.jpg'))
    r1 = rss()
    H, W_ = d.raw_source.shape[:2]
    print(f'image {H}x{W_}, load RSS +{r1 - r0:.0f} MB')

    def retained(d):
        b = d.raw_source.nbytes + d.raw_display.nbytes + d.source_float.nbytes
        b += sum(a.nbytes for a in d.whitened.values())
        b += sum(bd.display.nbytes for bd in d._bundles.values())
        return b / 1e6

    names = [c.name for c in color_interpreters.get_color_interpreters()]
    pnames = [p.name for p in processors.get_processors()]
    for p in pnames:
        for c in names:
            d.get_bundle(p, c)
            d.get_thumbnail(p, c)
    r2 = rss()
    print(f'after ALL {len(pnames)}x{len(names)} bundles+thumbnails: '
          f'RSS {r2:.0f} MB, retained {retained(d):.0f} MB, cache={len(d._bundles)}')

    case = np.zeros((H, W_), bool)
    notc = np.zeros((H, W_), bool)
    case[::40, 200:800:40] = True
    notc[::40, 1200:1800:40] = True
    peak = [r2]
    stop = [False]

    def watch():
        while not stop[0]:
            peak[0] = max(peak[0], rss())
            threading.Event().wait(0.02)

    t = threading.Thread(target=watch)
    t.start()
    out = era_main.run_classification(
        d, case, notc, 'LR',
        dict(blur=3, n_best=10, pca_pct=95, n_segments=100, compactness=10), W())
    stop[0] = True
    t.join()
    r4 = rss()
    print(f'classification LR: RSS {r2:.0f} -> {r4:.0f} MB, peak ~{peak[0]:.0f} MB')

    n_px = H * W_
    old = n_px * (36 * 6 + 3 + 3) / 1e6
    old_clf = n_px * (108 + 108 * 8) / 1e6
    print(f'OLD estimate same image: eager results ~{old:.0f} MB; '
          f'classification ~{old_clf:.0f} MB')
    n24 = 24e6
    print(f'@24MP: OLD ~{n24 * 36 * 6 / 1e9:.1f} GB results + '
          f'~{n24 * 108 * 9 / 1e9:.1f} GB clf | '
          f'NEW ~{(n24 * (6 + 8 + 4 * 12 + 6 * 3) + n24 * 30 * 4) / 1e9:.1f} GB')
