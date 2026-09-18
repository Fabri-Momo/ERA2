"""SVG export: the figure — not the image frame — must be vectorised."""
import re

import cv2
import numpy as np

from main import save_contours_as_svg


def _parse_paths(svg_text):
    return re.findall(r'<path d="([^"]+)"', svg_text)


def _bbox(d):
    coords = [float(v) for v in re.findall(r'[-\d.]+', d)]
    xs, ys = coords[0::2], coords[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def test_figure_not_frame(tmp_path):
    mask = np.full((100, 100), 255, np.uint8)
    cv2.rectangle(mask, (30, 30), (60, 60), 0, -1)  # black square = figure
    svg = tmp_path / 'out.svg'
    save_contours_as_svg(mask, str(svg), 100, 100)
    paths = _parse_paths(svg.read_text())
    assert paths, 'no contour exported'
    boxes = [_bbox(p) for p in paths]
    # No contour may cover the whole image (that would be the background).
    for x0, y0, x1, y1 in boxes:
        assert not (x0 <= 1 and y0 <= 1 and x1 >= 99 and y1 >= 99), \
            f'background/frame contour exported: {(x0,y0,x1,y1)}'
    # The figure (square around 30..60) must be present.
    assert any(abs(x0 - 30) <= 2 and abs(y0 - 30) <= 2
               and abs(x1 - 60) <= 2 and abs(y1 - 60) <= 2
               for x0, y0, x1, y1 in boxes)


def test_figure_with_hole(tmp_path):
    mask = np.full((100, 100), 255, np.uint8)
    cv2.rectangle(mask, (20, 20), (80, 80), 0, -1)
    cv2.rectangle(mask, (40, 40), (60, 60), 255, -1)  # inner hole
    svg = tmp_path / 'hole.svg'
    save_contours_as_svg(mask, str(svg), 100, 100)
    paths = _parse_paths(svg.read_text())
    # outer + inner contour expected (RETR_TREE keeps holes)
    assert len(paths) >= 2


def test_old_behaviour_was_wrong():
    """Document the bug: raw mask made OpenCV outline the background."""
    mask = np.full((100, 100), 255, np.uint8)
    cv2.rectangle(mask, (30, 30), (60, 60), 0, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                 cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(c) for c in contours]
    assert any(w == 100 and h == 100 for x, y, w, h in rects)
