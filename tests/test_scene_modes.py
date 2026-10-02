"""CustomScene pen-mode None: layers hidden, history and strokes kept."""
import numpy as np
import qimage2ndarray

from qt import QtCore, QtGui, QtWidgets
from widgets import CustomScene


class _ViewStub:
    """Minimal stand-in for ImageWidget (scene.view)."""
    bounding_box = None

    def update_prop(self):
        pass


def _scene():
    scene = CustomScene()
    scene.view = _ViewStub()
    scene.set_background_pixmap(QtGui.QPixmap(100, 100))
    return scene


def _select_stroke(scene):
    scene.set_pen_mode('select')
    scene.start_stroke(QtCore.QPointF(10, 10))
    scene.draw_stroke(QtCore.QPointF(50, 50))
    scene.end_stroke(QtCore.QPointF(50, 50))


def test_pen_mode_none_hides_layer_keeps_history():
    scene = _scene()
    _select_stroke(scene)
    assert scene.layers['select'].isVisible()
    assert len(scene.history['select']['all']) == 1

    scene.set_pen_mode(None)
    assert scene.current_pen_mode is None
    assert all(not l.isVisible() for l in scene.layers.values())
    assert len(scene.history['select']['all']) == 1  # history kept

    scene.set_pen_mode('select')
    assert scene.layers['select'].isVisible()
    img = scene.render_layer('select')
    arr = qimage2ndarray.byte_view(img)
    assert (arr[..., :3] != 255).any()  # stroke still rendered


def test_pen_mode_none_strokes_are_noops():
    scene = _scene()
    scene.set_pen_mode(None)
    scene.start_stroke(QtCore.QPointF(10, 10))
    scene.draw_stroke(QtCore.QPointF(50, 50))
    scene.end_stroke(QtCore.QPointF(50, 50))
    assert not scene.drawing
    assert scene.current_stroke is None
    assert all(len(h['all']) == 0 for h in scene.history.values())
    # undo/redo/flush must not blow up either
    scene.undo()
    scene.redo()
    scene.flush_history()
