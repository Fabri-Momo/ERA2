from qt import QtWidgets, QtGui, QtCore
from theme import ACCENT, ACCENT_DARK, BORDER, PANEL, TEXT

# Write select.png / include.png / exclude.png to the current directory while
# drawing.  Debugging aid only; disabled in normal use.
DEBUG_EXPORT_MASKS = False


class Help(QtWidgets.QDialog):

    def __init__(self, page, parent=None):
        super(Help, self).__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setAttribute(QtCore.Qt.WA_GroupLeader)

        self.pageLabel = QtWidgets.QLabel()

        self.textBrowser = QtWidgets.QTextBrowser()

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.textBrowser)
        self.setLayout(layout)

        self.textBrowser.setSearchPaths([":/"])
        self.textBrowser.setSource(QtCore.QUrl(page))
        self.resize(400, 600)
        self.setWindowTitle(self.tr("Help"))

class ColorVariant(QtWidgets.QAbstractButton):
    def __init__(self, label, thumbnail, parent=None):
        super(ColorVariant, self).__init__(parent=parent)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding
        )
        self.setText(label)
        self.image = thumbnail
        self.thumbnail_proportion = self.image.size().height()/self.image.size().width()
        self.setCheckable(True)
        self.setChecked(False)
        self._hover = False
        self.setMouseTracking(True)

    def sizeHint(self):
        return QtCore.QSize(150, 150)

    def enterEvent(self, event):
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        checked = self.isChecked()
        painter = QtGui.QPainter(self)
        painter.setRenderHints(QtGui.QPainter.Antialiasing |
                             QtGui.QPainter.SmoothPixmapTransform)

        # outer card
        card = QtCore.QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(PANEL))
        painter.drawRoundedRect(card, 8, 8)
        if checked:
            border_pen = QtGui.QPen(QtGui.QColor(ACCENT), 2)
        elif self._hover:
            border_pen = QtGui.QPen(QtGui.QColor('#C9752A'), 1)
        else:
            border_pen = QtGui.QPen(QtGui.QColor(BORDER), 1)
        painter.setPen(border_pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRoundedRect(card, 8, 8)

        # label band (22 px at top, left-aligned, elided)
        label_height = 22
        label_rect = QtCore.QRectF(card).adjusted(10, 0, -4, 0)
        label_rect.setHeight(label_height)
        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        metrics = QtGui.QFontMetrics(font)
        text = metrics.elidedText(self.text(), QtCore.Qt.ElideRight,
                                  int(label_rect.width()))
        painter.setPen(QtGui.QColor(ACCENT_DARK if checked else TEXT))
        painter.drawText(label_rect,
                         QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, text)

        # thumbnail box below the label band, 8 px margins, aspect preserved
        box = card.adjusted(8, label_height + 8, -8, -8)
        if self.thumbnail_proportion > box.height()/box.width():
            thumbnail = self.image.scaledToHeight(
                int(box.height()), QtCore.Qt.SmoothTransformation)
        else:
            thumbnail = self.image.scaledToWidth(
                int(box.width()), QtCore.Qt.SmoothTransformation)
        target = QtCore.QRectF(
            box.x() + (box.width() - thumbnail.width()) / 2,
            box.y() + (box.height() - thumbnail.height()) / 2,
            thumbnail.width(), thumbnail.height())
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(target, 4, 4)
        painter.setClipPath(clip)
        painter.drawImage(target, thumbnail)
        painter.setClipping(False)

        # checked badge: ochre circle + white check, top-right corner
        if checked:
            badge = QtCore.QRectF(card.right() - 22, card.top() + 6, 16, 16)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(ACCENT))
            painter.drawEllipse(badge)
            check_pen = QtGui.QPen(QtCore.Qt.white, 2)
            check_pen.setCapStyle(QtCore.Qt.RoundCap)
            check_pen.setJoinStyle(QtCore.Qt.RoundJoin)
            painter.setPen(check_pen)
            painter.drawPolyline(QtGui.QPolygonF([
                QtCore.QPointF(badge.left() + 0.25 * badge.width(),
                               badge.top() + 0.50 * badge.height()),
                QtCore.QPointF(badge.left() + 0.45 * badge.width(),
                               badge.top() + 0.70 * badge.height()),
                QtCore.QPointF(badge.left() + 0.75 * badge.width(),
                               badge.top() + 0.30 * badge.height()),
            ]))


class ImageWidget(QtWidgets.QGraphicsView):
    def __init__(self, parent=None, bounding_box=None):
        super(ImageWidget, self).__init__(parent=parent)
        self._zoom = 0
        self.bounding_box = bounding_box or QtCore.QRectF(0., 0., 200., 200.)
        self.prop = 1
        self.center = QtCore.QPointF(100., 100.)
        self.current_pen_size = 10

    def set_pen_size(self, size):
        self.current_pen_size = size

    def update_prop(self):
        self.prop = self.bounding_box.width()/self.bounding_box.height()

    def resizeEvent(self, event):
        if self._zoom == 0:
            self.init_view()
        else:
            self.update_center(event)
        return super(ImageWidget, self).resizeEvent(event)

    def update_center(self, event):
        delta_x = event.size().width() - event.oldSize().width()
        delta_y = event.size().height() - event.oldSize().height()
        #scale dependant factor
        zoomInFactor = 1.25
        m11 = self.transform().m11()
        scale_factor = 1/(zoomInFactor * m11)
        self.translate(delta_x * scale_factor, delta_y * scale_factor)

    def init_view(self, size=None):
        self.fitInView(self.bounding_box, QtCore.Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        # Zoom Factor
        zoomInFactor = 1.25
        zoomOutFactor = 1 / zoomInFactor
        m11 = self.transform().m11()

        # Set Anchors
        self.setTransformationAnchor(QtWidgets.QGraphicsView.NoAnchor)
        self.setResizeAnchor(QtWidgets.QGraphicsView.NoAnchor)

        # Save the scene pos
        oldPos = self.mapToScene(event.pos())

        # Zoom
        if event.angleDelta().y() > 0:
            zoomFactor = zoomInFactor * m11
            self._zoom += 1
        else:
            zoomFactor = zoomOutFactor * m11
            self._zoom -= 1
        if self._zoom > 0:
            self.setTransform(self.transform().fromScale(zoomFactor, zoomFactor))
        elif self._zoom == 0:
            self.init_view()
        else:
            self._zoom = 0

        # Get the new position
        newPos = self.mapToScene(event.pos())
        self.center = newPos

        # Move scene to old position
        delta = newPos - oldPos
        self.translate(delta.x(), delta.y())

    def history_event(self, event):
        if event == 'redo':
            self.scene().redo()
        elif event == 'undo':
            self.scene().undo()

    def keyPressEvent(self, event):
        if self.scene() is None:
            return super().keyPressEvent(event)
        if event.key() == QtCore.Qt.Key_R:
            self.scene().redo()
        elif event.key() == QtCore.Qt.Key_U:
            self.scene().undo()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if self.scene() is None:
            return
        current_point = self.mapToScene(event.pos())
        if event.button() == QtCore.Qt.LeftButton:
            self.scene().start_stroke(current_point)

    def mouseReleaseEvent(self, event):
        if self.scene() is None:
            return
        current_point = self.mapToScene(event.pos())
        if event.button() == QtCore.Qt.LeftButton:
            self.scene().end_stroke(current_point)

    def mouseMoveEvent(self, event):
        if self.scene() is None:
            return
        current_point = self.mapToScene(event.pos())
        self.scene().draw_stroke(current_point)


class CustomScene(QtWidgets.QGraphicsScene):
    def __init__(self, parent=None):
        super(CustomScene, self).__init__(parent=parent)
        self.background_pixmap = None
        self.view = None
        self.drawing = False
        self.current_stroke = None
        self.last_point = None
        self.history = {'include': {'all': [], 'length': 0},
                        'exclude': {'all': [], 'length': 0},
                        'select': {'all': [], 'length': 0},
                        }
        self.masked = QtWidgets.QGraphicsItemGroup()
        self.addItem(self.masked)
        self.masked.setVisible(False)
        self.layers = {'select': QtWidgets.QGraphicsItemGroup(),
                       'include': QtWidgets.QGraphicsItemGroup(),
                       'exclude': QtWidgets.QGraphicsItemGroup(),
                       }
        self.pens = {'include': QtGui.QPen(QtCore.Qt.black, 10, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin),
                     'exclude': QtGui.QPen(QtCore.Qt.red, 10, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin),
                     'select': QtGui.QPen(QtCore.Qt.yellow, 10, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin),
                     }
        self.current_pen_mode = 'select'
        self.active_pen = None

    def set_pen_mode(self, mode):
        self.current_pen_mode = mode
        for l in self.layers.values():
            l.setVisible(False)
        if mode == 'select':
            self.layers['select'].setVisible(True)
        else:
            for l in ['include', 'exclude']:
                self.layers[l].setVisible(True)

    def undo(self):
        layer = self.current_pen_mode
        if self.history[layer]['all'] and self.history[layer]['length'] > 0:
            item = self.history[layer]['all'][self.history[layer]['length'] - 1]
            self.masked.addToGroup(item)
            self.history[layer]['length'] -= 1
            self.update()

    def redo(self):
        layer = self.current_pen_mode
        if self.history[layer]['all'] and self.history[layer]['length'] < len(self.history[layer]['all']):
            item = self.history[layer]['all'][self.history[layer]['length']]
            self.layers[layer].addToGroup(item)
            self.history[layer]['length'] += 1

    def flush_history(self, length=None, layer=None):
        if layer is None:
            layer = self.current_pen_mode
        if length is None:
            length = self.history[layer]['length']
        while len(self.history[layer]['all']) > length:
            item = self.history[layer]['all'].pop()
            if item in self.layers[layer].childItems():
                self.layers[layer].removeFromGroup(item)
            elif item in self.masked.childItems():
                self.masked.removeFromGroup(item)
            del item
        self.history[layer]['length'] = len(self.history[layer]['all'])

    def flush_all_history(self):
        for layer in self.layers:
            self.flush_history(length=0, layer=layer)

    def set_pen(self, pen, layer):
        self.pens[layer] = pen

    def increment_pen(self, layer):
        self.pens[layer].setWidth(self.pens[layer].width() + 1)

    def decrement_pen(self, layer):
        self.pens[layer].setWidth(self.pens[layer].width() - 1)

    def set_pen_width(self, layer, width):
        self.pens[layer].setWidth(width)

    def sanitized_pos(self, pos):
        if self.background_pixmap is None:
            return pos
        if not self.background_pixmap.contains(pos):
            if pos.x() < 0:
                pos.setX(0)
            elif pos.x() > self.background_pixmap.boundingRect().width():
                pos.setX(int(self.background_pixmap.boundingRect().width()))
            if pos.y() < 0:
                pos.setY(0)
            elif pos.y() > self.background_pixmap.boundingRect().height():
                pos.setY(int(self.background_pixmap.boundingRect().height()))
        return pos

    def start_stroke(self, pos):
        if self.background_pixmap is None:
            return
        self.drawing = True
        self.flush_history()
        self.current_stroke = QtWidgets.QGraphicsItemGroup()
        self.addItem(self.current_stroke)
        if self.last_point is None:
            self.last_point = self.sanitized_pos(pos)

    def draw_stroke(self, pos):
        if self.background_pixmap is None:
            return
        layer = self.current_pen_mode
        active_pen = self.pens[layer]
        if self.drawing and self.last_point:
            pos = self.sanitized_pos(pos)
            line = QtCore.QLineF(self.last_point, pos)
            if line.length() > 3:
                line_item = QtWidgets.QGraphicsLineItem()
                line_item.setPen(active_pen)
                line_item.setLine(line)
                self.current_stroke.addToGroup(line_item)
                self.last_point = pos

    def end_stroke(self, pos):
        if self.background_pixmap is None or self.current_stroke is None:
            return
        layer = self.current_pen_mode
        self.drawing = False
        self.history[layer]['all'].append(self.current_stroke)
        self.layers[layer].addToGroup(self.current_stroke)
        self.last_point = None
        self.current_stroke = None
        self.history[layer]['length'] = len(self.history[layer]['all'])

    def set_background_pixmap(self, pixmap):
        if not self.background_pixmap:
            self.background_pixmap = QtWidgets.QGraphicsPixmapItem()
            self.addItem(self.background_pixmap)
        self.view.bounding_box = QtCore.QRectF(0., 0., float(pixmap.size().width()), float(pixmap.size().height()))
        self.view.update_prop()
        self.background_pixmap.setPixmap(pixmap)
        for value in self.layers.values():
            if value.scene() is None:
                self.addItem(value)

    def render_layer(self, layer):
        image = QtGui.QImage(self.background_pixmap.boundingRect().toRect().size(), QtGui.QImage.Format_RGB32)
        image.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(image)
        self.background_pixmap.setVisible(False)
        visible_layers = [l for l in self.layers.values() if l.isVisible()]
        hidden_layers = [l for l in self.layers.values() if not l.isVisible()]
        for l in visible_layers:
            l.setVisible(False)
        self.layers[layer].setVisible(True)
        self.render(painter, self.background_pixmap.boundingRect(), self.background_pixmap.boundingRect())
        painter.end()
        self.background_pixmap.setVisible(True)
        for l in visible_layers:
            l.setVisible(True)
        for l in hidden_layers:
            l.setVisible(False)
        if DEBUG_EXPORT_MASKS:
            image.save(f'{layer}.png')
        return image



class ResultView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None, bounding_box=None):
        super(ResultView, self).__init__(parent=parent)
        self._zoom = 0
        self.bounding_box = bounding_box or QtCore.QRectF(0., 0., 200., 200.)
        self.prop = 1
        self.center = QtCore.QPointF(100., 100.)

    def update_prop(self):
        self.prop = self.bounding_box.width()/self.bounding_box.height()

    def resizeEvent(self, event):
        if self._zoom == 0:
            self.init_view()
        else:
            self.update_center(event)
        return super(ResultView, self).resizeEvent(event)

    def update_center(self, event):
        delta_x = event.size().width() - event.oldSize().width()
        delta_y = event.size().height() - event.oldSize().height()
        #scale dependant factor
        zoomInFactor = 1.25
        m11 = self.transform().m11()
        scale_factor = 1/(zoomInFactor * m11)
        self.translate(delta_x * scale_factor, delta_y * scale_factor)

    def init_view(self):
        self.fitInView(self.bounding_box, QtCore.Qt.KeepAspectRatio)

    def wheelEvent(self, event, propagate=True):
        # Zoom Factor
        zoomInFactor = 1.25
        zoomOutFactor = 1 / zoomInFactor
        m11 = self.transform().m11()

        # Set Anchors
        self.setTransformationAnchor(QtWidgets.QGraphicsView.NoAnchor)
        self.setResizeAnchor(QtWidgets.QGraphicsView.NoAnchor)

        # Save the scene pos
        oldPos = self.mapToScene(event.pos())

        # Zoom
        if event.angleDelta().y() > 0:
            zoomFactor = zoomInFactor * m11
            self._zoom += 1
        else:
            zoomFactor = zoomOutFactor * m11
            self._zoom -= 1
        if self._zoom > 0:
            self.setTransform(self.transform().fromScale(zoomFactor, zoomFactor))
        elif self._zoom == 0:
            self.init_view()
        else:
            self._zoom = 0

        # Get the new position
        newPos = self.mapToScene(event.pos())
        self.center = newPos

        # Move scene to old position
        delta = newPos - oldPos
        self.translate(delta.x(), delta.y())
        if propagate and self.parent():
            self.parent().propagate_zoom(event)


class ResultWidget(QtWidgets.QWidget):
    def __init__(self, label, parent=None):
        super(ResultWidget, self).__init__(parent=parent)
        layout = QtWidgets.QVBoxLayout()
        self.background_pixmap = None
        self.scene = QtWidgets.QGraphicsScene()
        self.view = ResultView()
        self.view.setScene(self.scene)
        self.checkbox = QtWidgets.QCheckBox(label)
        layout.addWidget(self.checkbox)
        layout.addWidget(self.view)
        self.setLayout(layout)

    def propagate_zoom(self, event):
        for child in self.parent().children():
            if not child is self and isinstance(child, ResultWidget):
                child.view.wheelEvent(event, propagate=False)

    def set_background_pixmap(self, pixmap):
        if not self.background_pixmap:
            self.background_pixmap = QtWidgets.QGraphicsPixmapItem()
            self.scene.addItem(self.background_pixmap)
        self.view.bounding_box = QtCore.QRectF(0., 0., float(pixmap.size().width()), float(pixmap.size().height()))
        self.view.update_prop()
        self.background_pixmap.setPixmap(pixmap)
        self.init_view()

    def is_selected_for_export(self):
        if self.checkbox.isChecked():
            return True
        return False

    def get_image(self):
        return self.background_pixmap.pixmap()

    def init_view(self):
        # Set Anchors
        self.view.setTransformationAnchor(QtWidgets.QGraphicsView.NoAnchor)
        self.view.setResizeAnchor(QtWidgets.QGraphicsView.NoAnchor)
        if self.view.size().width() == 0 or self.view.size().height() == 0:
            return
        delta_prop = self.view.size().width()/self.view.size().height() - self.view.prop
        if delta_prop >= 0:
            scale = (self.view.size().height()+5)/self.view.bounding_box.height()
        else:
            scale = (self.view.size().width()+5)/self.view.bounding_box.width()
        self.view.centerOn(self.view.bounding_box.width()/2, self.view.bounding_box.height())
        self.view.setTransform(self.view.transform().fromScale(scale, scale))

    def uncheck(self):
        self.checkbox.setChecked(False)
