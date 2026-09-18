import os
import sys


# Prefer Qt6 (PyQt6) if not already configured.
os.environ.setdefault('QT_API', 'pyqt6')

# Qt6 does not ship fonts on Windows. Provide a fallback font directory
# (project/fonts or conda env fonts) before Qt is initialised.
if 'QT_QPA_FONTDIR' not in os.environ:
    _font_candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fonts'),
        os.path.join(sys.prefix, 'fonts'),
    ]
    for _font_dir in _font_candidates:
        if os.path.isdir(_font_dir):
            os.environ['QT_QPA_FONTDIR'] = _font_dir
            break

try:
    from qtpy import QtGui, QtWidgets, QtCore
    from qtpy.QtCore import Signal, Slot
    print(f'qtpy loaded ({os.environ.get("QT_API")})')
except Exception:
    # Fallback to native bindings if qtpy is unavailable.
    try:
        from PyQt6 import QtGui, QtWidgets, QtCore
        from PyQt6.QtCore import pyqtSignal as Signal, pyqtSlot as Slot
        print('PyQt6 loaded')
    except Exception:
        try:
            from PyQt5 import QtGui, QtWidgets, QtCore
            from PyQt5.QtCore import pyqtSignal as Signal, pyqtSlot as Slot
            print('PyQt5 loaded')
        except Exception:
            from PySide2 import QtGui, QtWidgets, QtCore
            from PySide2.QtCore import Signal, Slot
            print('pyside2 loaded')
