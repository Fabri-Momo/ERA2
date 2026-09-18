"""Ochre 'rock-art' visual theme for the whole application."""

from qt import QtGui

ACCENT = '#B5651D'
ACCENT_DARK = '#8E4B12'
ACCENT_LIGHT = '#F3E3D3'
BG = '#FAF7F2'
PANEL = '#FFFFFF'
BORDER = '#D9CFC2'
TEXT = '#2B2B2B'
MUTED = '#7A6F63'

STYLESHEET = f"""
QPushButton {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 14px;
    min-height: 26px;
}}
QPushButton:hover {{
    background: {ACCENT_LIGHT};
    border: 1px solid {ACCENT};
}}
QPushButton:pressed {{ background: #E8D2BC; }}
QPushButton:disabled {{ color: {MUTED}; background: #F1EDE7; }}

QPushButton[primary="true"] {{
    background: {ACCENT};
    color: white;
    border: 1px solid {ACCENT_DARK};
    font-weight: 600;
}}
QPushButton[primary="true"]:hover {{ background: #C9752A; }}
QPushButton[primary="true"]:pressed {{ background: {ACCENT_DARK}; }}
QPushButton[primary="true"]:disabled {{ background: #D8B9A0; }}

QRadioButton, QCheckBox {{ spacing: 6px; }}
QRadioButton::indicator {{
    width: 16px; height: 16px;
    border-radius: 8px;
    border: 1px solid {BORDER};
    background: {PANEL};
}}
QRadioButton::indicator:checked {{
    background: {ACCENT};
    border: 1px solid {ACCENT_DARK};
}}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border-radius: 4px;
    border: 1px solid {BORDER};
    background: {PANEL};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border: 1px solid {ACCENT_DARK};
}}
QRadioButton::indicator:hover, QCheckBox::indicator:hover {{
    border-color: {ACCENT};
}}

QSpinBox {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 2px 6px;
    background: {PANEL};
    min-height: 24px;
}}
QSpinBox:focus {{ border-color: {ACCENT}; }}
QSpinBox::up-button, QSpinBox::down-button {{
    subcontrol-origin: border;
    width: 18px;
    border-left: 1px solid {BORDER};
    background: #F1EDE7;
}}
QSpinBox::up-button {{
    subcontrol-position: top right;
    border-top-right-radius: 4px;
}}
QSpinBox::down-button {{
    subcontrol-position: bottom right;
    border-bottom-right-radius: 4px;
}}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {{ background: {ACCENT_LIGHT}; }}
QSpinBox::up-arrow {{ image: url(:/arrow-up.svg); width: 10px; height: 6px; }}
QSpinBox::down-arrow {{ image: url(:/arrow-down.svg); width: 10px; height: 6px; }}
QSpinBox::up-arrow:disabled, QSpinBox::up-arrow:off,
QSpinBox::down-arrow:disabled, QSpinBox::down-arrow:off {{ image: none; }}

QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    top: -1px;
}}
QTabBar::tab {{
    padding: 8px 16px;
    background: #F1EDE7;
    border: 1px solid {BORDER};
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background: {PANEL};
    color: {ACCENT_DARK};
    font-weight: 600;
    border-bottom: 2px solid {ACCENT};
}}
QTabBar::tab:hover:!selected {{ background: {ACCENT_LIGHT}; }}
QTabBar::tab:left {{
    padding: 16px 8px;
    border-top-left-radius: 6px;
    border-bottom-left-radius: 6px;
    border-top-right-radius: 0px;
    border-bottom: none;
    border-right: none;
    margin-right: 0px;
    margin-bottom: 2px;
}}

QProgressBar {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    background: {PANEL};
    text-align: center;
    height: 14px;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}

QToolBar {{
    background: {BG};
    border-bottom: 1px solid {BORDER};
    spacing: 6px;
    padding: 4px;
}}
QToolButton {{ border-radius: 6px; padding: 4px; }}
QToolButton:hover {{ background: {ACCENT_LIGHT}; }}

QStatusBar {{ color: {MUTED}; }}
QMenuBar::item:selected {{ background: {ACCENT_LIGHT}; }}
QMenu::item:selected {{ background: {ACCENT}; color: white; }}

QScrollArea, QGraphicsView {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    background: {PANEL};
}}

QSplitter::handle {{ background: {BORDER}; width: 2px; }}
"""


def apply_theme(app):
    """Apply the ERA theme: Fusion style + palette + global stylesheet."""
    app.setStyle('Fusion')
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(BG))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(PANEL))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(PANEL))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(ACCENT))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor('white'))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(TEXT))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(TEXT))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(TEXT))
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)
