"""Generate the WiX UI bitmaps (banner 493x58, dialog 493x312) from the icon.

    python installer/make_bitmaps.py

Run once when the icon or the theme colours change; the BMPs are versioned.
"""
import os

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ICON = os.path.join(ROOT, 'resources', 'images', 'icon.png')
BG = (0xFA, 0xF7, 0xF2)
ACCENT = (0xB5, 0x65, 0x1D)


def _paste_icon(canvas, size, x, y):
    icon = Image.open(ICON).convert('RGBA').resize((size, size), Image.LANCZOS)
    canvas.paste(icon, (x, y), icon)


def banner():
    im = Image.new('RGB', (493, 58), BG)
    # ochre rule at the bottom, icon at the right (WiX draws the title on the left)
    im.paste(ACCENT, (0, 56, 493, 58))
    _paste_icon(im, 44, 493 - 44 - 8, 6)
    im.save(os.path.join(HERE, 'banner.bmp'))


def dialog():
    im = Image.new('RGB', (493, 312), BG)
    # WiX puts the welcome text on the right (x >= 164); brand the left panel.
    im.paste(ACCENT, (0, 0, 164, 312))
    _paste_icon(im, 128, 18, 92)
    im.save(os.path.join(HERE, 'dialog.bmp'))


if __name__ == '__main__':
    banner()
    dialog()
    print('wrote installer/banner.bmp and installer/dialog.bmp')
