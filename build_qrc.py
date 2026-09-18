#!/usr/bin/env python
"""Regenerate qrc_resources.py from resources.qrc and patch the import for the qt shim."""
import os
import re
import shutil
import subprocess
import sys


def find_rcc():
    rcc = shutil.which('rcc')
    if rcc:
        return rcc
    # Conda layout
    candidates = [
        os.path.join(sys.prefix, 'Library', 'bin', 'rcc.exe'),
        os.path.join(sys.prefix, 'bin', 'rcc'),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    qrc = os.path.join(root, 'resources.qrc')
    out = os.path.join(root, 'qrc_resources.py')
    rcc = find_rcc()
    if not rcc:
        raise RuntimeError('rcc not found. Install a Qt package that provides rcc.')

    subprocess.run([rcc, '-g', 'python', qrc, '-o', out], check=True)

    # Patch the binding import to use the project qt shim, so the generated
    # file works with PyQt6, PyQt5 or PySide2 transparently.
    with open(out, 'r', encoding='utf-8') as f:
        text = f.read()
    text = re.sub(
        r'^\s*from\s+(?:PyQt[56]|PySide[26])\s+import\s+QtCore\s*$',
        'from qt import QtCore',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    with open(out, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'Generated {out} using {rcc}')


if __name__ == '__main__':
    main()
