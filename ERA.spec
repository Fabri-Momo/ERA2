# -*- mode: python ; coding: utf-8 -*-
import os
import sys

from PyInstaller.utils.hooks import collect_submodules

datas = [('doc', 'doc'), ('resources', 'resources')]

# Local plugin packages: collect_submodules cannot see them from the spec
# (the project dir is not on sys.path of the isolated subprocess), so the
# module list is kept explicit, mirroring processors._MODULES /
# color_interpreters._MODULES.
hiddenimports = (
    ['processors.zca', 'processors.pca', 'processors.cholesky',
     'processors.fastICA', 'processors.utils',
     'color_interpreters.im2XYZ', 'color_interpreters.im2LAB',
     'color_interpreters.im2LUV', 'color_interpreters.im2YCrCb',
     'color_interpreters.im2YUV', 'color_interpreters.im2hls',
     'color_interpreters.im2hsv', 'color_interpreters.im2CMY']
    + collect_submodules('sklearn')
    + ['skimage.segmentation', 'cleanlab', 'qimage2ndarray']
)

if sys.platform == 'darwin' and os.path.exists('icone_ERA.icns'):
    icon = 'icone_ERA.icns'
elif sys.platform == 'win32' and os.path.exists('icone_ERA.ico'):
    icon = 'icone_ERA.ico'
else:
    icon = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'PySide2', 'PySide6', 'tkinter', 'matplotlib',
              'IPython', 'jupyter'],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ERA',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ERA',
)

if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='ERA.app',
        icon=icon,
        bundle_identifier='fr.era.extractionofrockart',
        info_plist={
            'NSHighResolutionCapable': True,
            'NSPrincipalClass': 'NSApplication',
        },
    )
