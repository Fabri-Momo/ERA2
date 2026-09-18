import importlib
import logging

import numpy

logger = logging.getLogger(__name__)

# Explicit module list: pkgutil.iter_modules is unreliable under PyInstaller.
_MODULES = ('im2XYZ', 'im2LAB', 'im2LUV', 'im2YCrCb', 'im2YUV',
            'im2hls', 'im2hsv', 'im2CMY')


def clip_u8(x):
    """Round to nearest and clip to [0, 255] -> uint8."""
    return numpy.clip(numpy.round(x), 0, 255).astype(numpy.uint8)


class ColorInterpreter:
    """Base colour-space interpreter.

    All colour transforms in ERA are *false-colour* transforms: their input is
    decorrelated pseudo-RGB data, not physical sRGB primaries, so the result is
    meant for visual separation rather than colorimetry.

    interpret() takes a float32 image in [0, 1] and returns the *native*
    (unquantised) values of the colour space, float32.

    to_display() encodes native values to uint8 using the same conventions
    OpenCV applies to 8-bit images, for display/export only.
    """
    name = 'RGB'
    order = 0

    def interpret(self, matrix):
        """Return the native (float32) representation of the colour space."""
        return numpy.ascontiguousarray(matrix, dtype=numpy.float32)

    def to_display(self, native):
        """Encode native colour data to a uint8 image for Qt."""
        return clip_u8(native * 255.0)


def get_color_interpreters():
    color_interpreters_list = []
    for name in _MODULES:
        full_name = f'{__name__}.{name}'
        try:
            module = importlib.import_module(full_name)
        except Exception:
            # A programming error must not silently remove a colour space.
            logger.exception(
                'Failed to import colour interpreter module %s', full_name)
            continue
        try:
            interpreter = module.ColorInterpreter
        except AttributeError:
            logger.debug('Module %s has no ColorInterpreter class', full_name)
            continue
        color_interpreters_list.append(interpreter)
    color_interpreters_list.append(ColorInterpreter)
    return sorted(color_interpreters_list, key=lambda x: x.order)
