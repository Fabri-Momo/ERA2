import cv2
import numpy

from . import ColorInterpreter, clip_u8


class ColorInterpreter(ColorInterpreter):
    name = 'XYZ'
    order = 10

    def interpret(self, matrix):
        """False-colour RGB [0,1] -> native XYZ (float32, ~[0, 1.09])."""
        return cv2.cvtColor(numpy.asarray(matrix, dtype=numpy.float32), cv2.COLOR_RGB2XYZ)

    def to_display(self, native):
        # OpenCV 8-bit encoding of XYZ is a simple 0..255 rescale.
        return clip_u8(native * 255.0)
