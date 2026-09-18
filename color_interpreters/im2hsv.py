import cv2
import numpy

from . import ColorInterpreter, clip_u8


class ColorInterpreter(ColorInterpreter):
    name = 'HSV'
    order = 70

    def interpret(self, matrix):
        """False-colour RGB [0,1] -> native HSV (H [0,360], S/V [0,1])."""
        return cv2.cvtColor(numpy.asarray(matrix, dtype=numpy.float32), cv2.COLOR_RGB2HSV)

    def to_display(self, native):
        # OpenCV 8-bit encoding: H / 2, S * 255, V * 255.
        # Hue is circular: H ~ 360 must display as 0, never 180.
        out = native * numpy.array([0.5, 255.0, 255.0], dtype=numpy.float32)
        out[..., 0] = numpy.round(out[..., 0]) % 180
        return clip_u8(out)
