import cv2
import numpy

from . import ColorInterpreter, clip_u8


class ColorInterpreter(ColorInterpreter):
    name = 'LUV'
    order = 30

    def interpret(self, matrix):
        """False-colour RGB [0,1] -> native LUV (L in [0,100], u/v signed)."""
        return cv2.cvtColor(numpy.asarray(matrix, dtype=numpy.float32), cv2.COLOR_RGB2LUV)

    def to_display(self, native):
        # OpenCV 8-bit encoding: L * 255/100,
        # u -> (u + 134) * 255/354, v -> (v + 140) * 255/262.
        shifted = native + numpy.array([0.0, 134.0, 140.0],
                                       dtype=numpy.float32)
        scaled = shifted * numpy.array(
            [255.0 / 100.0, 255.0 / 354.0, 255.0 / 262.0],
            dtype=numpy.float32)
        return clip_u8(scaled)
