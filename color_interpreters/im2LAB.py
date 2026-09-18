import cv2
import numpy

from . import ColorInterpreter, clip_u8


class ColorInterpreter(ColorInterpreter):
    name = 'LAB'
    order = 20

    def interpret(self, matrix):
        """False-colour RGB [0,1] -> native LAB (L in [0,100], a/b signed)."""
        return cv2.cvtColor(numpy.asarray(matrix, dtype=numpy.float32), cv2.COLOR_RGB2LAB)

    def to_display(self, native):
        # OpenCV 8-bit encoding: L * 255/100, a + 128, b + 128.
        scaled = native * numpy.array([255.0 / 100.0, 1.0, 1.0],
                                      dtype=numpy.float32)
        scaled = scaled + numpy.array([0.0, 128.0, 128.0],
                                      dtype=numpy.float32)
        return clip_u8(scaled)
