import cv2
import numpy

from . import ColorInterpreter, clip_u8


class ColorInterpreter(ColorInterpreter):
    name = 'YCrCb'
    order = 40

    def interpret(self, matrix):
        """False-colour RGB [0,1] -> native YCrCb ([0,1], Cr/Cb centred .5)."""
        return cv2.cvtColor(numpy.asarray(matrix, dtype=numpy.float32), cv2.COLOR_RGB2YCrCb)

    def to_display(self, native):
        return clip_u8(native * 255.0)
