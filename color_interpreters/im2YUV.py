import cv2
import numpy

from . import ColorInterpreter, clip_u8


class ColorInterpreter(ColorInterpreter):
    name = 'YUV'
    order = 50

    def interpret(self, matrix):
        """False-colour RGB [0,1] -> native YUV (Y,U,V in ~[0,1], .5-centred)."""
        return cv2.cvtColor(numpy.asarray(matrix, dtype=numpy.float32), cv2.COLOR_RGB2YUV)

    def to_display(self, native):
        return clip_u8(native * 255.0)
