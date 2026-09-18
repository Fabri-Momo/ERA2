import numpy

from . import ColorInterpreter, clip_u8


class ColorInterpreter(ColorInterpreter):
    """Subtractive CMY false-colour transform.

    Historically named 'CMY(K)' although the K channel was discarded; the
    transform is applied to decorrelated pseudo-RGB data, not to physical
    printer primaries.
    """
    name = 'CMY'
    order = 80

    def interpret(self, matrix):
        """CMY on [0,1] data: C = 1-R, M = 1-G, Y = 1-B."""
        return numpy.ascontiguousarray(1.0 - matrix, dtype=numpy.float32)

    def to_display(self, native):
        return clip_u8(native * 255.0)
