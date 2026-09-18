import numpy
from .utils import whiten, reshape_scale


class Processor:
    name = 'Neutral'

    def process(self, matrix, selection=None, keep=0):
        return matrix
