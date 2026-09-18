import numpy
from .utils import whiten, reshape_scale


class Processor:
    name = 'ZCA cor'
    order = 50

    def process(self, matrix, mask=None, keep=0):
        w, h, c = matrix.shape
        matrix = matrix.reshape(w * h, -1)
        if mask is None:
            subset = matrix
        else:
            mask = mask.reshape(w * h)
            subset = matrix[mask]
        whitened_matrix, subset = whiten(matrix, subset, method='zca_cor')

        return reshape_scale(whitened_matrix, subset, w, h, keep)
