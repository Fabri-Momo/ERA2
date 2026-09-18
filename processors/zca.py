import numpy
from .utils import whiten, reshape_scale, neutral_image, prepare_whitening_data


class Processor:
    name = 'ZCA'
    order = 10

    def process(self, matrix, mask=None, contrast_boost=0, valid_mask=None):
        h, w, c = matrix.shape
        X, subset, _ = prepare_whitening_data(matrix, mask, valid_mask)
        if subset is None:
            return neutral_image(h, w, c)
        whitened_matrix, subset = whiten(X, subset, method='zca')
        return reshape_scale(whitened_matrix, subset, h, w, contrast_boost)
