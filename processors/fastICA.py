import logging
import warnings

import numpy
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning

from .utils import (
    reshape_scale,
    neutral_image,
    prepare_whitening_data,
    MAX_ICA_TRAINING_PIXELS,
    ICA_RANDOM_STATE,
    SELECTION_ERROR_MSG,
    InsufficientSelectionError,
)

logger = logging.getLogger(__name__)


class Processor:
    name = 'FastICA'
    order = 40

    def process(self, matrix, mask=None, contrast_boost=0, valid_mask=None):
        h, w, c = matrix.shape
        X, subset, user_selection = prepare_whitening_data(
            matrix, mask, valid_mask, require_full_rank=True)
        if subset is None:
            return neutral_image(h, w, c)

        # Estimating 3 ICA components never requires millions of pixels.
        # Fit on a deterministic subsample, then apply to the whole image.
        fit_data = subset
        if subset.shape[0] > MAX_ICA_TRAINING_PIXELS:
            rng = numpy.random.default_rng(ICA_RANDOM_STATE)
            idx = rng.choice(subset.shape[0], MAX_ICA_TRAINING_PIXELS,
                             replace=False)
            fit_data = subset[idx]

        # Parameters fixed explicitly so that changes in scikit-learn defaults
        # cannot silently alter the method.  'arbitrary-variance' matches the
        # whitening behaviour of the historical scikit-learn versions ERA was
        # developed with; random_state makes runs reproducible.
        transformer = FastICA(
            n_components=min(3, c),
            algorithm='parallel',
            whiten='arbitrary-variance',
            fun='logcosh',
            random_state=ICA_RANDOM_STATE,
            max_iter=500,
            tol=1e-4,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                transformer.fit(fit_data.astype(numpy.float64))
            except Exception as exc:
                if user_selection:
                    raise InsufficientSelectionError(
                        SELECTION_ERROR_MSG) from exc
                logger.warning('FastICA failed on degenerate data: %s', exc)
                return neutral_image(h, w, c)
        for warning in caught:
            if issubclass(warning.category, ConvergenceWarning):
                logger.warning('FastICA did not fully converge: %s',
                               warning.message)

        # ICA components have arbitrary order and sign.  Canonicalise them so
        # the output channels are stable across BLAS implementations and
        # machines: order rows by their dominant mixing coefficient, and make
        # that coefficient positive.  The mathematical content is unchanged.
        comp = transformer.components_
        dominant = numpy.argmax(numpy.abs(comp), axis=1)
        comp = comp[numpy.argsort(dominant, kind='stable')]
        signs = numpy.sign(comp[numpy.arange(comp.shape[0]),
                                numpy.argmax(numpy.abs(comp), axis=1)])
        signs[signs == 0] = 1
        transformer.components_ = comp * signs[:, None]

        a = self._transform_image(transformer, X)
        b = transformer.transform(fit_data.astype(numpy.float64))
        if not (numpy.isfinite(a).all() and numpy.isfinite(b).all()):
            if user_selection:
                raise InsufficientSelectionError(SELECTION_ERROR_MSG)
            return neutral_image(h, w, c)
        return reshape_scale(a, b, h, w, contrast_boost)

    @staticmethod
    def _transform_image(transformer, X, chunk=250_000):
        """transform() on large images without a giant float64 temporary."""
        out = numpy.empty((X.shape[0], transformer.n_components),
                          dtype=numpy.float32)
        for s in range(0, X.shape[0], chunk):
            e = min(s + chunk, X.shape[0])
            out[s:e] = transformer.transform(
                X[s:e].astype(numpy.float64))
        return out
