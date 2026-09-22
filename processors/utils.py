import numpy as np

# Regularisation added to covariance eigenvalues before inversion.  Keeps
# whitening well-defined on nearly singular (low-variance) selections.
WHITENING_EPSILON = 1e-5

# Validation thresholds for the pixel subset used to estimate whitening
# statistics.
MIN_SELECTION_PIXELS = 32       # user-drawn selections
MIN_FULLIMAGE_PIXELS = 4        # whole-image statistics
MIN_SELECTION_UNIQUE = 4        # distinct colours required in a selection
MIN_FULLIMAGE_UNIQUE = 2

# FastICA is fitted on at most this many pixels (deterministic subsample);
# the resulting transform is then applied to the whole image.
MAX_ICA_TRAINING_PIXELS = 100_000
ICA_RANDOM_STATE = 0

# Neutral output for degenerate (constant / unusable) channels.
NEUTRAL_VALUE = 0.5

# Rows per chunk when applying a transform to a full image.
TRANSFORM_CHUNK = 250_000

SELECTION_ERROR_MSG = (
    'Selection is too small or has insufficient colour variation '
    'for decorrelation.'
)


class InsufficientSelectionError(ValueError):
    """A manual selection cannot support whitening statistics."""


def matrix_stats(matrix):
    w, h, c = matrix.shape
    stats = {}
    stats['min'] = np.min(matrix.T.reshape(c, h * w), axis=1)
    stats['max'] = np.max(matrix.T.reshape(c, h * w), axis=1)
    return stats


def neutral_image(n_rows, n_cols, n_channels=3):
    """Deterministic neutral result for degenerate inputs."""
    return np.full((n_rows, n_cols, n_channels), NEUTRAL_VALUE,
                   dtype=np.float32)


def _has_min_unique(subset, min_unique):
    """Exact test `number of distinct rows >= min_unique`, cheap in the common case."""
    n = subset.shape[0]
    probe = 4096
    if n > probe:
        # distinct rows in a strided sample <= distinct rows overall, so a
        # positive answer on the sample is already exact
        sample = subset[:: max(1, n // probe)]
        if np.unique(sample, axis=0).shape[0] >= min_unique:
            return True
    return np.unique(subset, axis=0).shape[0] >= min_unique


def check_subset(subset, min_pixels, min_unique, require_full_rank=False):
    """Return True if *subset* (N, C) can support whitening statistics."""
    if subset is None or subset.shape[0] < min_pixels:
        return False
    if not np.isfinite(subset).all():
        return False
    if not _has_min_unique(subset, min_unique):
        return False
    if require_full_rank:
        x = subset.astype(np.float64)
        x = x - x.mean(axis=0)
        cov = (x.T @ x) / x.shape[0]
        eig = np.linalg.eigvalsh(cov)
        if eig[-1] <= 0 or eig[0] <= eig[-1] * 1e-12:
            return False
    return True


def prepare_whitening_data(matrix, mask, valid_mask, require_full_rank=False):
    """Split an image into (all pixels, statistics subset).

    matrix:      (H, W, C) float image.
    mask:        optional boolean (H, W) user selection.
    valid_mask:  optional boolean (H, W); alpha>0 / finite pixels.
    require_full_rank: also require a full-rank covariance on the subset
        (needed by FastICA, which has no regularised fallback).

    Returns (X, subset, user_selection).  subset is None when the data is
    degenerate and no explicit selection was drawn (callers should return a
    neutral image).  Raises InsufficientSelectionError when a user selection
    exists but is unusable.
    """
    h, w, c = matrix.shape
    X = matrix.reshape(h * w, c)
    if valid_mask is None:
        stats_mask = np.ones(h * w, dtype=bool)
    else:
        stats_mask = valid_mask.reshape(h * w)
    if mask is not None:
        stats_mask = stats_mask & mask.reshape(h * w)
    subset = X[stats_mask]

    if mask is not None:
        ok = check_subset(
            subset,
            min_pixels=MIN_SELECTION_PIXELS,
            min_unique=MIN_SELECTION_UNIQUE,
            require_full_rank=require_full_rank,
        )
        if not ok:
            raise InsufficientSelectionError(SELECTION_ERROR_MSG)
    else:
        ok = check_subset(
            subset,
            min_pixels=MIN_FULLIMAGE_PIXELS,
            min_unique=MIN_FULLIMAGE_UNIQUE,
            require_full_rank=require_full_rank,
        )
        if not ok:
            return X, None, False
    return X, subset, mask is not None


def _centred_covariance(x):
    """Covariance of x (N, C) computed in float64 with chunked accumulation."""
    x64 = np.asarray(x, dtype=np.float64)
    mu = x64.mean(axis=0)
    n = x64.shape[0]
    acc = np.zeros((x64.shape[1], x64.shape[1]), dtype=np.float64)
    n_chunks = max(1, int(np.ceil(n / 1_000_000)))
    for chunk in np.array_split(x64, n_chunks):
        xc = chunk - mu
        acc += xc.T @ xc
    return acc / n, mu


def _fix_eigenvector_signs(U):
    """Deterministic sign convention for eigenvectors (columns of U).

    Eigen/SVD routines return eigenvectors with an arbitrary sign that can
    differ between BLAS implementations or machines, silently flipping a whole
    output channel.  We force the largest-magnitude component of every
    eigenvector to be positive.  This does not change the whitening result
    mathematically (the output channel is simply multiplied by -1), but makes
    the transform reproducible across platforms.
    """
    U = U.copy()
    for k in range(U.shape[1]):
        pivot = np.argmax(np.abs(U[:, k]))
        if U[pivot, k] < 0:
            U[:, k] = -U[:, k]
    return U


def estimate_whitening(x, method='zca', epsilon=WHITENING_EPSILON):
    """Estimate a whitening transform from pixel subset x (N, C).

    Returns (W, mu) such that (x - mu) @ W.T has approximately identity
    covariance.  method is one of 'zca', 'pca', 'cholesky'.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError('empty whitening subset')
    if not np.isfinite(x).all():
        raise ValueError('non-finite values in whitening subset')

    Sigma, mu = _centred_covariance(x)
    U, Lambda, _ = np.linalg.svd(Sigma)
    if not np.isfinite(Lambda).all():
        raise ValueError('non-finite covariance spectrum')

    if method == 'pca':
        U = _fix_eigenvector_signs(U)
        W = np.dot(np.diag(1.0 / np.sqrt(Lambda + epsilon)), U.T)
    elif method == 'zca':
        # ZCA is invariant to eigenvector signs (U D U.T): no sign fix needed.
        W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + epsilon)), U.T))
    elif method == 'cholesky':
        # Cholesky factor of the inverse covariance; sign-invariant as well.
        W = np.linalg.cholesky(
            np.dot(U, np.dot(np.diag(1.0 / (Lambda + epsilon)), U.T))).T
    else:
        raise ValueError(f'Whitening method not found: {method}')
    return W, mu


def apply_whitening(X, W, mu, dtype=np.float32):
    """Apply (X - mu) @ W.T to all pixels of X ((N, C) or (H, W, C))."""
    flat = X.reshape(-1, X.shape[-1])
    out = np.empty((flat.shape[0], W.shape[0]), dtype=dtype)
    for s in range(0, flat.shape[0], TRANSFORM_CHUNK):
        e = min(s + TRANSFORM_CHUNK, flat.shape[0])
        chunk = flat[s:e].astype(np.float64)
        out[s:e] = (chunk - mu) @ W.T
    return out


def whiten(X, x, method='zca'):
    """Whiten X using statistics estimated on subset x.

    X: data to transform ((N, C) or (H, W, C)).
    x: subset used to estimate mean/covariance (N2, C).
    Returns (a, b): transformed X and transformed subset, float32.
    """
    X = X.reshape((-1, np.prod(X.shape[1:])))
    x = np.asarray(x)
    x = x.reshape((-1, np.prod(x.shape[1:])))
    W, mu = estimate_whitening(x, method=method)
    a = apply_whitening(X, W, mu)
    b = apply_whitening(x, W, mu)
    return a, b


def contrast_stretch(a, subset, contrast_boost):
    """Per-channel clip to subset quantiles + linear stretch to [0, 1].

    a:      (N, C) transformed image data.
    subset: (M, C) transformed statistics subset (defines the quantiles).
    contrast_boost: ERA contrast boost value (0..20); the discarded tail
        fraction is contrast_boost / 200 on each side, i.e.
            boost 0  -> min / max
            boost 1  -> 0.5% / 99.5%
            boost 20 -> 10%  / 90%

    Degenerate channels (high == low, NaN, Inf) return NEUTRAL_VALUE.
    """
    tail = contrast_boost / 200.0
    a = np.asarray(a, dtype=np.float32)
    out = np.empty_like(a)
    for ch in range(a.shape[1]):
        low = np.quantile(subset[:, ch], tail)
        high = np.quantile(subset[:, ch], 1.0 - tail)
        denom = high - low
        col = np.clip(a[:, ch], low, high)
        if not np.isfinite(denom) or abs(denom) < 1e-12:
            out[:, ch] = NEUTRAL_VALUE
        else:
            out[:, ch] = (col - low) / denom
    return out


def reshape_scale(im_matrix, subset, n_rows, n_cols, contrast_boost):
    """Contrast-stretch transformed pixels and reshape to an image.

    Returns a float32 (n_rows, n_cols, 3) pseudo-RGB image in [0, 1].
    No uint8 quantisation happens here: conversion to 8 bits is only done
    later, for display/export.
    """
    out = contrast_stretch(im_matrix, subset, contrast_boost)
    return out.reshape((n_rows, n_cols, -1)).astype(np.float32)
