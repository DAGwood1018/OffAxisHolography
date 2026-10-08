"""
Gini index and Tamura coefficient as edge-sparsity focus metrics for
digital holographic autofocusing.

Background
----------
In holographic autofocusing, a stack of reconstructed complex wavefronts
U(x, y; z) is produced by numerically propagating a recorded hologram to a
set of candidate depths z. A scalar "sharpness" or "focus" metric is
computed at every z, and the depth that optimizes the metric is taken as
the in-focus plane. Because in-focus amplitude/phase images have sparse,
sharp edges (most pixels are near-uniform background, a few pixels carry
large gradients at edges), sparsity measures of the *edge map* (gradient
magnitude) of the reconstructed image make good focus metrics:

* Gini index (GI): a measure of statistical dispersion/sparsity used in
  economics (income inequality) and repurposed for sparsity-based focusing.
  GI = 0 for a perfectly flat ("democratic") distribution and GI -> 1 for a
  maximally sparse ("one pixel has everything") distribution. Under proper
  autofocus criteria, GI is *maximized* at the true focal plane (edges are
  concentrated in a few strong-gradient pixels).

* Tamura coefficient (TC): a normalized dispersion measure defined as the
  square root of the coefficient of variation, TC = sqrt(std(x) / mean(x)).
  It is also a sparsity indicator (larger when the distribution is more
  "peaky" relative to its mean) and is likewise maximized near focus.

Both metrics are applied to the *edge/gradient-magnitude image* of the
reconstructed wavefront (complex field, amplitude or phase) rather than to
the raw intensity, since it is the sparsity of edges -- not of raw pixel
values -- that is most diagnostic of focus.
"""

from __future__ import annotations

import numpy as np
from numba import njit


# --------------------------------------------------------------------------
# Core sparsity measures
# --------------------------------------------------------------------------

@njit(cache=True)
def gini_index(x: np.ndarray) -> float:
    """
    Gini index of sparsity (Hurley & Rickard normalized form):

        Let c[1] <= c[2] <= ... <= c[N] be the values of |x| sorted in
        ascending order, and let ||x||_1 = sum_k c[k]. Then

            GI(x) = 1 - 2 * sum_{k=1}^{N} ( c[k] / ||x||_1 ) * ( (N - k + 0.5) / N )

    GI = 0 for a perfectly uniform vector, GI = 1 - 1/N when all energy
    sits in a single entry.

    Parameters
    ----------
    x : float64 array (any shape; it is flattened). Values are taken in
        absolute value, so signed input is fine.

    Returns
    -------
    float in [0, 1). Returns 0.0 for an all-zero or empty input.
    """
    flat = x.ravel()
    n = flat.size
    if n == 0:
        return 0.0

    c = np.abs(flat).astype(np.float64)
    c.sort()  # ascending

    l1 = 0.0
    for i in range(n):
        l1 += c[i]
    if l1 == 0.0:
        return 0.0

    acc = 0.0
    for k in range(1, n + 1):
        acc += (c[k - 1] / l1) * ((n - k + 0.5) / n)

    return 1.0 - 2.0 * acc


@njit(cache=True)
def tamura_coefficient(x: np.ndarray) -> float:
    """
    Tamura coefficient:

        TC(x) = sqrt( std(|x|) / mean(|x|) )

    Larger TC indicates a "peakier" / sparser distribution relative to its
    mean.

    Parameters
    ----------
    x : float64 array (any shape; it is flattened).

    Returns
    -------
    float >= 0. Returns 0.0 if the mean is zero or the input is empty.
    """
    flat = x.ravel()
    n = flat.size
    if n == 0:
        return 0.0

    s = 0.0
    for i in range(n):
        s += abs(flat[i])
    mean = s / n
    if mean == 0.0:
        return 0.0

    var_acc = 0.0
    for i in range(n):
        d = abs(flat[i]) - mean
        var_acc += d * d
    std = np.sqrt(var_acc / n)

    return np.sqrt(std / mean)


# --------------------------------------------------------------------------
# Edge map (gradient magnitude) for "edge sparsity" versions of the metrics
# --------------------------------------------------------------------------
#
# gradient_magnitude() is dtype-generic: numba compiles a separate
# specialization per input type, and
#
#     out[y, x] = sqrt( |gx|^2 + |gy|^2 )
#
# is the correct Sobel-gradient magnitude whether gx, gy are real (img is
# an amplitude or phase image) or complex (img is the complex wavefront
# U = A * exp(i*phi)). For complex input the map is jointly sensitive to
# amplitude and phase transitions.

@njit(cache=True)
def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """
    Sobel-operator gradient magnitude of a 2-D image. Border pixels (where
    the 3x3 stencil doesn't fit) are set to 0.

    Parameters
    ----------
    img : 2-D float64 or complex128 array

    Returns
    -------
    2-D float64 array, same shape as img.
    """
    h, w = img.shape
    out = np.zeros((h, w), dtype=np.float64)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            gx = (
                img[y - 1, x + 1] + 2.0 * img[y, x + 1] + img[y + 1, x + 1]
                - img[y - 1, x - 1] - 2.0 * img[y, x - 1] - img[y + 1, x - 1]
            )
            gy = (
                img[y + 1, x - 1] + 2.0 * img[y + 1, x] + img[y + 1, x + 1]
                - img[y - 1, x - 1] - 2.0 * img[y - 1, x] - img[y - 1, x + 1]
            )
            out[y, x] = np.sqrt(abs(gx) ** 2 + abs(gy) ** 2)

    h, w = out.shape
    if h < 3 or w < 3:
        return np.zeros(0, dtype=np.float64)
    return out[1:h - 1, 1:w - 1].copy().ravel()


@njit(cache=True)
def gini_sparsity_metric(img):
    """Gini index of the Sobel edge map of a 2-D image (edge sparsity)."""
    return gini_index(gradient_magnitude(img))


@njit(cache=True)
def tamura_sparsity_metric(img):
    """Tamura coefficient of the Sobel edge map of a 2-D image (edge sparsity)."""
    return tamura_coefficient(gradient_magnitude(img))
