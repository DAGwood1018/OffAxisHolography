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
reconstructed amplitude (or phase) rather than to the raw intensity, since
it is the sparsity of edges -- not of raw pixel values -- that is most
diagnostic of focus.

This module provides:
    gini_index(x)              -- Gini index of a 1-D array of nonnegative values
    tamura_coefficient(x)      -- Tamura coefficient of a 1-D array
    gradient_magnitude(img)    -- simple Sobel-based edge map of a 2-D image
    edge_sparsity_gini(img)    -- GI of the edge map of a 2-D image
    edge_sparsity_tamura(img)  -- TC of the edge map of a 2-D image
    autofocus_scan(stack, metric) -- apply a metric slice-by-slice to a z-stack

All numerically heavy routines are compiled with numba's @njit for speed,
which matters because autofocus routines evaluate the metric over dozens to
hundreds of propagated planes.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


# --------------------------------------------------------------------------
# Core sparsity measures
# --------------------------------------------------------------------------

@njit(cache=True)
def gini_index(x: np.ndarray) -> float:
    """
    Gini index of sparsity for a 1-D array of nonnegative values.

    Standard closed-form estimator (Hurley & Rickard's normalized form,
    the one used in the sparsity/focus literature):

        Let c[1] <= c[2] <= ... <= c[N] be the values of |x| sorted in
        ascending order, and let ||x||_1 = sum_k c[k]. Then

            GI(x) = 1 - 2 * sum_{k=1}^{N} ( c[k] / ||x||_1 ) * ( (N - k + 0.5) / N )

    GI = 0 for a perfectly uniform vector, GI -> 1 as the vector becomes
    maximally sparse (all energy concentrated in one entry).

    Parameters
    ----------
    x : 1-D float64 array
        Values (e.g. flattened edge/gradient-magnitude image). Values are
        taken in absolute value, so signed input is fine.

    Returns
    -------
    float
        Gini index in [0, 1]. Returns 0.0 for an all-zero or empty input.
    """
    n = x.size
    if n == 0:
        return 0.0

    c = np.empty(n, dtype=np.float64)
    for i in range(n):
        v = x.flat[i]
        c[i] = v if v >= 0.0 else -v
    c.sort()  # ascending

    l1 = 0.0
    for i in range(n):
        l1 += c[i]
    if l1 == 0.0:
        return 0.0

    acc = 0.0
    for k in range(1, n + 1):
        weight = (n - k + 0.5) / n
        acc += (c[k - 1] / l1) * weight

    gi = 1.0 - 2.0 * acc
    return gi


@njit(cache=True)
def tamura_coefficient(x: np.ndarray) -> float:
    """
    Tamura coefficient of a 1-D array of nonnegative values.

        TC(x) = sqrt( std(x) / mean(x) )

    Larger TC indicates a "peakier" / sparser distribution relative to its
    mean. Values are taken in absolute value so signed input is fine.

    Parameters
    ----------
    x : 1-D float64 array

    Returns
    -------
    float
        Tamura coefficient (>= 0). Returns 0.0 if the mean is zero or the
        input is empty.
    """
    n = x.size
    if n == 0:
        return 0.0

    s = 0.0
    for i in range(n):
        v = x.flat[i]
        s += v if v >= 0.0 else -v
    mean = s / n
    if mean == 0.0:
        return 0.0

    var_acc = 0.0
    for i in range(n):
        v = x.flat[i]
        v = v if v >= 0.0 else -v
        d = v - mean
        var_acc += d * d
    std = np.sqrt(var_acc / n)

    ratio = std / mean
    if ratio < 0.0:
        ratio = 0.0
    return np.sqrt(ratio)


# --------------------------------------------------------------------------
# Edge map (gradient magnitude) for "edge sparsity" versions of the metrics
# --------------------------------------------------------------------------
#
# gradient_magnitude() is written to be dtype-generic: numba compiles a
# fresh specialization per input type, and the formula below,
#
#     out[y, x] = sqrt( |gx|^2 + |gy|^2 )
#
# is the correct Sobel-gradient magnitude whether gx, gy come out real
# (img is a real float64 amplitude or phase image) *or complex* (img is
# the complex wavefront U = A * exp(i*phi) itself). For real input,
# |gx|^2 == gx*gx, so this exactly reduces to the plain real Sobel
# magnitude. For complex input, |gx|^2 is the squared modulus of the
# complex Sobel response, so the two orthogonal complex gradients combine
# into a single real, physically meaningful edge-strength map that is
# jointly sensitive to amplitude *and* phase transitions -- which is what
# "edge sparsity of the complex optical wavefront" requires. (An earlier
# version of this function used gx*gx + gy*gy directly, which is wrong for
# complex input: that computes the square of a complex number rather than
# a squared magnitude, and would silently return nonsense.)

@njit(cache=True)
def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """
    Sobel-operator gradient magnitude of a 2-D image, used as the "edge
    map" whose sparsity is measured by GI / TC. Border pixels (where the
    3x3 stencil doesn't fully fit) are set to 0.

    Accepts either:
      * a real float64 array (e.g. amplitude |U| or unwrapped phase), or
      * a complex128 array (the complex wavefront U itself) -- in which
        case the returned map reflects joint amplitude+phase edges.

    Parameters
    ----------
    img : 2-D float64 or complex128 array

    Returns
    -------
    2-D float64 array, same shape as img (always real-valued).
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

    return out


@njit(cache=True)
def gini_sparsity_metric(img):
    """Gini index of the Sobel edge map of a 2-D image (edge sparsity)."""
    edges = gradient_magnitude(img)
    flat = edges.reshape(edges.size)
    return gini_index(flat)


@njit(cache=True)
def tamura_sparsity_metric(img):
    """Tamura coefficient of the Sobel edge map of a 2-D image (edge sparsity)."""
    edges = gradient_magnitude(img)
    flat = edges.reshape(edges.size)
    return tamura_coefficient(flat)
