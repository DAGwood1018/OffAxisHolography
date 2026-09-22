import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter

@njit(inline='always')
def wrap_to_pi(x):
    """
    Wrap phase to (-pi, pi].
    """
    return (x + np.pi) % (2 * np.pi) - np.pi

def calc_residues(wrapped_phase):
    """
    Compute phase residues using the standard 2×2 circulation.

    Parameters
    ----------
    wrapped_phase : (M, N) ndarray
        Wrapped phase image in radians.

    Returns
    -------
    residues : (M-1, N-1) ndarray of int8
        Values are {-1, 0, +1}.
    """

    p00 = wrapped_phase[:-1, :-1]
    p01 = wrapped_phase[:-1, 1:]
    p11 = wrapped_phase[1:, 1:]
    p10 = wrapped_phase[1:, :-1]

    d1 = wrap_to_pi(p01 - p00)
    d2 = wrap_to_pi(p11 - p01)
    d3 = wrap_to_pi(p10 - p11)
    d4 = wrap_to_pi(p00 - p10)

    residues = np.rint((d1 + d2 + d3 + d4) / (2 * np.pi)).astype(np.int8)
    return residues


def residue_density(residues, sigma=3.0):
    """
    Build a smooth per-pixel "residue density" map used to order the
    flood-fill integration: pixels far from any residue (clean, reliable
    interior) get a low value, pixels near residues (close to branch
    cuts / decorrelated regions) get a high value.

    Each residue at dual-grid position (i, j) contributes |residue| to the
    4 pixel corners of the loop it was computed from; the resulting (M, N)
    map is then Gaussian-blurred so "nearby" residues raise a pixel's
    density even if the pixel itself isn't a corner of a residue loop.

    Parameters
    ----------
    residues : (M-1, N-1) int array
    sigma : Gaussian blur radius in pixels. 0 disables blurring (density is
        then just "is this pixel a corner of a nonzero residue loop").

    Returns
    -------
    density : (M, N) float array, >= 0.
    """

    assert residues.ndim == 2, "Expecting 2D map of phase residues."
    M, N = residues.shape[0]+1, residues.shape[1]+1
    density = np.zeros((M, N), dtype=np.float64)
    absres = np.abs(residues).astype(np.float64)
    density[:-1, :-1] += absres
    density[:-1, 1:] += absres
    density[1:, :-1] += absres
    density[1:, 1:] += absres

    if sigma > 0:
        density = gaussian_filter(density, sigma=sigma)
    return density


def calc_stat_costs(contrast, nlevels=2**16 - 1, eps=1e-9):
    """
    Statistically-motivated integer edge costs, replacing the ad hoc
    "-log(contrast)" transform with the actual interferometric phase-noise
    model.

    Background
    ----------
    For an interferometric/holographic measurement with coherence gamma, the
    Cramer-Rao-bound approximation for the phase standard deviation is
    (Rodriguez & Martin 1992):

        sigma_phi^2(gamma) = (1 - gamma^2) / (2 * gamma^2)

    A Gaussian model for phase noise gives log P(observed gradient) ~
    -(gradient)^2 / (2 * sigma^2), so the log-likelihood cost of "there was
    a real cycle slip here" is naturally proportional to 1/sigma^2 --
    i.e. *inverse-variance weighting* is the statistically correct edge
    cost, not an arbitrary log(contrast) transform. This is the same
    principle behind SNAPHU's "statistical cost mode".

    The variance of a *difference* of two (assumed independent) noisy
    phases is the sum of their variances, so each edge's cost uses
    sigma_A^2 + sigma_B^2 from its two endpoint pixels.

    Parameters
    ----------
    contrast : (M, N) ndarray, fringe visibility / coherence in [0, 1].
    nlevels : int, integer scaling factor controlling cost resolution
        (same role as in the original calc_costs).
    eps : numerical safety margin.

    Returns
    -------
    cost_h : (M, N-1) ndarray of int32
    cost_v : (M-1, N) ndarray of int32
    """

    contrast = np.asarray(contrast, dtype=np.float64)
    contrast = np.clip(contrast, 0.0, 1.0 - 1e-6)  # gamma=1 -> zero variance (singular cost); avoid it

    gamma2 = contrast ** 2
    sigma2 = (1.0 - gamma2) / (2.0 * gamma2 + eps)

    # combine endpoint variances (variance of a difference = sum of variances,
    # assuming the two pixels' noise is independent)
    sigma2_h = sigma2[:, :-1] + sigma2[:, 1:]
    sigma2_v = sigma2[:-1, :] + sigma2[1:, :]

    raw_cost_h = 1.0 / (sigma2_h + eps)
    raw_cost_v = 1.0 / (sigma2_v + eps)

    Nmax = max(raw_cost_h.max(), raw_cost_v.max())
    cost_h = np.round(nlevels * raw_cost_h / Nmax).astype(np.int32)
    cost_v = np.round(nlevels * raw_cost_v / Nmax).astype(np.int32)

    # floor at 1 to avoid zero-cost edge degeneracy in the MCF solver
    cost_h = np.clip(cost_h, 1, nlevels)
    cost_v = np.clip(cost_v, 1, nlevels)
    return cost_h, cost_v


def calc_log_costs(contrast, nlevels=2**16 - 1, eps=1e-9):
    """
    Compute integer edge costs for MCF phase unwrapping
    using fringe contrast (visibility) as the coherence measure.

    For holographic interferometry, fringe contrast V = |gamma_12| exactly,
    so this is not an approximation — contrast IS coherence.

    V = (I_max - I_min) / (I_max + I_min)  in [0, 1]

    High contrast -> reliable phase -> high cost (avoid cutting here)
    Low contrast  -> speckle null / noise -> low cost (preferred cut site)

    Parameters
    ----------
    contrast : (M, N) ndarray
        Fringe visibility / contrast in [0, 1].
    nlevels : int
        Integer scaling factor controlling cost resolution.
    eps : float
        Ensures log(0) is not computed.

    Returns
    -------
    cost_h : (M, N-1) ndarray of int32
        Costs for horizontal edges (pixel (i,j) <-> (i, j+1)).
    cost_v : (M-1, N) ndarray of int32
        Costs for vertical edges   (pixel (i,j) <-> (i+1, j)).
    """

    contrast = np.asarray(contrast, dtype=np.float64)
    contrast = np.clip(contrast, 0.0, 1.0)

    # Edge contrast: minimum of the two endpoint values.
    # We use min rather than mean because one low-contrast endpoint
    # is enough to make the phase unreliable on that edge.
    c_h = np.minimum(contrast[:, :-1], contrast[:, 1:])   # (M, N-1)
    c_v = np.minimum(contrast[:-1, :], contrast[1:, :])   # (M-1, N)
    log_cost_h = -np.log(c_h+eps)
    log_cost_v = -np.log(c_v+eps)

    cost_h = np.round(nlevels * log_cost_h).astype(np.int32)
    cost_v = np.round(nlevels * log_cost_v).astype(np.int32)

    # Floor at 1 to avoid zero-cost edge degeneracy in the MCF solver
    cost_h = np.clip(cost_h, 1, nlevels)
    cost_v = np.clip(cost_v, 1, nlevels)
    return cost_h, cost_v


def calc_lin_costs(contrast, nlevels=2**16 - 1):
    """
    Compute integer edge costs for MCF phase unwrapping
    using fringe contrast (visibility) as the coherence measure.

    For holographic interferometry, fringe contrast V = |gamma_12| exactly,
    so this is not an approximation — contrast IS coherence.

    V = (I_max - I_min) / (I_max + I_min)  in [0, 1]

    High contrast -> reliable phase -> high cost (avoid cutting here)
    Low contrast  -> speckle null / noise -> low cost (preferred cut site)

    Parameters
    ----------
    contrast : (M, N) ndarray
        Fringe visibility / contrast in [0, 1].
    nlevels : int
        Integer scaling factor controlling cost resolution.

    Returns
    -------
    cost_h : (M, N-1) ndarray of int32
        Costs for horizontal edges (pixel (i,j) <-> (i, j+1)).
    cost_v : (M-1, N) ndarray of int32
        Costs for vertical edges   (pixel (i,j) <-> (i+1, j)).
    """

    contrast = np.asarray(contrast, dtype=np.float64)
    contrast = np.clip(contrast, 0.0, 1.0)

    # Edge contrast: minimum of the two endpoint values.
    # We use min rather than mean because one low-contrast endpoint
    # is enough to make the phase unreliable on that edge.
    c_h = np.minimum(contrast[:, :-1], contrast[:, 1:])   # (M, N-1)
    c_v = np.minimum(contrast[:-1, :], contrast[1:, :])   # (M-1, N)

    cost_h = np.round(nlevels * (1-c_h)).astype(np.int32)
    cost_v = np.round(nlevels * (1-c_v)).astype(np.int32)

    # Floor at 1 to avoid zero-cost edge degeneracy in the MCF solver
    cost_h = np.clip(cost_h, 1, nlevels)
    cost_v = np.clip(cost_v, 1, nlevels)
    return cost_h, cost_v
