import numpy as np
from numba import njit

from unwrapy.phase_unwrapping import wrap_to_pi


@njit(cache=True)
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


def residue_density(residues, M, N, sigma=3.0):
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
    M, N : output map shape
    sigma : Gaussian blur radius in pixels. 0 disables blurring (density is
        then just "is this pixel a corner of a nonzero residue loop").

    Returns
    -------
    density : (M, N) float array, >= 0.
    """
    density = np.zeros((M, N), dtype=np.float64)
    absres = np.abs(residues).astype(np.float64)
    density[:-1, :-1] += absres
    density[:-1, 1:] += absres
    density[1:, :-1] += absres
    density[1:, 1:] += absres

    if sigma > 0:
        from scipy.ndimage import gaussian_filter
        density = gaussian_filter(density, sigma=sigma)

    return density
