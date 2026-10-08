"""
Autofocus routines for digital holography using edge-sparsity metrics
(Gini index / Tamura coefficient, see metrics.py) on the complex field.

Propagation uses the angular spectrum method and is compiled with numba.
np.fft inside @njit requires the rocket-fft package:

    pip install rocket-fft

Main entry points
-----------------
    propagate(field, z, wl, sz, nb=0, paraxial=True)          [njit]
    scan_focus(field, wl, sz, zmin, zmax, nz, metric="gini", ...)
    find_best_focus(field, wl, sz, zmin, zmax, metric="gini", ...)
    autofocus_scan(stack, metric="gini")

Inputs are complex fields (amplitude + phase). All metrics are
*maximized* at focus.
"""

import numpy as np
from numba import njit, prange

from .metrics import gini_sparsity_metric, tamura_sparsity_metric


def _use_gini(metric):
    if metric == "gini":
        return True
    if metric == "tamura":
        return False
    raise ValueError(f"metric must be 'gini' or 'tamura', got {metric!r}")


def _as_field(field):
    field = np.ascontiguousarray(field, dtype=np.complex128)
    if field.ndim != 2:
        raise ValueError("field must be a 2-D complex array")
    return field


# --------------------------------------------------------------------------
# Propagation (angular spectrum method)
# --------------------------------------------------------------------------

@njit(cache=True)
def _fftfreq(n, d):
    """Same as np.fft.fftfreq(n, d)."""
    f = np.empty(n, dtype=np.float64)
    for i in range(n):
        f[i] = (i if i < (n + 1) // 2 else i - n) / (n * d)
    return f


@njit(cache=True)
def _transfer_function(Nyp, Nxp, z, wl, sz, paraxial):
    """
    Angular spectrum transfer function on an (Nyp, Nxp) FFT grid.

        exact    : H = exp(i k z sqrt(1 - wl^2 f^2)),  H = 0 where f > 1/wl
        paraxial : H = exp(i k z) * exp(-i pi wl z f^2)      (Fresnel)

    Evanescent components are dropped rather than allowed to grow
    exponentially for z < 0.
    """
    fx = _fftfreq(Nxp, sz)
    fy = _fftfreq(Nyp, sz)
    k = 2.0 * np.pi / wl
    H = np.empty((Nyp, Nxp), dtype=np.complex128)
    for iy in range(Nyp):
        for ix in range(Nxp):
            f2 = fy[iy] * fy[iy] + fx[ix] * fx[ix]
            if paraxial:
                H[iy, ix] = np.exp(1j * (k - np.pi * wl * f2) * z)
            else:
                arg = 1.0 - wl * wl * f2
                if arg > 0.0:
                    H[iy, ix] = np.exp(1j * k * z * np.sqrt(arg))
                else:
                    H[iy, ix] = 0.0
    return H


@njit(cache=True)
def _pad(field, nb):
    """Zero-pad by nb on each side (always returns a new complex128 array)."""
    Ny, Nx = field.shape
    out = np.zeros((Ny + 2 * nb, Nx + 2 * nb), dtype=np.complex128)
    out[nb:nb + Ny, nb:nb + Nx] = field
    return out


@njit(cache=True)
def _propagate_spectrum(Ft, z, wl, sz, Ny, Nx, nb, paraxial):
    """Propagate a precomputed (padded) spectrum Ft to z and crop to (Ny, Nx)."""
    Nyp, Nxp = Ft.shape
    H = _transfer_function(Nyp, Nxp, z, wl, sz, paraxial)
    U = np.fft.ifft2(Ft * H)
    return np.ascontiguousarray(U[nb:nb + Ny, nb:nb + Nx])


@njit(cache=True)
def propagate(field, z, wl, sz, nb=0, paraxial=True):
    """
    Propagate a complex field a distance z with the angular spectrum method.

    Parameters
    ----------
    field : 2-D complex128 array, shape (Ny, Nx)
    z : float          Propagation distance (same units as wl and sz).
    wl : float         Wavelength.
    sz : float         Pixel pitch (square pixels).
    nb : int           Zero-padding width on each side (reduces wrap-around).
    paraxial : bool    Fresnel (True) or exact angular spectrum (False).

    Returns
    -------
    2-D complex128 array, same shape as `field`.
    """
    Ny, Nx = field.shape
    Ft = np.fft.fft2(_pad(field, nb))
    return _propagate_spectrum(Ft, z, wl, sz, Ny, Nx, nb, paraxial)


# --------------------------------------------------------------------------
# numba workers
# --------------------------------------------------------------------------

@njit(cache=True)
def _score(U, use_gini):
    if use_gini:
        return gini_sparsity_metric(U)
    return tamura_sparsity_metric(U)


@njit(parallel=True, cache=True)
def _core_scan(field, zaxis, wl, sz, nb, paraxial, use_gini):
    """Score the field at every z in zaxis (parallel over planes).
    The input spectrum is computed once and shared by all planes."""
    Ny, Nx = field.shape
    Ft = np.fft.fft2(_pad(field, nb))
    scores = np.empty(zaxis.size, dtype=np.float64)
    for i in prange(zaxis.size):
        U = _propagate_spectrum(Ft, zaxis[i], wl, sz, Ny, Nx, nb, paraxial)
        scores[i] = _score(U, use_gini)
    return scores


@njit(cache=True)
def _golden_section(field, a, b, tol, wl, sz, nb, paraxial, use_gini):
    """Golden-section search for the metric maximum on [a, b].
    Returns (z0, score at z0)."""
    Ny, Nx = field.shape
    Ft = np.fft.fft2(_pad(field, nb))
    invphi = (np.sqrt(5.0) - 1.0) / 2.0

    c = b - (b - a) * invphi
    d = a + (b - a) * invphi
    fc = _score(_propagate_spectrum(Ft, c, wl, sz, Ny, Nx, nb, paraxial), use_gini)
    fd = _score(_propagate_spectrum(Ft, d, wl, sz, Ny, Nx, nb, paraxial), use_gini)
    while abs(b - a) > tol:
        if fc > fd:           # maximum lies in [a, d]
            b = d
            d, fd = c, fc
            c = b - (b - a) * invphi
            fc = _score(_propagate_spectrum(Ft, c, wl, sz, Ny, Nx, nb, paraxial), use_gini)
        else:                 # maximum lies in [c, b]
            a = c
            c, fc = d, fd
            d = a + (b - a) * invphi
            fd = _score(_propagate_spectrum(Ft, d, wl, sz, Ny, Nx, nb, paraxial), use_gini)

    z0 = 0.5 * (a + b)
    f0 = _score(_propagate_spectrum(Ft, z0, wl, sz, Ny, Nx, nb, paraxial), use_gini)
    return z0, f0

# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def scan_for_focus(field, wl, sz, zmin, zmax, nz, metric="gini", nb=0, paraxial=True):
    """
    Propagate a complex field to nz planes in [zmin, zmax] and score each.

    Returns
    -------
    zaxis (nz,), scores (nz,). Focus is at zaxis[np.argmax(scores)].
    """
    use_gini = _use_gini(metric)
    zaxis = np.linspace(zmin, zmax, int(nz))
    scores = _core_scan(_as_field(field), zaxis, float(wl), float(sz),
                        int(nb), bool(paraxial), use_gini)
    return zaxis, scores


def find_best_focus(field, wl, sz, zmin, zmax, metric="gini", nz_coarse=41,
                    tol=None, nb=0, paraxial=True, return_scan=False):
    """
    Locate the in-focus plane in two stages:

    1. Coarse scan of nz_coarse planes over [zmin, zmax] (parallel) to find
       the global maximum of the metric (robust to side lobes).
    2. Golden-section refinement inside the bracket around that maximum.

    Parameters
    ----------
    field : 2-D complex array.
    wl, sz, nb, paraxial : see propagate().
    zmin, zmax : float      Search range.
    metric : {"gini", "tamura"}
    nz_coarse : int         Number of coarse planes (>= 3). Make the step
                            smaller than the width of the focus peak.
    tol : float or None     Final bracket width. Default (zmax-zmin)*1e-5.
    return_scan : bool      Also return the coarse (zaxis, scores).

    Returns
    -------
    z0, U0                  Best-focus distance and the field there,
    (z0, U0, (zaxis, scores)) if return_scan.
    """
    use_gini = _use_gini(metric)
    if nz_coarse < 3:
        raise ValueError("nz_coarse must be >= 3")
    if tol is None:
        tol = abs(zmax - zmin) * 1e-5

    field = _as_field(field)
    wl, sz, nb, paraxial = float(wl), float(sz), int(nb), bool(paraxial)

    zaxis = np.linspace(zmin, zmax, int(nz_coarse))
    scores = _core_scan(field, zaxis, wl, sz, nb, paraxial, use_gini)

    i = int(np.argmax(scores))
    a = float(zaxis[max(i - 1, 0)])
    b = float(zaxis[min(i + 1, zaxis.size - 1)])
    z0, f0 = _golden_section(field, a, b, float(tol), wl, sz, nb, paraxial, use_gini)

    # Guard: never return something worse than the best coarse sample
    if f0 < scores[i]:
        z0 = float(zaxis[i])

    U0 = propagate(field, z0, wl, sz, nb, paraxial)
    if return_scan:
        return z0, U0, (zaxis, scores)
    return z0, U0


# --------------------------------------------------------------------------
# Convenience wrappers
# --------------------------------------------------------------------------

def scan_focus_gini(field, wl, sz, zmin, zmax, nz, nb=0, paraxial=True):
    return scan_for_focus(field, wl, sz, zmin, zmax, nz, metric="gini", nb=nb, paraxial=paraxial)


def scan_focus_tamura(field, wl, sz, zmin, zmax, nz, nb=0, paraxial=True):
    return scan_for_focus(field, wl, sz, zmin, zmax, nz, metric="tamura", nb=nb, paraxial=paraxial)


def find_best_focus_gini(field, wl, sz, zmin, zmax, tol=None, nb=0, paraxial=True):
    return find_best_focus(field, wl, sz, zmin, zmax, metric="gini", tol=tol, nb=nb, paraxial=paraxial)


def find_best_focus_tamura(field, wl, sz, zmin, zmax, tol=None, nb=0, paraxial=True):
    return find_best_focus(field, wl, sz, zmin, zmax, metric="tamura", tol=tol, nb=nb, paraxial=paraxial)
