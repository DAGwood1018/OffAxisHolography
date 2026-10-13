"""
Autofocus for objects using the transport-of-intensity equation (TIE).
Will only work well for phase-only objects.

Method
------
For a phase object, a small defocus dz produces intensity changes

    ln I(z) ~= ln I(z_f) - (wl * (z - z_f) / 2pi) * lap(phi)

so regressing band-passed ln|t|^2 against the band-passed phase Laplacian
gives a *signed* estimate of z - z_f at any plane. The focus is the plane
where that estimate crosses zero (with slope ~1). Uniform curvature
(constant Laplacian) is removed by the band-pass, and intensity patterns that
are uncorrelated with the object's phase (dust, parasitic fringes) add noise
but no bias.

Object and background fields are propagated separately and divided at each
plane:  t(z) = P_z{S} / P_z{B}.

Main entry points
-----------------
    OffAxisFilter.out_pxl_sz(mag)                  pixel pitch of OffAxisFilter output
    PairPropagator (holography.fourier_propagators) propagation of S/B pairs
    tie_defocus(t, wl, sz, mask)                   one-plane TIE estimate of z - z_f
    scan_tie(prop, S, B, zmin, zmax, dz)           TIE estimate over a range of planes
    tie_focus_iterate(prop, S, B, z_start)         Newton-like refinement
    find_focus_tie(S, B, wl, sz, zmin, zmax, ...)  scan + zero crossing -> z_f

Stacks: everything accepts a single field (M, N) or a stack (K, M, N), and
returns one result per array. A single background (M, N) can be shared by a
stack of object fields. Stacks are propagated together, which is about 2x
faster than looping.

Units: wl, sz, z in metres (any consistent unit works). Use the pixel pitch
of the field you pass in (see OffAxisFilter.out_pxl_sz) in the space where you
want z (object space: divide the camera pitch by the lateral magnification).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from py_off_axis_holo.holography.fourier_propagators import PairPropagator

def wrapped_laplacian(t):
    """Laplacian of arg(t) (rad/pixel^2) from wrapped differences, no unwrapping."""
    u = t / (np.abs(t) + 1e-12)
    gx = np.angle(u[..., :, 1:] * np.conj(u[..., :, :-1]))
    gy = np.angle(u[..., 1:, :] * np.conj(u[..., :-1, :]))
    lap = np.zeros(t.shape)
    lap[..., :, 1:-1] += gx[..., :, 1:] - gx[..., :, :-1]
    lap[..., 1:-1, :] += gy[..., 1:, :] - gy[..., :-1, :]
    return lap


def tie_defocus(t, wl, sz, mask=None, hp_sigma=15.0, lp_sigma=2.0):
    """
    TIE estimate of how far the plane of `t` is from the object plane.

    :param t: complex field (ideally object / background ratio), (M, N) or (K, M, N).
    :param wl: wavelength.
    :param sz: pixel pitch of t.
    :param mask: boolean pixels to use (None = all); (M, N) shared, or one per array.
    :param hp_sigma: Gaussian sigma (pixels) of the slow part removed from
        both ln I and lap(phi) (illumination drift, global curvature).
    :param lp_sigma: Gaussian sigma (pixels) of the noise smoothing.
    :return: (z - z_focus, correlation coefficient of the regression); floats
        for a single field, arrays of length K for a stack.
        |corr| near 0 means the plane carries no usable phase structure.
    """

    def _tie_terms(t, sz):
        """ln I and the phase Laplacian (rad/length^2) of a complex field."""
        I = np.abs(t) ** 2
        li = np.log(I + 1e-12 * I.mean(axis=(-2, -1), keepdims=True))
        lap = wrapped_laplacian(t) / sz ** 2
        return li, lap

    def _tie_regress(x, y, m, wl):
        """
        Regress band-passed ln I (y) on band-passed lap(phi) (x) over mask m,
        per array -> (z - z_f, corr).
        """
        ax = (-2, -1)
        m = m.astype(float)
        n = m.sum(axis=ax)
        sxy, sxx, syy = (m * x * y).sum(axis=ax), (m * x * x).sum(axis=ax), (m * y * y).sum(axis=ax)
        sx, sy = (m * x).sum(axis=ax), (m * y).sum(axis=ax)
        slope = sxy / sxx
        cov = sxy - sx * sy / n
        corr = cov / np.sqrt((sxx - sx ** 2 / n) * (syy - sy ** 2 / n))
        off = -2 * np.pi * slope / wl
        if np.ndim(off) == 0:
            return float(off), float(corr)
        return off, corr

    li, lap = _tie_terms(t, sz)
    gf = lambda a, sigma: gaussian_filter(a, (0,) * (a.ndim - 2) + (sigma, sigma))
    bp = lambda a: gf(a, lp_sigma) - gf(a, hp_sigma)
    if mask is None:
        mask = np.ones(t.shape[-2:], bool)
    return _tie_regress(bp(lap), bp(li), np.broadcast_to(mask, t.shape), wl)


def tie_focal_scan(prop, S, B, zmin, zmax, dz, mask=None, hp_sigma=15.0, lp_sigma=2.0):
    """
    TIE defocus estimate at every plane of np.arange(zmin, zmax + dz/2, dz).
    Spectra are computed once and the whole stack is propagated together.

    :param prop: PairPropagator for the field size.
    :param S, B: object / background fields, (M, N) or (K, M, N); B may be a
        single (M, N) background shared by the stack.
    :param mask: pixels to use; default focusing_mask(t(0)).
    :return: zaxis (nz,), offsets, correlations with shape (nz,) for a single
        field or (K, nz) for a stack. Near the focus offsets ~ z - z_f.
    """

    FS, FB = prop.spectrum(S), prop.spectrum(B)
    zaxis = np.arange(zmin, zmax + dz / 2, dz)
    if mask is None:
        mask = np.ones_like(FS, dtype=float)
    offs = np.empty(FS.shape[:-2] + zaxis.shape)
    corr = np.empty_like(offs)
    for i, z in enumerate(zaxis):
        t = prop.ratio_from_spectra(FS, FB, z)
        offs[..., i], corr[..., i] = tie_defocus(t, prop.wl, prop.sz, mask,
                                                 hp_sigma=hp_sigma, lp_sigma=lp_sigma)
    return zaxis, offs, corr


def zero_crossings(zaxis, offs):
    """
    Upward zero crossings of one TIE offset curve.

    :return: list of (z_cross, local slope). A trustworthy focus has slope ~1.
    """
    idx = np.where((np.diff(np.sign(offs)) != 0) & (np.diff(offs) > 0))[0]
    out = []
    for i in idx:
        dz = zaxis[i + 1] - zaxis[i]
        slope = (offs[i + 1] - offs[i]) / dz
        out.append((zaxis[i] - offs[i] / slope, slope))
    return out


def refine_tie_focus(prop, S, B, z_start=0.0, mask=None, n_iter=8, tol=0.2e-3, **kw):
    """
    Refine a focus estimate: z <- z - tie_defocus(t(z)). Converges in a few
    steps when started within the linear range of the TIE. For a stack every
    array is refined at its own z (z_start may be a scalar or one value per
    array); arrays that have converged are frozen.

    :return: z_focus (float or (K,)), history [(z, offset, corr), ...]
    """
    FS, FB = prop.spectrum(S), prop.spectrum(B)
    stacked = FS.ndim == 3
    z = np.broadcast_to(np.asarray(z_start, float), FS.shape[:-2]).copy() if stacked \
        else float(z_start)
    done = np.zeros(FS.shape[:-2], bool)
    hist = []
    for _ in range(n_iter):
        t = prop.ratio_from_spectra(FS, FB, z)
        dz, c = tie_defocus(t, prop.wl, prop.sz, mask, **kw)
        dz = np.where(done, 0.0, dz)
        hist.append((np.copy(z), dz, c))
        z = z - dz
        done = done | (np.abs(dz) < tol)
        if np.all(done):
            break
    return (z if stacked else float(z)), hist


def find_focus_tie(S, B, wl, sz, zmin, zmax, dz=None,
                   mask=None, hp_sigma=15.0, lp_sigma=2.0,
                   nb=0, paraxial=True, threads=1, prop=None):
    """
    Find the object plane of a transparent object from object/background
    fields. Works on a single pair or a stack (each array gets its own focus).

    1. Scan the TIE defocus estimate over [zmin, zmax].
    2. Take the upward zero crossing whose slope is closest to 1.
    3. Refine with tie_focus_iterate (kept only if it stays within 2*dz).

    :param S, B: object and background fields, (M, N) or (K, M, N); B may be a
        single (M, N) background shared by the stack. To propagate an already
        divided field t = S/B, pass (t, np.ones(t.shape[-2:])).
    :param wl, sz: wavelength and pixel pitch of S/B (see OffAxisFilter.out_pxl_sz).
    :param zmin, zmax: search range (same units, same space as sz).
    :param dz: scan step; default (zmax - zmin) / 40.
    :param mask: pixels to use; default focusing_mask(t(0)).
    :param nb: zero padding for propagation (ignored if prop is given).
    :param threads: FFT threads (ignored if prop is given).
    :param prop: an existing PairPropagator for this field size, to reuse its
        FFT plans across calls.
    :return: z_focus, t(z_focus). For a single pair z_focus is a float (NaN and
        t None if no zero crossing was found). For a stack z_focus has shape
        (K,) and t (K, M', N'); arrays without a crossing get NaN in both.
    """

    if prop is None:
        prop = PairPropagator(*np.shape(S)[-2:], wl, sz, nb=nb, paraxial=paraxial, threads=threads)
    if dz is None:
        dz = (zmax - zmin) / 40
    FS, FB = prop.spectrum(S), prop.spectrum(B)
    if mask is None:
        mask = np.ones_like(FS, dtype=float)
    zaxis, offs, _ = tie_focal_scan(prop, S, B, zmin, zmax, dz, mask, hp_sigma, lp_sigma)

    offs2 = np.atleast_2d(offs)
    z0 = np.full(offs2.shape[0], np.nan)
    for k, o in enumerate(offs2):
        cross = zero_crossings(zaxis, o)
        if cross:
            z0[k] = min(cross, key=lambda c: abs(c[1] - 1))[0]
    found = np.isfinite(z0)

    if offs.ndim == 1:                                   # single pair
        if not found[0]:
            return np.nan, None
        z1, _ = refine_tie_focus(prop, S, B, z0[0], mask, hp_sigma=hp_sigma, lp_sigma=lp_sigma)
        zf = z1 if abs(z1 - z0[0]) <= 2 * abs(dz) else z0[0]   # guard against drift
        return zf, prop.ratio_from_spectra(FS, FB, zf)

    if found.any():
        z1, _ = refine_tie_focus(prop, S, B, np.where(found, z0, 0.0), mask,
                                 hp_sigma=hp_sigma, lp_sigma=lp_sigma)
        keep = found & (np.abs(z1 - z0) <= 2 * abs(dz))
        z0 = np.where(keep, z1, z0)
    t0 = prop.ratio_from_spectra(FS, FB, np.where(found, z0, 0.0))
    t0[~found] = np.nan
    return z0, t0
