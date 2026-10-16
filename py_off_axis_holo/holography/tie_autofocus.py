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

Band-pass
---------
Every function takes a `bandpass` function a -> a' that is applied to both
ln I and lap(phi) (arrays of shape (M, N) or (K, M, N)). The default is
holography_helpers.gaussian_bandpass (difference of Gaussians, lp_sigma=2,
hp_sigma=15 pixels); use functools.partial to change the widths. A band-pass
may return NaN for pixels that should be ignored (border, dead regions); the
regression skips non-finite pixels.

Auto mask
---------
With auto_mask=True the regression ignores
  * "dead" pixels: where the smoothed |t| at the recorded plane (z = 0) is below
    half its median. These are pixels without usable fringe modulation, e.g.
    an overexposed (saturated) core or light deflected outside the recorded
    band; their ln I and phase are meaningless.
  * a border of 10% of the smaller image side, where the band-pass filters
    run off the edge of the image.
The mask is found once, at z = 0, and used for every plane (the region that
was not recorded does not move when the field is propagated).

Main entry points
-----------------
    OffAxisFilter.out_pxl_sz(mag)                  pixel pitch of OffAxisFilter output
    PairPropagator (holography.fourier_propagators) propagation of S/B pairs
    tie_defocus(t, wl, sz, bandpass)               one-plane TIE estimate of z - z_f
    tie_mask(t)                                    the pixels kept with auto_mask=True
    tie_focal_scan(prop, S, B, zmin, zmax, dz)     TIE estimate over a range of planes
    refine_tie_focus(prop, S, B, z_start)          Newton-like refinement
    find_focus_tie(S, B, wl, sz, zmin, zmax, ...)  scan + zero crossing -> z_f

Stacks: everything accepts a single field (M, N) or a stack (K, M, N), and
returns one result per array. A single background (M, N) can be shared by a
stack of object fields.

Units: wl, sz, z in metres (any consistent unit works). Use the pixel pitch
of the field you pass in (see OffAxisFilter.out_pxl_sz) in the space where you
want z (object space: divide the camera pitch by the lateral magnification).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from py_off_axis_holo.holography.fourier_propagators import PairPropagator
from py_off_axis_holo.holography.holography_helpers import gaussian_bandpass


def wrapped_laplacian(t):
    """Laplacian of arg(t) (rad/pixel^2) from wrapped differences, no unwrapping."""
    u = t / (np.abs(t) + 1e-12)
    gx = np.angle(u[..., :, 1:] * np.conj(u[..., :, :-1]))
    gy = np.angle(u[..., 1:, :] * np.conj(u[..., :-1, :]))
    lap = np.zeros(t.shape)
    lap[..., :, 1:-1] += gx[..., :, 1:] - gx[..., :, :-1]
    lap[..., 1:-1, :] += gy[..., 1:, :] - gy[..., :-1, :]
    return lap


DEAD_REL_AMP = 0.5        # dead if smoothed |t| < DEAD_REL_AMP * median(|t|)
DEAD_SMOOTH = 3.0         # smoothing (pixels) of |t| before thresholding
BORDER_FRAC = 0.1         # border width as a fraction of the smaller image side


def tie_mask(t):
    """
    Pixels to keep for the TIE regression (see module docstring).

    :param t: field (M, N) or stack (K, M, N), normally t at the recorded plane.
    :return: boolean mask, same shape as t; True = keep.
    """
    sig = (0,) * (t.ndim - 2) + (DEAD_SMOOTH, DEAD_SMOOTH)
    a = gaussian_filter(np.abs(t), sig)
    keep = a >= DEAD_REL_AMP * np.median(a, axis=(-2, -1), keepdims=True)
    b = int(round(BORDER_FRAC * min(t.shape[-2:])))
    if b > 0:
        keep[..., :b, :] = keep[..., -b:, :] = False
        keep[..., :, :b] = keep[..., :, -b:] = False
    return keep


def _masked(bandpass, keep):
    """Band-pass that returns NaN outside keep."""
    return lambda a: np.where(keep, bandpass(a), np.nan)


def _tie_regress(x, y, wl):
    """
    Regress band-passed ln I (y) on band-passed lap(phi) (x) per array, over the
    pixels where both are finite -> (z - z_f, corr).
    """
    ax = (-2, -1)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.where(m, x, 0.0), np.where(m, y, 0.0)
    n = m.sum(axis=ax)
    sxy, sxx, syy = (x * y).sum(axis=ax), (x * x).sum(axis=ax), (y * y).sum(axis=ax)
    sx, sy = x.sum(axis=ax), y.sum(axis=ax)
    slope = sxy / sxx
    cov = sxy - sx * sy / n
    corr = cov / np.sqrt((sxx - sx ** 2 / n) * (syy - sy ** 2 / n))
    off = -2 * np.pi * slope / wl
    if np.ndim(off) == 0:
        return float(off), float(corr)
    return off, corr


def tie_defocus(t, wl, sz, bandpass=gaussian_bandpass, auto_mask=False):
    """
    TIE estimate of how far the plane of `t` is from the object plane.

    :param t: complex field (ideally object / background ratio), (M, N) or (K, M, N).
    :param wl: wavelength.
    :param sz: pixel pitch of t.
    :param bandpass: function applied to ln I and lap(phi) (see module docstring).
    :param auto_mask: ignore dead pixels and the border, found from this t
        (see module docstring).
    :return: (z - z_focus, correlation coefficient of the regression); floats
        for a single field, arrays of length K for a stack.
        |corr| near 0 means the plane carries no usable phase structure.
    """

    I = np.abs(t) ** 2
    li = np.log(I + 1e-12 * I.mean(axis=(-2, -1), keepdims=True))
    lap = wrapped_laplacian(t) / sz ** 2
    if auto_mask:
        bandpass = _masked(bandpass, tie_mask(t))
    return _tie_regress(bandpass(lap), bandpass(li), wl)


def tie_focal_scan(prop, S, B, zmin, zmax, dz, bandpass=gaussian_bandpass, auto_mask=False):
    """
    TIE defocus estimate at every plane of np.arange(zmin, zmax + dz/2, dz).
    Spectra are computed once and the whole stack is propagated together.

    :param prop: PairPropagator for the field size.
    :param S, B: object / background fields, (M, N) or (K, M, N); B may be a
        single (M, N) background shared by the stack.
    :param bandpass: function applied to ln I and lap(phi).
    :param auto_mask: ignore dead pixels (from t at z = 0) and the border.
    :return: zaxis (nz,), offsets, correlations with shape (nz,) for a single
        field or (K, nz) for a stack. Near the focus offsets ~ z - z_f.
    """

    FS, FB = prop.forwards(S), prop.forwards(B)
    if auto_mask:
        bandpass = _masked(bandpass, tie_mask(prop.ratio(FS, FB, 0.0)))
    zaxis = np.arange(zmin, zmax + dz / 2, dz)
    offs = np.empty(FS.shape[:-2] + zaxis.shape)
    corr = np.empty_like(offs)
    for i, z in enumerate(zaxis):
        t = prop.ratio(FS, FB, z)
        offs[..., i], corr[..., i] = tie_defocus(t, prop.wl, prop.out_pxl_sz(), bandpass)
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


def refine_tie_focus(prop, S, B, z_start=0.0, bandpass=gaussian_bandpass, auto_mask=False,
                     n_iter=8, tol=0.2e-3):
    """
    Refine a focus estimate: z <- z - tie_defocus(t(z)). Converges in a few
    steps when started within the linear range of the TIE. For a stack every
    array is refined at its own z (z_start may be a scalar or one value per
    array); arrays that have converged are frozen. auto_mask: ignore dead
    pixels (from t at z = 0) and the border.

    :return: z_focus (float or (K,)), history [(z, offset, corr), ...]
    """
    FS, FB = prop.forwards(S), prop.forwards(B)
    if auto_mask:
        bandpass = _masked(bandpass, tie_mask(prop.ratio(FS, FB, 0.0)))
    stacked = FS.ndim == 3
    z = np.broadcast_to(np.asarray(z_start, float), FS.shape[:-2]).copy() if stacked \
        else float(z_start)
    done = np.zeros(FS.shape[:-2], bool)
    hist = []
    for _ in range(n_iter):
        t = prop.ratio(FS, FB, z)
        dz, c = tie_defocus(t, prop.wl, prop.out_pxl_sz(), bandpass)
        dz = np.where(done, 0.0, dz)
        hist.append((np.copy(z), dz, c))
        z = z - dz
        done = done | (np.abs(dz) < tol)
        if np.all(done):
            break
    return (z if stacked else float(z)), hist


def find_focus_tie(S, B, wl, sz, zmin, zmax, nz=100, bandpass=gaussian_bandpass, auto_mask=False,
                   **kwargs):
    """
    Find the object plane of a transparent object from object/background
    fields. Works on a single pair or a stack (each array gets its own focus).

    1. Scan the TIE defocus estimate over [zmin, zmax].
    2. Take the upward zero crossing whose slope is closest to 1.
    3. Refine with refine_tie_focus (kept only if it stays within 2*dz).

    :param S, B: object and background fields, (M, N) or (K, M, N); B may be a
        single (M, N) background shared by the stack. To propagate an already
        divided field t = S/B, pass (t, np.ones(t.shape[-2:])).
    :param wl, sz: wavelength and pixel pitch of S/B (see OffAxisFilter.out_pxl_sz).
    :param zmin, zmax: search range (same units, same space as sz).
    :param nz: number of scan steps; default 100.
    :param bandpass: function applied to ln I and lap(phi); default gaussian_bandpass.
        Its widths are in pixels of the propagated field (multiply them by
        `upsample` if you upsample).
    :param auto_mask: ignore dead pixels (e.g. an overexposed core; found from t
        at z = 0) and a border of 10% of the image (see module docstring).
    :param kwargs: passed to PairPropagator (nb, upsample, paraxial, threads, ...).
    :return: z_focus, t(z_focus). For a single pair z_focus is a float (NaN and
        t None if no zero crossing was found). For a stack z_focus has shape
        (K,) and t (K, M', N'); arrays without a crossing get NaN in both.
    """

    prop = PairPropagator(*np.shape(S)[-2:], wl, sz, **kwargs)
    dz = (zmax - zmin) / nz
    FS, FB = prop.forwards(S), prop.forwards(B)
    if auto_mask:                     # find the mask once and reuse it below
        bandpass = _masked(bandpass, tie_mask(prop.ratio(FS, FB, 0.0)))
    zaxis, offs, _ = tie_focal_scan(prop, S, B, zmin, zmax, dz, bandpass)

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
        z1, _ = refine_tie_focus(prop, S, B, z0[0], bandpass)
        zf = z1 if abs(z1 - z0[0]) <= 2 * abs(dz) else z0[0]   # guard against drift
        return zf, prop.ratio(FS, FB, zf)

    if found.any():
        z1, _ = refine_tie_focus(prop, S, B, np.where(found, z0, 0.0), bandpass)
        keep = found & (np.abs(z1 - z0) <= 2 * abs(dz))
        z0 = np.where(keep, z1, z0)
    t0 = prop.ratio(FS, FB, np.where(found, z0, 0.0))
    t0[~found] = np.nan
    return z0, t0
