"""
Autofocus for transparent (pure-phase) objects using the transport-of-
intensity equation (TIE).

Why not the sparsity metrics in metrics.py / focal_scans.py?
------------------------------------------------------------
Gini / Tamura reward concentrated energy and sharp edges, which is what
"in focus" means for amplitude objects. A transparent object (e.g. a fluid
surface) has *uniform* amplitude at its own plane; intensity contrast only
appears with defocus. A curved surface also acts as a lens, and sparsity
metrics lock onto that lens' focal spot instead of the surface.

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
    filtered_pixel_size(spfilter, sz_cam, mag)     pixel pitch of OffAxisFilter output
    PairPropagator (holography.fourier_propagators) cached propagation of an S/B pair
    tie_defocus(t, wl, sz, mask)                   one-plane TIE estimate of z - z_f
    find_focus_tie(S, B, wl, sz, zmin, zmax, ...)  scan + zero crossing -> z_f
    tie_focus_iterate(prop, z_start, mask)         Newton-like refinement

Units: wl, sz, z in metres (any consistent unit works). Use the pixel pitch
of the field you pass in (see filtered_pixel_size) in the space where you
want z (object space: divide the camera pitch by the lateral magnification).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from py_off_axis_holo.holography.fourier_propagators import PairPropagator  # noqa: F401 (re-exported)


# --------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------

def focusing_mask(t, margin=25, rel_amp=0.5, smooth=3.0):
    """
    Pixels to use for focusing: drop a border of `margin` pixels and the
    "dead" regions where the hologram has no fringe modulation
    (smoothed |t| < rel_amp * median), whose phase is undefined.
    """
    a = gaussian_filter(np.abs(t), smooth)
    m = a > rel_amp * np.median(a)
    if margin > 0:
        m[:margin, :] = m[-margin:, :] = False
        m[:, :margin] = m[:, -margin:] = False
    return m


# --------------------------------------------------------------------------
# TIE estimator
# --------------------------------------------------------------------------

def wrapped_laplacian(t):
    """Laplacian of arg(t) (rad/pixel^2) from wrapped differences, no unwrapping."""
    u = t / (np.abs(t) + 1e-12)
    gx = np.angle(u[:, 1:] * np.conj(u[:, :-1]))
    gy = np.angle(u[1:, :] * np.conj(u[:-1, :]))
    lap = np.zeros(t.shape)
    lap[:, 1:-1] += gx[:, 1:] - gx[:, :-1]
    lap[1:-1, :] += gy[1:, :] - gy[:-1, :]
    return lap


def tie_defocus(t, wl, sz, mask=None, hp_sigma=25.0, lp_sigma=2.0):
    """
    TIE estimate of how far the plane of `t` is from the object plane.

    :param t: complex field (ideally object / background ratio).
    :param wl: wavelength.
    :param sz: pixel pitch of t.
    :param mask: boolean pixels to use (None = all).
    :param hp_sigma: Gaussian sigma (pixels) of the slow part removed from
        both ln I and lap(phi) (illumination drift, global curvature).
    :param lp_sigma: Gaussian sigma (pixels) of the noise smoothing.
    :return: (z - z_focus, correlation coefficient of the regression).
        |corr| near 0 means the plane carries no usable phase structure.
    """
    I = np.abs(t) ** 2
    li = np.log(I + 1e-12 * I.mean())
    lap = wrapped_laplacian(t) / sz ** 2
    bp = lambda a: gaussian_filter(a, lp_sigma) - gaussian_filter(a, hp_sigma)
    li, lap = bp(li), bp(lap)
    if mask is None:
        mask = np.ones(t.shape, bool)
    x, y = lap[mask], li[mask]
    slope = np.dot(x, y) / np.dot(x, x)
    corr = np.corrcoef(x, y)[0, 1]
    return -2 * np.pi * slope / wl, corr


def scan_tie(prop, zmin, zmax, dz, mask=None, hp_sigma=25.0, lp_sigma=2.0):
    """
    TIE defocus estimate at every plane of np.arange(zmin, zmax + dz/2, dz).

    :return: zaxis, offsets, correlations. Near the focus offsets ~ z - z_f.
    """
    zaxis = np.arange(zmin, zmax + dz / 2, dz)
    offs = np.empty(zaxis.size)
    corr = np.empty(zaxis.size)
    for i, z in enumerate(zaxis):
        offs[i], corr[i] = tie_defocus(prop.ratio(z), prop.wl, prop.sz, mask,
                                       hp_sigma=hp_sigma, lp_sigma=lp_sigma)
    return zaxis, offs, corr


def zero_crossings(zaxis, offs):
    """
    Upward zero crossings of the TIE offset curve.

    :return: list of (z_cross, local slope). A trustworthy focus has slope ~1.
    """
    idx = np.where((np.diff(np.sign(offs)) != 0) & (np.diff(offs) > 0))[0]
    out = []
    for i in idx:
        dz = zaxis[i + 1] - zaxis[i]
        slope = (offs[i + 1] - offs[i]) / dz
        out.append((zaxis[i] - offs[i] / slope, slope))
    return out


def tie_focus_iterate(prop, z_start=0.0, mask=None, n_iter=8, tol=0.2e-3, **kw):
    """
    Refine a focus estimate: z <- z - tie_defocus(t(z)). Converges in a few
    steps when started within the linear range of the TIE.

    :return: z_focus, history [(z, offset, corr), ...]
    """
    z, hist = float(z_start), []
    for _ in range(n_iter):
        dz, c = tie_defocus(prop.ratio(z), prop.wl, prop.sz, mask, **kw)
        hist.append((z, dz, c))
        z -= dz
        if abs(dz) < tol:
            break
    return z, hist


def find_focus_tie(S, B, wl, sz, zmin, zmax, dz=None, mask=None, nb=128,
                   hp_sigma=25.0, lp_sigma=2.0, refine=True, paraxial=True,
                   threads=1, return_scan=False):
    """
    Find the object plane of a transparent object from an object/background
    field pair.

    1. Scan the TIE defocus estimate over [zmin, zmax].
    2. Take the upward zero crossing whose slope is closest to 1.
    3. Optionally refine with tie_focus_iterate.

    :param S, B: object and background fields (B may be None).
    :param wl, sz: wavelength and pixel pitch of S/B (see filtered_pixel_size).
    :param zmin, zmax: search range (same units, same space as sz).
    :param dz: scan step; default (zmax - zmin) / 40.
    :param mask: pixels to use; default default_mask(t(0)).
    :param nb: zero padding for propagation.
    :param threads: FFT threads.
    :return: z_focus, t(z_focus)  [, (zaxis, offsets, correlations)]
        z_focus is NaN if no zero crossing was found.
    """
    prop = PairPropagator(S, B, wl, sz, nb=nb, paraxial=paraxial, threads=threads)
    if mask is None:
        mask = focusing_mask(prop.ratio(0.0))
    if dz is None:
        dz = (zmax - zmin) / 40
    zaxis, offs, corr = scan_tie(prop, zmin, zmax, dz, mask, hp_sigma, lp_sigma)
    cross = zero_crossings(zaxis, offs)
    if not cross:
        z0 = np.nan
        t0 = None
    else:
        z0 = min(cross, key=lambda c: abs(c[1] - 1))[0]
        if refine:
            z1, _ = tie_focus_iterate(prop, z0, mask, hp_sigma=hp_sigma, lp_sigma=lp_sigma)
            if abs(z1 - z0) <= 2 * abs(dz):     # guard against drifting on noisy data
                z0 = z1
        t0 = prop.ratio(z0)
    if return_scan:
        return z0, t0, (zaxis, offs, corr)
    return z0, t0
