"""
Autofocus for objects using the transport-of-intensity equation (TIE).
Will only work well for phase-only objects.

Method
------
For a phase object, a small defocus dz produces intensity changes

    ln I(z) ~= ln I(z_f) - (wl * (z - z_f) / 2pi) * lap(phi)

so regressing band-passed ln|t|^2 against the band-passed phase Laplacian
gives a *signed* estimate of z - z_f at any plane. The focus is the plane
where that estimate crosses zero, and near it the estimate is linear in z with
slope ~1. Uniform curvature (constant Laplacian) is removed by the band-pass,
and intensity patterns that are uncorrelated with the object's phase (dust,
parasitic fringes) add noise but no bias.

Object and background fields are propagated separately and divided at each
plane:  t(z) = P_z{S} / P_z{B}.

Workflow
--------
    prop = PairPropagator(M, N, wl, sz, nb=...)
    mask = tie_mask(prop(S, B, 0.0))                     # optional
    zaxis, offs, corr = tie_focal_scan(prop, S, B, zmin, zmax, nz, bandpass, mask)
    cross = zero_crossings(zaxis, offs)                  # look for 'linear': True
    z_f, hist = refine_tie_focus(prop, S, B, cross[0]['z'], bandpass, mask)

Check the scan (and zero_crossings' 'linear' flag) before refining:
refine_tie_focus just iterates from the point it is given.
find_focus_tie does all of this in one call.

Band-pass and mask (independent)
--------------------------------
bandpass: function a -> a' applied to both ln I and lap(phi), arrays of shape
    (M, N) or (K, M, N). Default holography_helpers.gaussian_bandpass
    (difference of Gaussians, lp_sigma=2, hp_sigma=15 px); change the widths
    with functools.partial. None = no filtering. A band-pass may also return
    NaN for pixels to skip.
mask: boolean array, True = pixel used in the regression; shape (M, N), shared
    by a stack, or (K, M, N). None = all pixels. tie_mask(t) builds the usual
    one: "dead" pixels without fringe modulation (e.g. an overexposed core)
    and/or an image border. Build it from t at the recorded plane (z = 0); the
    region that was not recorded does not move when the field is propagated.
    The mask is applied after the band-pass, so it does not change the
    filtering, only which pixels enter the fit.

Validity
--------
* Propagation: the angular-spectrum kernel is only adequately sampled for
  |z| <= PairPropagator.near_field_limit() = N * sz**2 / wl (N = smaller padded
  dimension). The scan and the refinement warn when z goes beyond it; add
  padding (nb) to raise the limit.
* TIE linearisation: the estimate is linear in z only for roughly
  |z - z_f| <~ 4 pi lp_sigma**2 sz**2 / wl (lp_sigma of the band-pass, pixels).

Stacks: everything accepts a single field (M, N) or a stack (K, M, N), and
returns one result per array. A single background (M, N) can be shared by a
stack of object fields.

Units: wl, sz, z in metres (any consistent unit works). Use the pixel pitch
of the field you pass in (see OffAxisFilter.out_pxl_sz) in the space where you
want z (object space: divide the camera pitch by the lateral magnification).
"""

from __future__ import annotations

from warnings import warn

import numpy as np
from scipy.ndimage import gaussian_filter

from py_off_axis_holo.holography.fourier_propagators import PairPropagator


# --------------------------------------------------------------------------
# Mask
# --------------------------------------------------------------------------

def tie_mask(t, dead=True, border=True, rel_amp=0.5, smooth=3.0, border_frac=0.1):
    """
    Pixels to keep for the TIE regression.

    :param t: field (M, N) or stack (K, M, N), normally t at the recorded plane.
    :param dead: drop pixels whose smoothed |t| is below rel_amp * median(|t|)
        (no usable fringe modulation, e.g. an overexposed core).
    :param border: drop a border of border_frac * (smaller image side), where
        the band-pass runs off the edge of the image.
    :param rel_amp: threshold for dead pixels, relative to the median amplitude.
    :param smooth: Gaussian sigma (pixels) applied to |t| before thresholding.
    :param border_frac: border width as a fraction of the smaller image side.
    :return: boolean mask, same shape as t; True = keep.
    """

    keep = np.ones(t.shape, bool)
    if dead:
        a = gaussian_filter(np.abs(t), (0,) * (t.ndim - 2) + (smooth, smooth))
        keep &= a >= rel_amp * np.median(a, axis=(-2, -1), keepdims=True)
    if border:
        b = int(round(border_frac * min(t.shape[-2:])))
        if b > 0:
            keep[..., :b, :] = keep[..., -b:, :] = False
            keep[..., :, :b] = keep[..., :, -b:] = False
    return keep


def tie_defocus_fit(t, wl, sz, bandpass=None, mask=None, tol=1e-12):
    """
    TIE estimate of how far the plane of `t` is from the object plane.

    :param t: complex field (ideally object / background ratio), (M, N) or (K, M, N).
    :param wl: wavelength.
    :param sz: pixel pitch of t.
    :param bandpass: function applied to ln I and lap(phi); None = no filtering.
    :param mask: boolean pixels to use (see module docstring); None = all.
    :return: (z - z_focus, correlation coefficient of the regression); floats
        for a single field, arrays of length K for a stack.
        |corr| near 0 means the plane carries no usable phase structure.
    """

    def _phase_laplacian(field, tol=tol):
        """Laplacian of arg(t) (rad/pixel^2) from wrapped differences, doesn't require phase unwrapping."""
        u = field / (np.abs(field) + tol)
        gx = np.angle(u[..., :, 1:] * np.conj(u[..., :, :-1]))
        gy = np.angle(u[..., 1:, :] * np.conj(u[..., :-1, :]))
        lap = np.zeros(field.shape)
        lap[..., :, 1:-1] += gx[..., :, 1:] - gx[..., :, :-1]
        lap[..., 1:-1, :] += gy[..., 1:, :] - gy[..., :-1, :]
        return lap

    I = np.abs(t) ** 2
    log_int = np.log(I + 1e-12 * I.mean(axis=(-2, -1), keepdims=True))
    lap_phi = _phase_laplacian(t) / sz ** 2
    if bandpass is not None:
        log_int, lap_phi = bandpass(log_int), bandpass(lap_phi)

    ax = (-2, -1)
    m = np.isfinite(lap_phi) & np.isfinite(log_int)
    if mask is not None:
        m &= np.broadcast_to(mask, m.shape)
    lap_phi, log_int = np.where(m, lap_phi, 0.0), np.where(m, log_int, 0.0)
    n = m.sum(axis=ax)

    sxy = (lap_phi * log_int).sum(axis=ax)
    sxx = (lap_phi * lap_phi).sum(axis=ax)
    syy = (log_int * log_int).sum(axis=ax)
    sx, sy = lap_phi.sum(axis=ax), log_int.sum(axis=ax)

    slope = sxy / sxx
    cov = sxy - sx * sy / n
    corr = cov / np.sqrt((sxx - sx ** 2 / n) * (syy - sy ** 2 / n))
    off = -2 * np.pi * slope / wl
    if np.ndim(off) == 0:
        return float(off), float(corr)
    return off, corr


# --------------------------------------------------------------------------
# Scan and zero crossings
# --------------------------------------------------------------------------


def tie_focal_scan(prop, S, B, zmin, zmax, nz=100, bandpass=None, mask=None):
    """
    TIE defocus estimate at nz + 1 evenly spaced planes in [zmin, zmax].
    Spectra are computed once and the whole stack is propagated together.
    Warns if the range exceeds prop.near_field_limit().

    :param prop: PairPropagator for the field size.
    :param S, B: object / background fields, (M, N) or (K, M, N); B may be a
        single (M, N) background shared by the stack.
    :param zmin, zmax: scan range.
    :param nz: number of steps (nz + 1 planes). Default 100.
    :param bandpass: function applied to ln I and lap(phi); None = no filtering.
    :param mask: boolean pixels to use; None = all.
    :return: zaxis (nz+1,), offsets, correlations with shape (nz+1,) for a
        single field or (K, nz+1) for a stack. Near a focus offsets ~ z - z_f.
    """

    FS, FB = prop.forwards(S), prop.forwards(B)
    zaxis = np.linspace(zmin, zmax, int(nz) + 1)
    offs = np.empty(FS.shape[:-2] + zaxis.shape)
    corr = np.empty_like(offs)
    for i, z in enumerate(zaxis):
        t = prop.ratio(FS, FB, z)
        offs[..., i], corr[..., i] = tie_defocus_fit(t, prop.wl, prop.out_pxl_sz(), bandpass, mask)
    return zaxis, offs, corr


def zero_crossings(zaxis, offs, window=3, slope_range=(0.7, 1.4), min_r2=0.9):
    """
    Upward zero crossings of a TIE offset curve, with a test of whether the
    response around each one is linear with slope ~1 (as it should be near a
    real focus).

    :param zaxis: planes of the scan (nz,).
    :param offs: offsets from tie_focal_scan, (nz,) or (K, nz).
    :param window: number of scan points on each side of the crossing used for
        the local straight-line fit.
    :param slope_range: accepted interval for the fitted slope.
    :param min_r2: minimum coefficient of determination of the local fit.
    :return: for a single curve, a list of dicts with keys
        z (interpolated crossing), slope, r2, n (points in the fit) and linear
        (True if slope is in slope_range and r2 >= min_r2), sorted with the
        linear ones first and then by |slope - 1|. For a stack, one such list
        per array.
    """

    offs = np.asarray(offs, float)
    if offs.ndim == 2:
        return [zero_crossings(zaxis, o, window, slope_range, min_r2) for o in offs]

    out = []
    ok = np.isfinite(offs)
    idx = np.where(ok[:-1] & ok[1:] & (np.diff(np.sign(offs)) != 0) & (np.diff(offs) > 0))[0]
    for i in idx:
        z0 = zaxis[i] - offs[i] * (zaxis[i + 1] - zaxis[i]) / (offs[i + 1] - offs[i])
        lo, hi = max(i - window + 1, 0), min(i + window + 1, len(zaxis))
        zs, os_ = zaxis[lo:hi], offs[lo:hi]
        good = np.isfinite(os_)
        zs, os_ = zs[good], os_[good]
        if zs.size >= 3:
            slope, icpt = np.polyfit(zs, os_, 1)
            resid = os_ - (slope * zs + icpt)
            ss = np.sum((os_ - os_.mean()) ** 2)
            r2 = 1.0 - np.sum(resid ** 2) / ss if ss > 0 else 0.0
        else:
            slope = (offs[i + 1] - offs[i]) / (zaxis[i + 1] - zaxis[i])
            r2 = np.nan
        linear = bool(slope_range[0] <= slope <= slope_range[1] and np.isfinite(r2) and r2 >= min_r2)
        out.append(dict(z=float(z0), slope=float(slope), r2=float(r2), n=int(zs.size), linear=linear))
    out.sort(key=lambda c: (not c['linear'], abs(c['slope'] - 1)))
    return out


# --------------------------------------------------------------------------
# Refinement
# --------------------------------------------------------------------------

def refine_tie_focus(prop, S, B, z_start, bandpass=None, mask=None, n_iter=10, tol=1e-5):
    """
    Refine a focus estimate with Newton steps z <- z - tie_defocus_fit(t(z)),
    starting from z_start (e.g. the 'z' of a crossing from zero_crossings).
    No check is made that z_start is near a real zero crossing: inspect the
    scan / the 'linear' flag of zero_crossings first. Warns if an array does
    not converge within n_iter steps, and if any plane goes beyond
    prop.near_field_limit().

    :param prop: PairPropagator for the field size.
    :param S, B: object / background fields (single or stack; B may be shared).
    :param z_start: starting plane: scalar, or one value per array.
    :param bandpass: function applied to ln I and lap(phi); None = no filtering.
    :param mask: boolean pixels to use; None = all.
    :param n_iter: maximum number of Newton steps.
    :param tol: convergence tolerance on the step (same units as z).
    :return: z_focus (float, or (K,) for a stack), history [(z, step, corr), ...]
    """

    FS, FB = prop.forwards(S), prop.forwards(B)
    stacked = FS.ndim == 3
    shape = FS.shape[:-2]
    z = np.broadcast_to(np.asarray(z_start, float), shape).copy()

    done = np.zeros(shape, bool)
    hist = []
    for _ in range(n_iter):
        off, c = tie_defocus_fit(prop.ratio(FS, FB, z if stacked else float(z)),
                                 prop.wl, prop.out_pxl_sz(), bandpass, mask)
        step = np.where(done, 0.0, off)
        hist.append((np.copy(z), step, c))
        z = np.array(z - step, dtype=float)          # keep an array, even 0-d
        done |= np.abs(step) < tol
        if np.all(done):
            break
    if not np.all(done):
        which = "" if not stacked else f" (arrays {np.flatnonzero(~done).tolist()})"
        warn(f"refine_tie_focus: did not converge to tol {tol:.3g} in {n_iter} steps{which}", stacklevel=2)
    return (z if stacked else float(z)), hist


# --------------------------------------------------------------------------
# One-call convenience
# --------------------------------------------------------------------------

def find_focus_tie(S, B, wl, sz, zmin, zmax, nz=100, bandpass=None, mask=None,
                   **kwargs):
    """
    Convenience wrapper: tie_focal_scan -> zero_crossings -> refine_tie_focus.
    For each array the linear crossing with slope closest to 1 is refined
    Works on a single pair or a stack.

    :param S, B: object and background fields, (M, N) or (K, M, N); B may be a
        single (M, N) background shared by the stack. To propagate an already
        divided field t = S/B, pass (t, np.ones(t.shape[-2:])).
    :param wl, sz: wavelength and pixel pitch of S/B (see OffAxisFilter.out_pxl_sz).
    :param zmin, zmax: search range (same units, same space as sz).
    :param nz: number of scan steps; default 100.
    :param bandpass: function applied to ln I and lap(phi); None = no filtering.
        Its widths are in pixels of the propagated field (multiply them by
        `upsample` if you upsample).
    :param mask: boolean pixels to use, e.g. tie_mask(...) of t at z = 0; None = all.
    :param kwargs: passed to PairPropagator (nb, upsample, paraxial, threads, ...).
    :return: z_focus, t(z_focus), zaxis, offsets. z_focus is a float (NaN if no
        linear crossing was found) or (K,) for a stack; t is NaN where z_focus is.
    """

    prop = PairPropagator(*np.shape(S)[-2:], wl, sz, **kwargs)
    zaxis, offs, _ = tie_focal_scan(prop, S, B, zmin, zmax, nz, bandpass, mask)

    crosses = zero_crossings(zaxis, offs)
    if offs.ndim == 1:
        crosses = [crosses]
    z_start = np.array([c[0]['z'] if c and c[0]['linear'] else np.nan for c in crosses])
    found = np.isfinite(z_start)

    z_f = np.full(z_start.shape, np.nan)
    if found.any():
        sel = np.where(found)[0]
        Ssel = S if offs.ndim == 1 else np.asarray(S)[sel]
        Bsel = B if (offs.ndim == 1 or np.ndim(B) == 2) else np.asarray(B)[sel]
        msel = mask if (mask is None or offs.ndim == 1 or np.ndim(mask) == 2) else np.asarray(mask)[sel]
        zs = z_start[0] if offs.ndim == 1 else z_start[sel]
        zr, _ = refine_tie_focus(prop, Ssel, Bsel, zs, bandpass, msel)
        z_f[sel] = zr

    FS, FB = prop.forwards(S), prop.forwards(B)
    if offs.ndim == 1:
        if not np.isfinite(z_f[0]):
            return np.nan, np.full(prop.output_shape[-2:], np.nan, complex), zaxis, offs
        return float(z_f[0]), prop.ratio(FS, FB, float(z_f[0])), zaxis, offs
    ok = np.isfinite(z_f)
    t0 = prop.ratio(FS, FB, np.where(ok, z_f, 0.0))
    t0[~ok] = np.nan
    return z_f, t0, zaxis, offs

