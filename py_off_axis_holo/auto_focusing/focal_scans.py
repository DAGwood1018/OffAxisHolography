import numpy as np
from numba import njit, prange

from .metrics import gini_sparsity_metric, tamura_sparsity_metric


@njit(cache=True, inline='always')
def propagate(field, z, wl, sz, nb=0, paraxial=True):
    """
    Propagates a complex field to a single plane using the Angular Spectrum Method.

    Parameters
    ----------
    field : complex128[:, :]
        Input field, shape (Ny, Nx).
    z : float
        Propagation distance.
    wl : float
        Wavelength.
    sz : float
        Pixel pitch.
    nb : int, optional
        Zero-padding width on each side. Default is 0.

    Returns
    -------
    complex128[:, :]
        Propagated field cropped back to the original size.
    """

    Ny, Nx = field.shape
    k = 2.0 * np.pi / wl

    # Pad field if requested
    if nb > 0:
        Nyp = Ny + 2 * nb
        Nxp = Nx + 2 * nb

        padded = np.zeros((Nyp, Nxp), dtype=field.dtype)
        padded[nb:nb + Ny, nb:nb + Nx] = field
    else:
        padded = field
        Nyp, Nxp = Ny, Nx

    # Frequency coordinates on padded grid
    fx = np.fft.fftfreq(Nxp, sz)
    fy = np.fft.fftfreq(Nyp, sz)

    FX = fx.reshape(1, Nxp)
    FY = fy.reshape(Nyp, 1)

    if paraxial:
        H = np.exp(1j * k * z * (1-np.pi * wl *  (FX ** 2 + FY ** 2)))
    else:
        arg = 1.0 - (wl * FX) ** 2 - (wl * FY) ** 2
        H = np.exp(1j * k * z * np.sqrt(arg + 0j))

    Ft = np.fft.fft2(padded)
    propagated = np.fft.ifft2(Ft * H)

    # Crop back to original size
    if nb > 0:
        return propagated[nb:nb + Ny, nb:nb + Nx]
    return propagated

# --------------------------------------------------------------------------
# Autofocus scan helper
# --------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _autofocus_scan_core(stack: np.ndarray, use_gini: bool) -> np.ndarray:
    """njit worker: stack must already be the representation to score
    (complex128 wavefront, or real float64 amplitude/phase)."""
    n_planes = stack.shape[0]
    scores = np.empty(n_planes, dtype=np.float64)
    for i in prange(n_planes):
        if use_gini:
            scores[i] = edge_sparsity_gini(stack[i])
        else:
            scores[i] = edge_sparsity_tamura(stack[i])
    return scores


def autofocus_scan(stack: np.ndarray, use_gini: bool = True,
                    field_type: str = "complex") -> np.ndarray:
    """
    Apply the edge-sparsity Gini or Tamura metric to every plane of a
    reconstructed z-stack and return the metric curve.

    Parameters
    ----------
    stack : 3-D array, shape (n_planes, height, width)
        The numerically back-propagated complex wavefront U(x, y; z_i) at
        each candidate depth z_i, stacked along axis 0. May be complex128
        (a genuinely complex stack) or already-real float64 if you've
        precomputed amplitude/phase yourself.
    use_gini : bool
        If True, use the Gini index; otherwise use the Tamura coefficient.
    field_type : {"complex", "amplitude", "phase"}
        Which representation of the wavefront the edge map is computed on:
          * "complex"   -- use the complex field directly (the approach
                            matching "edge sparsity of the complex optical
                            wavefront"); requires a complex128 `stack` and
                            is jointly sensitive to amplitude and phase
                            edges in a single metric.
          * "amplitude" -- score np.abs(stack), i.e. the intensity/
                            amplitude edge map only.
          * "phase"     -- score np.angle(stack), i.e. the phase edge map
                            only. Note: np.angle wraps to (-pi, pi], so
                            genuine phase discontinuities (fringes, vortex
                            structures) and spurious 2*pi wrap edges are
                            not distinguished here; unwrap first if that
                            matters for your sample.
        Ignored (treated as a no-op) if `stack` is already a real array.

    Returns
    -------
    1-D float64 array of length n_planes with the metric value per plane.
    The in-focus plane is typically the argmax of this curve.
    """
    if np.iscomplexobj(stack):
        if field_type == "complex":
            data = stack
        elif field_type == "amplitude":
            data = np.abs(stack)
        elif field_type == "phase":
            data = np.angle(stack)
        else:
            raise ValueError(
                "field_type must be 'complex', 'amplitude', or 'phase'"
            )
    else:
        data = np.asarray(stack, dtype=np.float64)

    return _autofocus_scan_core(data, use_gini)

@njit(parallel=True, cache=True)
def scan_focus_gini(field, wl, sz, zmin, zmax, nz, nb=0, paraxial=True):
    """
    Propagates a complex field to N planes between zmin and zmax and computes
    the SPEC focus metric at each plane.

    :param field:       2D complex input field, shape (Ny, Nx).
    :param wl:          Wavelength of light.
    :param sz:          Pixel pitch (assumes square pixels).
    :param zmin:        Minimum propagation distance.
    :param zmax:        Maximum propagation distance.
    :param nz:          Number of z planes to sample.
    :param nb:          Zero padding to apply to the field. Default is 0.
    :param alpha:       If alpha<0 AMP method used, if 1>alpha>=0 determines the % of eigvals to discard in EIG method.
    :param paraxial:    Whether to use the paraxial approximation to propagation.
    :return:            zaxis (N,), scores (N,).
    """

    zaxis = np.linspace(zmin, zmax, nz)
    scores = np.empty(nz, dtype=np.float64)

    for i in prange(nz):
        fieldz = propagate(field, zaxis[i], wl, sz, nb=nb, paraxial=paraxial)
        scores[i] = gini_sparsity_metric(fieldz)
    return zaxis, scores

@njit(cache=True)
def find_best_focus_gini(field, wl, sz, zmin, zmax, tol=1e-6, nb=0, paraxial=True):
    """
    Performs a golden section search around the global minimum found in scores.

    :param field:       2D complex input field.
    :param wl:          Wavelength of light.
    :param sz:          Pixel pitch (assumes square pixels).
    :param zmin:        Minimum propagation distance.
    :param zmax:        Maximum propagation distance.
    :param tol:         Convergence tolerance.
    :param nb:          Zero padding to apply to the field. Default is 0.
    :param paraxial:    Whether to use the paraxial approximation to propagation.
    :return:            z value at which focus metric is minimized/maximized and the field at that z value.
    """

    gr = (np.sqrt(5.0) + 1.0) / 2.0
    zc = zmax - (zmax - zmin) / gr
    zd = zmin + (zmax - zmin) / gr

    Uc, Ud = propagate(field, zc, wl, sz, nb=nb, paraxial=paraxial), propagate(field, zd, wl, sz, nb=nb, paraxial=paraxial)
    fc = -gini_sparsity_metric(Uc)
    fd = -gini_sparsity_metric(Ud)
    while abs(zmax - zmin) > tol:
        if fc < fd:
            zmax = zd
            zd, fd = zc, fc
            zc = zmax - (zmax - zmin) / gr
            Uc = propagate(field, zc, wl, sz, nb=nb, paraxial=paraxial)
            fc = -gini_sparsity_metric(Uc)
        else:
            zmin = zc
            zc, fc = zd, fd
            zd = zmin + (zmax - zmin) / gr
            Ud = propagate(field, zd, wl, sz, nb=nb, paraxial=paraxial)
            fd = -gini_sparsity_metric(Ud)

    z0 = (zmin + zmax) / 2.0
    return z0, propagate(field, z0, wl, sz, nb=nb, paraxial=paraxial)

