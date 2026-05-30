import numpy as np
from numba import njit, prange


@njit(cache=True)
def AMP(field):
    """
    :param field: Complex field.
    :return: AMP measure of field. Minimized for pure amplitude, maximized for pure phase
    :rtype: float
    """

    return np.sum(np.sqrt(np.abs(field)).flatten())

@njit(cache=True)
def EIG(field, alpha=0.0):
    """
    :param field: Complex field.
    :param alpha: Percentage of eigenvalues to throw out. Only necessary if magnification isn't fixed (lensless).
    :return: Score related to how in-focus the image is. Minimized for pure amplitude, maximized for pure phase.
    :rtype: float
    """

    if alpha < 0 or alpha >= 1.0:
        alpha = 0.0
    u = np.abs(field)
    N = np.sqrt(np.sum((u**2).flatten()))
    U = u / N
    mu = np.mean(U.flatten())
    G = U - mu
    eigvals = np.linalg.eigvalsh(G @ G.T)
    kappa = int(np.floor(alpha * len(eigvals)))
    if kappa == 0:
        return -1*np.sum(eigvals)
    else:
        return -1*np.sum(eigvals[0:-kappa])

@njit(cache=True)
def angular_spectrum(field, z, wl, sz, nb=0):
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
        Nyp = Ny
        Nxp = Nx

    # Frequency coordinates on padded grid
    fx = np.fft.fftfreq(Nxp, sz)
    fy = np.fft.fftfreq(Nyp, sz)

    FX = fx.reshape(1, Nxp)
    FY = fy.reshape(Nyp, 1)

    arg = 1.0 - (wl * FX) ** 2 - (wl * FY) ** 2
    H = np.exp(1j * k * z * np.sqrt(arg))

    Ft = np.fft.fft2(padded)
    propagated = np.fft.ifft2(Ft * H)

    # Crop back to original size
    if nb > 0:
        return propagated[nb:nb + Ny, nb:nb + Nx]
    return propagated

@njit(parallel=True, cache=True)
def scan_focus(field, wl, sz, zmin, zmax, nz, alpha=-1.0, nb=0):
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
    :return:            zaxis (N,), scores (N,).
    """

    zaxis = np.linspace(zmin, zmax, nz)
    scores = np.empty(nz, dtype=np.float64)

    for i in prange(nz):
        fieldz = angular_spectrum(field, zaxis[i], wl, sz, nb=nb)
        scores[i] = AMP(fieldz) if alpha < 0 or alpha >= 1.0 else EIG(fieldz, alpha)
    return zaxis, scores

@njit(cache=True)
def find_best_focus(field, wl, sz, zmin, zmax, tol=1e-6, alpha=-1.0, nb=0, maximize=False):
    """
    Performs a golden section search around the global minimum found in scores.

    :param field:       2D complex input field.
    :param wl:          Wavelength of light.
    :param sz:          Pixel pitch (assumes square pixels).
    :param zmin:        Minimum propagation distance.
    :param zmax:        Maximum propagation distance.
    :param tol:         Convergence tolerance.
    :param alpha:       If alpha<0 AMP method used, if 1>alpha>=0 determines the % of eigvals to discard in EIG method.
    :param nb:          Zero padding to apply to the field. Default is 0.
    :param maximize:    Whether the focus metric should be maximized. If False it is minimized.
    :return:            z value at which focus metric is minimized/maximized and the field at that z value.
    """

    gr = (np.sqrt(5.0) + 1.0) / 2.0
    zc = zmax - (zmax - zmin) / gr
    zd = zmin + (zmax - zmin) / gr

    Uc, Ud = angular_spectrum(field, zc, wl, sz, nb=nb), angular_spectrum(field, zd, wl, sz, nb=nb)
    fc = AMP(Uc) if alpha < 0 or alpha >= 1.0 else EIG(Uc, alpha)
    fd = AMP(Ud) if alpha < 0 or alpha >= 1.0 else EIG(Ud, alpha)
    if maximize:
        fc *= -1
        fd *= -1
    while abs(zmax - zmin) > tol:
        if fc < fd:
            zmax = zd
            zd, fd = zc, fc
            zc = zmax - (zmax - zmin) / gr
            Uc = angular_spectrum(field, zc, wl, sz, nb=nb)
            fc = AMP(Uc) if alpha < 0 or alpha >= 1.0 else EIG(Uc, alpha)
            if maximize:
                fc *= -1
        else:
            zmin = zc
            zc, fc = zd, fd
            zd = zmin + (zmax - zmin) / gr
            Ud = angular_spectrum(field, zd, wl, sz, nb=nb)
            fd = AMP(Ud) if alpha < 0 or alpha >= 1.0 else EIG(Ud, alpha)
            if maximize:
                fd *= -1

    z0 = (zmin + zmax) / 2.0
    return z0, angular_spectrum(field, z0, wl, sz, nb=nb)