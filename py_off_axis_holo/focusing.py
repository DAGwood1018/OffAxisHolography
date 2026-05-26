import numpy as np
from numba import njit, prange

#TODO the logic of this search method is currently flawed I am missing the absolute min in my test case.

@njit
def AMP(field):
    """
    :param field: Complex field.
    :return: AMP measure of field. Minimized for pure amplitude, maximized for pure phase
    :rtype: float
    """

    return np.sum(np.sqrt(np.abs(field)).flatten())

@njit
def EIG(field, alpha=0.0):
    """
    :param field: Complex field.
    :param alpha: Percentage of eigenvalues to throw out. Only necessary if magnification isn't fixed (lensless).
    :return: Score related to how in-focus the image is. Maximized for pure amplitude, minimized for pure phase.
    :rtype: float
    """

    u = np.abs(field)
    N = np.sqrt(np.sum((u**2).flatten()))
    U = u / N
    mu = np.mean(U.flatten())
    G = U - mu
    eigvals = np.linalg.eigvalsh(G @ G.T)
    kappa = int(np.floor(alpha * len(eigvals)))
    if kappa == 0:
        return np.sum(eigvals)
    else:
        return np.sum(eigvals[0:-kappa])

@njit
def angular_spectrum(field, z, wl, sz):
    """
    Propagates a complex field to a single plane using the Angular Spectrum Method.

    :param field:       2D complex input field, shape (Ny, Nx).
    :param z:           Propagation distance.
    :param sz:          Pixel pitch (assumes square pixels).
    :param wl:  Wavelength of light.
    :return:            2D complex propagated field, shape (Ny, Nx).
    """

    Ny, Nx = field.shape
    k = 2 * np.pi / wl

    fx = np.fft.fftshift(np.fft.fftfreq(Nx, sz))
    fy = np.fft.fftshift(np.fft.fftfreq(Ny, sz))

    # Manual broadcasting: reshape to (1, Nx) and (Ny, 1)
    FX = fx.reshape(1, Nx)
    FY = fy.reshape(Ny, 1)

    arg = 1.0 - (wl * FX)**2 - (wl * FY)**2
    H = np.exp(1j * k * z * np.sqrt(arg))
    F = np.fft.fftshift(np.fft.fft2(field))
    return np.fft.ifft2(np.fft.ifftshift(F * H))


@njit(parallel=True)
def scan_focus(field, wl, sz, zmin, zmax, nz):
    """
    Propagates a complex field to N planes between zmin and zmax and computes
    the SPEC focus metric at each plane.

    :param field:       2D complex input field, shape (Ny, Nx).
    :param zmin:        Minimum propagation distance.
    :param zmax:        Maximum propagation distance.
    :param nz:           Number of z planes.
    :param sz:          Pixel pitch (assumes square pixels).
    :param wl:  Wavelength of light.
    :return:            z_values (N,), scores (N,).
    """

    z_values = np.linspace(zmin, zmax, nz)
    scores = np.empty(nz, dtype=np.float64)

    for i in prange(nz):
        fieldz = angular_spectrum(field, z_values[i], wl, sz)
        scores[i] = AMP(fieldz)
    return z_values, scores

@njit
def find_focal_dist(field, zaxis, scores, wl, sz, tol=1e-6):
    """
    Performs a golden section search around the global minimum found in scores.

    :param field:       2D complex input field.
    :param zaxis:    1D array of z values from SPEC_propagate.
    :param scores:      1D array of SPEC scores from SPEC_propagate.
    :param sz:          Pixel pitch.
    :param wl:  Wavelength of light.
    :param tol:         Convergence tolerance.
    :return:            z value at which SPEC is minimized.
    """

    idx = np.argmin(scores)

    if idx - 1 > 0:
        zmin = zaxis[idx - 1]
    else:
        zmin = zaxis[0]

    if idx + 1 < len(zaxis) - 1:
        zmax = zaxis[idx + 1]
    else:
        zmax = zaxis[len(zaxis) - 1]

    gr = (np.sqrt(5.0) + 1.0) / 2.0

    zc = zmax - (zmax - zmin) / gr
    zd = zmin + (zmax - zmin) / gr
    fc = AMP(angular_spectrum(field, zc, sz, wl))
    fd = AMP(angular_spectrum(field, zd, sz, wl))

    while abs(zmax - zmin) > tol:
        if fc < fd:
            zmax = zd
            zd, fd = zc, fc
            zc = zmax - (zmax - zmin) / gr
            fc = AMP(angular_spectrum(field, zc, sz, wl))
        else:
            zmin = zc
            zc, fc = zd, fd
            zd = zmin + (zmax - zmin) / gr
            fd = AMP(angular_spectrum(field, zd, sz, wl))

    return (zmin + zmax) / 2.0