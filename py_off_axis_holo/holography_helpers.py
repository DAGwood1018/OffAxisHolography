import cv2
import numpy as np
from numba import njit
from scipy import signal


def format_holo(arr):
    """
    :param arr: Hologram image.
    :type arr: ndaray
    :return: Hologram formatted to uint8.
    :rtype: ndarray<unit8>
    """

    diff = arr.max() - arr.min()
    img = cv2.convertScaleAbs(arr, alpha=255.0 / diff, beta=-arr.min() * 255.0 / diff)
    return img.astype('uint8', copy=False)


def optimal_tilt(dims, k0, sz):
    """
    :param dims: The  X, Y meshgrids.
    :type dims: iterable<int, int>
    :param k0: The wavevector of the imaging light.
    :type k0: float
    :param sz: The size of the camera pixels.
    :type sz: float
    :return: The tilt that will give the max/ideal resolution given system constraints.
    :rtype: ndarray
    """

    N, M = dims[0], dims[1]
    fx = (8 * N + M - np.sqrt(8 * (N ** 2 + M ** 2) + 2 * N * M)) / 14
    fy = (N + 8 * M - np.sqrt(8 * (N ** 2 + M ** 2) + 2 * N * M)) / 14

    kx = 2 * np.pi * fx / (N * sz)
    ky = 2 * np.pi * fy / (M * sz)
    return - np.arcsin(np.array([kx, ky]) / k0)


def tukey_window(dims, alpha=0.1):
    """
    Create a rectangular Tukey window.

    :param dims: Dimensions of window
    :type dims: int
    :param alpha: Size of transition region as a fraction. Default is 0.1.
    :type alpha: float
    :return: A Tukey window.
    :rtype: ndarray
    """

    if dims is int or len(dims) == 1:
        return signal.windows.tukey(dims, alpha=alpha)
    window = signal.windows.tukey(dims[0], alpha=alpha)
    for n in dims[1:]:
        wn = signal.windows.tukey(n, alpha=alpha)
        window = np.tensordot(window, wn, axes=0)
    return window


def ellip_tukey_window(dims, alpha=0.1):
    """
    Create an elliptical Tukey window.

    :param dims: Dimensions of window
    :type dims: int
    :param alpha: Size of transition region as a fraction. Default is 0.1.
    :type alpha: float
    :return: An elliptical Tukey window.
    :rtype: ndarray
    """

    if isinstance(dims, int) or len(dims) == 1:
        n = dims if isinstance(dims, int) else dims[0]
        return signal.windows.tukey(n, alpha=alpha)

    # Build coordinate grids normalized by each axis radius
    grids = []
    for n in dims:
        center = (n - 1) / 2
        r = n / 2
        coords = (np.arange(n) - center) / r  # values in (-1, 1]
        grids.append(coords)

    # Compute elliptical radius at each grid point
    mesh = np.meshgrid(*grids, indexing='ij')
    rho = np.sqrt(sum(g ** 2 for g in mesh))  # 0 at center, 1 on ellipse boundary

    N = max(dims)
    ref = signal.windows.tukey(N, alpha=alpha)

    # rho in [0,1] maps from center to right edge of the reference window
    idx = (rho * (N // 2 - 1)).astype(int).clip(0, N // 2 - 1)
    idx_from_center = N // 2 + idx

    window = ref[idx_from_center]
    window[rho > 1] = 0
    return window


def pad_arr(a, nb):
    """
    Adds padding (zero entries) to a ndarray.
    Array shape becomes 2*nb + a.shape.

    :param a: Array to pad.
    :type a: ndarray
    :param nb: Number of zeros to pad each array dimension by.
    :type nb: int, list<int>
    :return: Padded array.
    :rtype: ndarray
    """

    if type(nb) is int:
        if nb <= 0:
            return a
        nb = [nb for _ in range(len(a.shape))]
    if type(nb) is list:
        if len(nb) < len(a.shape):
            for i in range(len(a.shape)-len(nb)):
                nb.append(0)
    padding = tuple((nb[i], nb[i]) for i in range(len(a.shape)))
    return np.pad(a, padding, mode='constant')


def unpad_arr(a, nb):
    """
    Removes padding (zero entries) from a ndarray.
    Shape becomes a.shape - 2 * nb.

    :param a: Array to unpad.
    :type a: ndarray
    :param nb: Number of zeros to pad array dimensions by.
    :type nb: int, list<int>
    :return: Unpadded array.
    :rtype: ndarray
    """

    if type(nb) is int:
        if nb <= 0:
            return a
        nb = [nb for _ in range(len(a.shape))]
    elif type(nb) is list:
        if len(nb) < len(a.shape):
            for i in range(len(a.shape)-len(nb)):
                nb.append(0)

    slice_indices = []
    for i in range(len(a.shape)):
        if nb[i] == 0:
            slice_indices.append(slice(None))
        else:
            slice_indices.append(slice(nb[i], -nb[i]))
    return a[tuple(slice_indices)]


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
        Nyp, Nxp = Ny, Nx

    # Frequency coordinates on padded grid
    fx = np.fft.fftfreq(Nxp, sz)
    fy = np.fft.fftfreq(Nyp, sz)

    FX = fx.reshape(1, Nxp)
    FY = fy.reshape(Nyp, 1)

    arg = 1.0 - (wl * FX) ** 2 - (wl * FY) ** 2
    H = np.exp(1j * k * z * np.sqrt(arg + 0j))

    Ft = np.fft.fft2(padded)
    propagated = np.fft.ifft2(Ft * H)

    # Crop back to original size
    if nb > 0:
        return propagated[nb:nb + Ny, nb:nb + Nx]
    return propagated
