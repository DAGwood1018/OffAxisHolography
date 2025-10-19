import cv2
import numpy as np
from scipy import signal

def format_img(arr):
    diff = arr.max() - arr.min()
    img = cv2.convertScaleAbs(arr, alpha=255.0 / diff, beta=-arr.min() * 255.0 / diff)
    return img.astype('uint8', copy=False)

def ref_phase_shift(XY, tilt, k0, sz):
    k = k0 * np.sin(tilt)
    return np.exp(1j * sz * (k[0] * XY[0] + k[1] * XY[1]))

def optimal_tilt(dims, k0, sz):
    N, M = dims[0], dims[1]
    fx = (8 * N + M - np.sqrt(8 * (N ** 2 + M ** 2) + 2 * N * M)) / 14
    fy = (N + 8 * M - np.sqrt(8 * (N ** 2 + M ** 2) + 2 * N * M)) / 14

    kx = 2 * np.pi * fx / (N * sz)
    ky = 2 * np.pi * fy / (M * sz)
    return - np.arcsin(np.array([kx, ky]) / k0)

def res_limit(f1, dims, sz):
    M, N = dims
    kmax = 2 * np.pi / sz
    f1 = f1[0] / N * kmax, f1[1] / M * kmax
    k = np.sqrt(np.sum(np.array(f1) ** 2)) / 3
    return 2 * np.pi / k

def phase_diff(dz, wl, n, n0=0):
    return 2 * np.pi * (n-n0) * dz / wl

def path_diff(phase, wl, n, n0=0):
    return phase * wl / (2 * np.pi * (n-n0))

def wrap_to_pi(phi):
    phi = phi - 2 * np.pi * np.floor((phi + np.pi) / (2 * np.pi))
    return phi

def tukey_window(dims, alpha=0.5):
    if dims is int or len(dims) == 1:
        return signal.windows.tukey(dims, alpha=alpha)
    window = signal.windows.tukey(dims[0], alpha=alpha)
    for n in dims[1:]:
        wn = signal.windows.tukey(n, alpha=alpha)
        window = np.tensordot(window, wn, axes=0)
    return window

def crop_to_mask(a, mask):
    coords = np.argwhere(mask)
    m_min, n_min = coords.min(axis=0)
    m_max, n_max = coords.max(axis=0)
    return a[m_min:m_max + 1, n_min:n_max + 1]

def gridspace(N, M, nb=0, center=False):
    """
    :param N, M: Shape of 2D arrays to operate on.
    :type N, M: int
    :param nb: Number of zeros to pad array dimensions by.
    :type nb: int
    :param center: Whether to center meshgrids at 0. Default is 'False'.
    :type center: bools
    :return: Meshgrids of X and Y.
    :rtype: ndarray, ndarray
    """

    nb = nb if nb > 0 else 0
    dims = 2 * nb + np.array([N, M])
    x = np.linspace(-0.5 * N, 0.5 * N, dims[0]) if center else np.linspace(0, N, dims[0])
    y = np.linspace(-0.5 * M, 0.5 * M, dims[1]) if center else np.linspace(0, M, dims[1])
    return np.meshgrid(x, y, indexing='xy')

def freqspace(N, M, nb=0):
    """
    :param N, M: Shape of 2D arrays to operate on.
    :type N, M: int
    :param nb: Number of zeros to pad array dimensions by.
    :type nb: int
    :return: Meshgrids of X and Y.
    :rtype: ndarray, ndarray
    """

    nb = nb if nb > 0 else 0
    dims = 2*nb + np.array([N, M])
    x = np.fft.fftfreq(dims[0])
    y = np.fft.fftfreq(dims[1])
    return np.meshgrid(x, y, indexing='xy')

def pad_arr(a, nb):
    """
    Adds padding (zero entries) to a ndarray.
    Array shape becomes nb * a.shape.

    :param a: Array to pad.
    :type a: ndarray
    :param nb: Number of zeros to pad array dimensions by.
    :type nb: int
    :return: Padded array.
    :rtype: ndarray
    """

    if nb <= 0:
        return a
    padding = tuple((nb, nb) for i in range(len(a.shape)))
    return np.pad(a, padding, mode='constant')

def unpad_arr(a, nb):
    """
    Removes padding (zero entries) from a ndarray.
    Shape becomes a.shape // nb.

    :param a: Array to unpad.
    :type a: ndarray
    :param nb: Number of zeros to pad array dimensions by.
    :type nb: int
    :return: Unpadded array.
    :rtype: ndarray
    """

    if nb <= 0:
        return a
    slice_indices = tuple(slice(nb, -nb) for i in range(len(a.shape)))
    return a[slice_indices]

def pad_to_square(image):
    """
    Pads a 2D image (NumPy array) to make it square.

    Parameters:
        image (ndarray): 2D input image.

    Returns:
        padded_image (ndarray): Square padded image.
    """

    h, w = image.shape
    if h == w:
        return image

    size = max(h, w)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2

    padded_image = np.pad(
        image,
        ((pad_h, size - h - pad_h), (pad_w, size - w - pad_w)),
        mode='constant',
        constant_values=0
    )
    return padded_image

def discrete_residue(F):
    """
    Vectorized computation of:
    np.round(((F[i, j+1]-F[i,j]) + (F[i+1,j+1]-F[i,j+1]) +
              (F[i+1,j]-F[i+1,j+1]) + (F[i,j]-F[i+1,j])) / (2*np.pi))
    """

    F = np.asarray(F)

    # compute the terms for interior points
    term1 = wrap_to_pi(F[:-1, 1:] - F[:-1, :-1])
    term2 = wrap_to_pi(F[1:, 1:] - F[:-1, 1:])
    term3 = wrap_to_pi(F[1:, :-1] - F[1:, 1:])
    term4 = wrap_to_pi(F[:-1, :-1] - F[1:, :-1])

    return np.round((term1 + term2 + term3 + term4) / (2 * np.pi))



