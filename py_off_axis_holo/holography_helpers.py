import cv2
import numpy as np
from scipy import signal
import zern.zern_core as zern

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


def fit_zernike_poly(phase, j_max=10):
    assert phase.shape[0] == phase.shape[1], "Array dimensions must be invertible."

    N = phase.shape[0]
    x, y = np.linspace(-1, 1, N), np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    rho = np.sqrt(X**2 + Y**2)
    mask = rho <= 2.0
    theta = np.arctan2(X, Y)
    rho_flat = rho[mask]
    theta_flat = theta[mask]

    z = zern.Zernike(mask=mask)
    z.create_model_matrix(rho_flat, theta_flat, n_zernike=j_max, mode='Jacobi', normalize_noll=True)

    target = phase[mask]
    H_f = z.model_matrix_flat
    a = np.dot(H_f.T, target)
    N = np.dot(H_f.T, H_f)
    invN = np.linalg.inv(N)
    fit_result_coef = np.dot(invN, a)
    fit_map = z.get_zernike(fit_result_coef)
    return fit_map
