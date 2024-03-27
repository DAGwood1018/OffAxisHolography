import cv2
import numpy as np
from scipy import signal

def reference_wave(XY, tilt, k0, sz, A=1):
    k = k0 * np.sin(tilt)
    return A * np.exp(1j * sz * (k[0] * XY[0] + k[1] * XY[1]))

def object_wave(XY, phi, k0, sz, A=1):
    return A * np.exp(1j * sz * k0 * (XY[0] + XY[1]) + 1j * phi)

def optimal_tilt(k0, sz, err=0.1):
    kmax = np.pi / sz
    k = (1 - err) * (2 * np.sqrt(2) / (3 + np.sqrt(2)) * kmax)
    return np.arcsin(k / k0)

def fringe_dist(f1, dims, pxl_sz=None):
    dims = pxl_sz * np.array(dims) if not pxl_sz is None else np.array(dims)
    M, N = tuple(dims)
    return 1 / np.sqrt((f1[0]/N)**2 + (f1[1]/M)**2)

def resolution_lim(f1, dims, sz):
    M, N = dims
    kmax = 2 * np.pi / sz
    f1 = f1[0] / N * kmax, f1[1] / M * kmax
    k = np.sqrt(np.sum(np.array(f1) ** 2)) / 3
    return 2 * np.pi / k

def phase_diff(h, wl, n, n0):
    return 2 * np.pi * (n-n0) * h / wl

def height_profile(phase, wl, n, n0):
    return phase * wl / (2 * np.pi * (n-n0))

def tukey_window(dims, alpha=0.5):
    if dims is int or len(dims) == 1:
        return signal.windows.tukey(dims, alpha=alpha)
    window = signal.windows.tukey(dims[0], alpha=alpha)
    for n in dims[1:]:
        wn = signal.windows.tukey(n, alpha=alpha)
        window = np.tensordot(window, wn, axes=0)
    return window


def gridspace(N, M, nb=1, center=False):
    """
    :param N, M: Shape of 2D arrays to operate on.
    :type N, M: int
    :param nb: Determines grid spacing.
    :type nb: float
    :param center: Whether to center meshgrids at 0. Default is 'False'.
    :type center: bools
    :return: Meshgrids of X and Y.
    :rtype: ndarray, ndarray
    """

    nb = nb if nb >= 1 else 1
    dims = np.ceil(nb * np.array([N, M])).astype('int64')
    x = np.linspace(-0.5 * N, 0.5 * N, dims[0]) if center else np.linspace(0, N, dims[0])
    y = np.linspace(-0.5 * M, 0.5 * M, dims[1]) if center else np.linspace(0, M, dims[1])
    return np.meshgrid(x, y, indexing='xy')


def freqspace(N, M, nb=1):
    """
    :param N, M: Shape of 2D arrays to operate on.
    :type N, M: int
    :param nb: Determines grid spacing.
    :type nb: float
    :return: Meshgrids of X and Y.
    :rtype: ndarray, ndarray
    """

    nb = nb if nb >= 1 else 1
    dims = np.ceil(nb * np.array([N, M])).astype('int64')
    x = np.fft.fftfreq(dims[0])
    y = np.fft.fftfreq(dims[1])
    return np.meshgrid(x, y, indexing='xy')


def pad_arr(a, nb):
    """
    Adds padding (zero entries) to a ndarray.
    Array shape becomes nb * a.shape.

    :param a: Array to pad.
    :type a: ndarray
    :param nb: Factor to pad size by.
    :type nb: float
    :return: Padded array.
    :rtype: ndarray
    """

    if nb <= 1:
        return a
    shape = np.ceil(nb * np.array(a.shape)).astype('int64')
    diff = shape - a.shape
    rpad = diff // 2
    lpad = rpad + diff % 2
    padding = tuple((r, l) for r, l in zip(rpad, lpad))
    return np.pad(a, padding, mode='constant')


def unpad_arr(a, nb):
    """
    Removes padding (zero entries) from a ndarray.
    Shape becomes a.shape // nb.

    :param a: Array to unpad.
    :type a: ndarray
    :param nb: Factor to unpad size by.
    :type nb: float
    :return: Unpadded array.
    :rtype: ndarray
    """

    if nb <= 1:
        return a
    shape = np.array(a.shape) // nb
    diff = np.array(a.shape) - shape
    rb = (diff // 2).astype('int64')
    lb = (rb + diff % 2).astype('int64')
    ind = tuple(slice(r, -l) for r, l in zip(rb, lb))
    return a[ind]


def format_img(arr):
    diff = arr.max() - arr.min()
    img = cv2.convertScaleAbs(arr, alpha=255.0 / diff, beta=-arr.min() * 255.0 / diff)
    return img.astype('uint8', copy=False)
