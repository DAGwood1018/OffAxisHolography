import cv2
import numpy as np
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


def ref_phase_shift(XY, tilt, k0, sz):
    """
    :param XY: The  X, Y meshgrids.
    :type XY: iterable<ndarray, ndarray>
    :param tilt: The tilt along x and y
    :type tilt: iterable<float, float>
    :param k0: The wavevector of the imaging light.
    :type k0: float
    :param sz: The size of the camera pixels.
    :type sz: float
    :return: Plane wave with phase shift given by the applied tilt.
    :rtype: ndarray
    """

    k = k0 * np.sin(tilt)
    return np.exp(1j * sz * (k[0] * XY[0] + k[1] * XY[1]))


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


def res_limit(f1, dims, sz):
    M, N = dims
    kmax = 2 * np.pi / sz
    f1 = f1[0] / N * kmax, f1[1] / M * kmax
    k = np.sqrt(np.sum(np.array(f1) ** 2)) / 3
    return 2 * np.pi / k


def phase_diff(dz, wl, n, n0=0):
    return 2 * np.pi * (n - n0) * dz / wl


def wrap_to_pi(phi):
    phi = phi - 2 * np.pi * np.floor((phi + np.pi) / (2 * np.pi))
    return phi


def tukey_window(dims, alpha=0.1):
    if dims is int or len(dims) == 1:
        return signal.windows.tukey(dims, alpha=alpha)
    window = signal.windows.tukey(dims[0], alpha=alpha)
    for n in dims[1:]:
        wn = signal.windows.tukey(n, alpha=alpha)
        window = np.tensordot(window, wn, axes=0)
    return window


def ellip_tukey_window(dims, alpha=0.1):
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


def crop_to_mask(a, mask):
    coords = np.argwhere(mask)
    m_min, n_min = coords.min(axis=0)
    m_max, n_max = coords.max(axis=0)
    return a[m_min:m_max + 1, n_min:n_max + 1]


def gridspace(M, N, nb=0, center=False):
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
    dims = 2 * nb + np.array([M, N])
    x = np.linspace(-0.5 * N, 0.5 * N, dims[1]) if center else np.linspace(0, N, dims[1])
    y = np.linspace(-0.5 * M, 0.5 * M, dims[0]) if center else np.linspace(0, M, dims[0])
    return np.meshgrid(x, y, indexing='xy')


def pad_arr(a, nb):
    """
    Adds padding (zero entries) to a ndarray.
    Array shape becomes 2*nb + a.shape.

    :param a: Array to pad.
    :type a: ndarray
    :param nb: Number of zeros to pad each array dimension by.
    :type nb: list<int>
    :return: Padded array.
    :rtype: ndarray
    """

    if type(nb) is int:
        if nb <= 0:
            return a
        nb = [nb for _ in range(len(a.shape))]
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
    :type nb: int
    :return: Unpadded array.
    :rtype: ndarray
    """

    if type(nb) is int:
        if nb <= 0:
            return a
        nb = [nb for _ in range(len(a.shape))]
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

def pad_to_sqr(image):
    """
    Pads a 2D image (NumPy array) to make it square.

    Parameters:
        image (ndarray): 2D input image.

    Returns:
        padded_image (ndarray): Square padded image.
        crop_h (slice): Indices to crop to along image height to return to original aspect ratio
        crop_w (slice): Indices to crop to along image width to return to original aspect ratio
    """

    assert image.ndim == 2, "Only designed for 2D images."
    h, w = image.shape
    if h == w:
        return image

    size = max(h, w)
    pad_h_r = (size - h) // 2
    pad_w_r = (size - w) // 2
    if size % 2 == 0:
        pad_h_l = pad_h_r
        pad_w_l = pad_w_r
    else:
        pad_h_l = pad_h_r + 1
        pad_w_l = pad_w_r + 1

    padded_image = np.pad(
        image,
        ((pad_h_r, pad_h_l), (pad_w_r, pad_w_l)),
        mode='constant',
        constant_values=0
    )
    return padded_image, slice(pad_h_r, -pad_h_l), slice(pad_w_r, -pad_w_l)

