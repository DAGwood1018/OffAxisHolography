from _warnings import warn

import cv2
import numpy as np
from scipy import signal
from skimage.feature import peak_local_max


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


def select_mask_roi(Fh):
    """
    :param Fh: Fourier transformed interferogram.
    :type Fh: ndarray
    :return: ROI you select, the local peak within that ROI
    :rtype: tuple, ndarray
    """

    Fh_img = format_holo(Fh)
    cv2.namedWindow('Masking', cv2.WINDOW_NORMAL)
    cv2.setWindowTitle('Masking', 'Select Off-axis Order of Interest')
    while True:
        ROI = cv2.selectROI('Masking', Fh_img, fromCenter=True)
        roi_mask = np.zeros(Fh_img.shape)
        roi_mask[int(ROI[1]):int(ROI[1] + ROI[3]), int(ROI[0]): int(ROI[0] + ROI[2])] = 1
        if ROI == (0, 0, 0, 0):
            cv2.destroyWindow('Masking')
            raise RuntimeError("ROI selection cancelled.")

        peaks = peak_local_max(Fh_img * roi_mask, min_distance=min([ROI[2], ROI[3]]))
        if len(peaks) == 1:
            break
        warn("Failed to identify a single peak. Try a different ROI.")

    f1 = np.array([(peaks[0][0]), (peaks[0][1])])
    cv2.destroyWindow('Masking')
    print(f'>>> Selected ROI centered at a tilt of ({f1[0]}, {f1[1]}).')
    return ROI, f1


def find_peak_in_roi(Fh, ROI):
    """
    Finds peak in already existing ROI.

    :param Fh: Fourier transformed interferogram.
    :type Fh: ndarray
    :param ROI: The region of interest where phase info is located (x, y, w, h).
    :type ROI: tuple
    :return: Location of peak in ROI.
    :rtype: ndarray
    """

    Fh_img = format_holo(Fh)
    while True:
        roi_mask = np.zeros(Fh_img.shape)
        roi_mask[int(ROI[1]):int(ROI[1] + ROI[3]), int(ROI[0]): int(ROI[0] + ROI[2])] = 1
        if ROI == (0, 0, 0, 0):
            raise RuntimeError("ROI selection cancelled.")

        peaks = peak_local_max(Fh_img * roi_mask, min_distance=min([ROI[2], ROI[3]]))
        if len(peaks) == 1:
            break
        warn("Failed to identify a single peak. Try a different ROI.")

    fp1 = np.array([(peaks[0][0]), (peaks[0][1])])
    return fp1


def off_axis_masks(Fh, fp1=None, kNA=-1):
    """
    Creates the necessary masks for off-axis spatial filtering.

    :param Fh: Fourier transformed interferogram.
    :type Fh: ndarray
    :param fp1: Location of off-axis peak. If None will be asked to select ROI.
    :type fp1: ndarray
    :param kNA: The numerical aperture of the system (in terms of pixels). This determines the radius of the mask.
    If not given, it will be assumed that you aligned system in the most optimal way given location of peak.
    :type kNA: float
    :return: Used radius, Mask at +1, mask at 0, mask at -1.
    :rtype: float, tuple<ndarray, ndarray, ndarray>
    """

    fp1 = np.array(fp1)
    if fp1 is None:
        _, fp1 = select_mask_roi(np.abs(Fh) ** (1 / 4))
    else:
        assert len(fp1) == 2, "Peak coordinates have the wrong dimensionality."
        assert fp1[0] < Fh.shape[0] and fp1[1] < Fh.shape[1], "Invalid peak position given."

    cy, cx = Fh.shape[0] / 2, Fh.shape[1] / 2
    fm1 = Fh.shape - fp1
    Fy, Fx = np.ogrid[:Fh.shape[0], :Fh.shape[1]]
    r = np.sqrt(np.sum(np.array([fp1[0] - cy, fp1[1] - cx]) ** 2))
    if kNA is None or kNA <= 0:
        r_k = r / 3
    else:
        assert kNA < r, "Radius of mask can not exceed distance from image center."
        r_k = kNA

    r_k = int(np.floor(r_k))
    mask10 = np.sqrt((Fx - fp1[1]) ** 2 + (Fy - fp1[0]) ** 2) <= r_k
    mask01 = np.sqrt((Fx - fm1[1]) ** 2 + (Fy - fm1[0]) ** 2) <= r_k
    mask00 = np.sqrt((Fx - cx) ** 2 + (Fy - cy) ** 2) <= r_k
    return r_k, mask00, mask10, mask01
