from __future__ import annotations

import os

import cv2
import numba as nb
import numpy as np
import scipy.fft
from scipy import signal
from scipy.ndimage import binary_dilation
from skimage.feature import peak_local_max
from warnings import warn


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


def obj_to_cam_z(z_obj, mag=1.0):
    """Longitudinal magnification: dz_cam = mag**2 * dz_obj."""
    return z_obj * mag ** 2


def cam_to_obj_z(z_cam, mag=1.0):
    """Inverse of obj_to_cam_z."""
    return z_cam / mag ** 2


def view_filter(Fh, mask):
    """
    Visualize the calibrated filter applied to given interferogram.

    :param Fh: Fourier transformed interferogram.
    :type Fh: ndarray
    :param mask: The spatial filter to apply in Fourier plane.
    :type mask: ndarray
    :return: True if window is destroyed correctly.
    :rtype: bool
    """

    Fh_img = format_holo(np.abs(Fh) ** (1 / 4))

    cv2.namedWindow('visualize_roi', cv2.WINDOW_NORMAL)
    cv2.imshow('visualize_roi', Fh_img)

    cv2.waitKey(1500)
    Fh_img *= mask.astype('uint8')
    cv2.imshow('visualize_roi', Fh_img)
    cv2.waitKey(4000)

    if cv2.getWindowProperty('visualize_roi', cv2.WND_PROP_VISIBLE) >= 1:
        cv2.destroyWindow('visualize_roi')
        return True
    return False


def select_mask_roi(Fh):
    """
    :param Fh: Fourier transformed interferogram.
    :type Fh: ndarray
    :return: ROI you select, the local peak within that ROI
    :rtype: tuple, ndarray
    """

    Fh_img = format_holo(np.abs(Fh)**(1/4))
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

    Fh_img = format_holo(np.abs(Fh)**(1/4))
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
    :return: Used radius, Mask at 0, mask at +1, mask at -1.
    :rtype: float, tuple<ndarray, ndarray, ndarray>
    """

    if fp1 is None:
        _, fp1 = select_mask_roi(Fh)
    else:
        fp1 = np.array(fp1)
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


def full_bandpass(shape, fp1, nb=0, kNA=-1):
    """
    Fourier support of a raw off-axis hologram: DC disk (radius 2*kNA) plus the
    +1 and -1 side-band disks (radius kNA), in fftshift order on the raw
    (unpadded, possibly non-square) grid.

    fp1 and kNA are given on the square, padded grid used by OffAxisFilter
    (side max(shape) + 2*nb), i.e. the same numbers passed to set_calibration().

    :param shape: Shape of the raw hologram (M, N).
    :type shape: tuple<int>
    :param fp1: Location of the +1 order on the padded grid.
    :type fp1: tuple<int>
    :param nb: Padding used by OffAxisFilter. Default is 0.
    :type nb: int
    :param kNA: Side-band radius on the padded grid. If <= 0, uses the same
        default as off_axis_masks (distance of fp1 from the centre / 3).
    :type kNA: float
    :return: Boolean bandpass filter.
    :rtype: ndarray<bool>
    """

    ny, nx = shape
    n_pad = max(ny, nx) + 2 * nb
    c = n_pad / 2
    if kNA is None or kNA <= 0:
        kNA_pad = int(np.floor(np.hypot(fp1[0] - c, fp1[1] - c) / 3))
    else:
        kNA_pad = kNA

    sy, sx = (fp1[0] - c) / n_pad, (fp1[1] - c) / n_pad
    rk = kNA_pad / n_pad
    fy = (np.arange(ny) - ny // 2) / ny
    fx = (np.arange(nx) - nx // 2) / nx
    FY, FX = np.meshgrid(fy, fx, indexing='ij')
    m = np.hypot(FY - sy, FX - sx) <= rk
    m |= np.hypot(FY + sy, FX + sx) <= rk
    m |= np.hypot(FY, FX) <= 2.0 * rk
    return m


def desaturate(holo, bp_filter, sat_level=None, n_iter=300,
               ndilate=1, tol=1e-4, workers=-1):
    """
    Fill saturated (clipped) pixels of a hologram by band-limited
    extrapolation (Gerchberg-Papoulis with an inequality constraint).

    Alternates between
      * projecting onto the known Fourier support of the hologram, and
      * restoring every unsaturated pixel to its measured value while
        forcing saturated pixels to stay >= sat_level.

    Speed: real-to-complex FFTs (scipy.fft, multithreaded), numba kernels for
    the per-pixel steps, and the image is padded to an FFT-friendly size
    (e.g. 964 = 4*241 is a slow length). Padded pixels are treated as unknown,
    so they are filled band-limited instead of imposing zeros or wrap-around.

    :param holo: raw hologram (2-D).
    :type holo: ndarray
    :param bp_filter: boolean Fourier bandpass filter (fftshift order) on the raw grid,
        e.g. from full_bandpass. Must be point-symmetric about DC.
    :type bp_filter: ndarray<bool>
    :param sat_level: clipping level; default = holo.max().
    :type sat_level: float
    :param n_iter: maximum number of iterations. Default is 300.
    :type n_iter: int
    :param ndilate: dilate the saturated set by this many pixels (pixels next to
        a clipped one are often also non-linear). Default is 1.
    :type ndilate: int
    :param tol: stop when the relative change of the replaced pixels < tol. Default is 1e-4.
    :type tol: float
    :param check_every: test convergence every this many iterations. Default is 5.
    :type check_every: int
    :param workers: FFT threads (-1 = all cores). Default is -1.
    :type workers: int
    :return: repaired float hologram, boolean mask of replaced pixels.
    :rtype: ndarray, ndarray<bool>
    """

    @nb.njit(cache=True, parallel=True)
    def _halve_spectrum(X, half_bp):
        """Zero the half spectrum outside the support (in place)."""
        ny, nh = X.shape
        for i in nb.prange(ny):
            for j in range(nh):
                if not half_bp[i, j]:
                    X[i, j] = 0.0

    @nb.njit(cache=True, parallel=True)
    def _desat_constraints(x, y, holo, state, sat_level):
        """
        Data-consistency step, in place on x. Returns (sum d^2, sum x^2) over the
        replaced pixels for the convergence test.
        state: 0 = measured (reset to holo), 1 = clipped (kept >= sat_level),
               2 = free (dilation ring or FFT padding).
        """

        ny, nx = x.shape
        num = np.zeros(ny)
        den = np.zeros(ny)
        for i in nb.prange(ny):
            for j in range(nx):
                s = state[i, j]
                if s == 0:
                    x[i, j] = holo[i, j]
                else:
                    v = y[i, j]
                    if s == 1 and v < sat_level:
                        v = sat_level
                    d = v - x[i, j]
                    num[i] += d * d
                    den[i] += x[i, j] * x[i, j]
                    x[i, j] = v
        return num.sum(), den.sum()

    def _resample_mask(mask_shifted, shape):
        """
        Nearest-neighbour resample of an fftshift-ed frequency mask to a new grid
        (frequencies in cycles/pixel are preserved).
        """

        ny0, nx0 = mask_shifted.shape
        ny, nx = shape
        iy = np.clip(np.round((np.arange(ny) - ny // 2) / ny * ny0).astype(int) + ny0 // 2, 0, ny0 - 1)
        ix = np.clip(np.round((np.arange(nx) - nx // 2) / nx * nx0).astype(int) + nx0 // 2, 0, nx0 - 1)
        return mask_shifted[np.ix_(iy, ix)]

    holo = np.ascontiguousarray(holo, dtype=np.float64)
    if sat_level is None:
        sat_level = holo.max()
    sat = holo >= sat_level
    if not sat.any():
        return holo.copy(), sat
    sat_g = binary_dilation(sat, iterations=ndilate) if ndilate else sat.copy()

    ny0, nx0 = holo.shape
    ny = scipy.fft.next_fast_len(ny0, real=True)
    nx = scipy.fft.next_fast_len(nx0, real=True)
    sup = bp_filter if (ny, nx) == (ny0, nx0) else _resample_mask(bp_filter, (ny, nx))
    sup_half = np.ascontiguousarray(np.fft.ifftshift(sup)[:, :nx // 2 + 1])

    # pixel states on the (possibly padded) grid
    state = np.full((ny, nx), 2, dtype=np.uint8)            # padding: free
    st = np.zeros((ny0, nx0), dtype=np.uint8)
    st[sat_g] = 2
    st[sat] = 1
    state[:ny0, :nx0] = st
    ref = np.zeros((ny, nx))
    ref[:ny0, :nx0] = holo
    x = ref.copy()
    x[ny0:, :] = holo.mean()                                 # neutral start for padding
    x[:, nx0:] = holo.mean()

    workers = (os.cpu_count() or 1) if (workers is None or workers < 1) else int(workers)
    for it in range(1, n_iter + 1):
        X = scipy.fft.rfft2(x, workers=workers)
        _halve_spectrum(X, sup_half)
        y = scipy.fft.irfft2(X, s=(ny, nx), workers=workers)
        num, den = _desat_constraints(x, y, ref, state, float(sat_level))

        change = np.sqrt(num / (den + 1e-12))
        if change < tol:
            break
    return x[:ny0, :nx0].copy(), sat_g


