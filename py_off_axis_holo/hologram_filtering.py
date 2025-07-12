import numpy as np
import logging
import cv2

from scipy.optimize import minimize
from skimage.feature import peak_local_max
from py_off_axis_holo.discrete_transforms import DFT
from py_off_axis_holo.holography_helpers import ref_phase_shift, gridspace, format_img, crop_to_mask
from warnings import warn

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('hologram_filtering.log'),
                              logging.StreamHandler()])

def select_mask_roi(Fh, scale_input=False):
    if scale_input:
        Fh = np.abs(Fh)**(1/4)
    Fh_img = format_img(Fh)

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
    logging.info(f'Selected ROI centered at a tilt of ({f1[0]}, {f1[1]}).')
    return roi_mask, f1

def off_axis_masks(Fh, f10=None, mask_radius=None, scale_input=False):
    if f10 is None:
        _, f10 = select_mask_roi(Fh, scale_input)
    else:
        assert len(f10) == 2, "Peak coordinates have the wrong dimensionality."
        assert f10[0] < Fh.shape[0] and f10[1] < Fh.shape[1], "Invalid peak position given."

    cy, cx = Fh.shape[0] / 2, Fh.shape[1] / 2
    f01 = Fh.shape - f10
    Fy, Fx = np.ogrid[:Fh.shape[0], :Fh.shape[1]]
    r = np.floor(np.sqrt(np.sum(np.array([f10[0] - cy, f10[1] - cx]) ** 2)))
    if mask_radius is None:
        r_1 = r / 3
    else:
        assert mask_radius < r, "Radius of mask can not exceed distance from image center."
        r_1 = mask_radius
    r_1 = int(np.floor(r_1))
    mask10 = np.sqrt((Fx - f10[1]) ** 2 + (Fy - f10[0]) ** 2) <= r_1
    mask01 = np.sqrt((Fx - f01[1]) ** 2 + (Fy - f01[0]) ** 2) <= r_1
    mask00 = np.sqrt((Fx - cx) ** 2 + (Fy - cy) ** 2) <= r_1
    return mask10, mask00, mask01


class OffAxisFilter(DFT):

    def __init__(self, M, N, wl, sz, nb=0, threads=1, dtype='complex128', **kwargs):
        """
        *Using un-normalized FFTs is fine since we are only concerned with the phase information.
        """

        super().__init__((M, N), nb, threads=threads, dtype=dtype, **kwargs)
        self._dims = np.array([M, N])
        self._k0 = 2 * np.pi / wl
        self._sz = sz
        self._masks = None
        self._ref = None
        self._window = np.ones(self.input_shape)

    def __call__(self, fringes):
        if not self.masked:
            self.calibrate(fringes)

        if self._ref is None:
            Fh = self.forwards(fringes)
            Fh *= self._masks[0]
            Fh = crop_to_mask(Fh, self._masks[0])
        else:
            Fh = self.forwards(fringes * self._ref)
            Fh *= self._masks[1]
            Fh = crop_to_mask(Fh, self._masks[1])

        if tuple(self.output_shape) != tuple(Fh.shape):
            self._ifft.refactor(Fh.shape)
        return self.backwards(Fh)

    @property
    def masked(self):
        if self._masks is None:
            return False
        return True

    def _calc_tilt(self, f):
        """
        Calculates tilt from frequency coordinates. f is
        assumed to be relative to the center coordinates.

        :param f: Frequency coords of +/-1 order in the form [fx, fy]
        :type f: array
        :return: Tilt in the form [x_tilt, y_tilt]
        :rtype: ndarray
        """

        M, N = self._dims
        kx = 2 * np.pi * f[0] / (N * self._sz)
        ky = 2 * np.pi * f[1] / (M * self._sz)
        return - np.arcsin(np.array([kx, ky]) / self._k0)

    def _calc_freq(self, tilt):
        """
        Calculates frequency coordinates from tilt.

        :param tilt: Tilt in the form [x_tilt, y_tilt]
        :type tilt: array
        :return: Frequency coords of +/-1 order in the form [fx, fy] given relative to the center coordinates
        :rtype: ndarray
        """

        M, N = self._dims
        k = - self._k0 * np.sin(np.array(tilt))
        fx = N * self._sz * k[0] / (2 * np.pi)
        fy = M * self._sz * k[1] / (2 * np.pi)
        return np.array([fx, fy])

    def _min_tilt(self, f, field, X, Y):
        tilt = self._calc_tilt(f)
        R = ref_phase_shift((X, Y), tilt, self._k0, self._sz)

        f = R * field / np.abs(field)**2
        xvec = -1j * np.gradient(f, axis=0)
        yvec = -1j * np.gradient(f, axis=1)
        mag = np.absolute(xvec) ** 2 + np.absolute(yvec) ** 2
        return mag.sum()

    def _optimize_tilt(self, f, phase, method='TNC', tol=1e-6, step=None, opts=None):
        if not self.masked:
            raise RuntimeError("Must have masks calibrated to optimize tilt compensation.")

        logging.info("Aligning Mask to Inferred Tilt...")
        M, N = self._dims
        X, Y = gridspace(N, M, 0, True)

        Fh = self.forwards(phase) * self._masks[0]
        if tuple(self.output_shape) != tuple(Fh.shape):
            self._ifft.refactor(Fh.shape)
        field = self.backwards(Fh)

        if opts is None:
            opts = {}
        if type(step) is int:
            bounds = [(f[0] - step, f[0] + step), (f[1] - step, f[1] + step)]
        else:
            bounds = None
        res = minimize(self._min_tilt, f, args=(field, X, Y),
                       method=method, bounds=bounds, tol=tol, options=opts)
        logging.info("Optimization Routine Complete.")
        return np.array(res.x)

    def _visualize_roi(self, fringes):
        if not self._ref is None:
            fringes = fringes * self._ref
        Fh = self.forwards(fringes)
        Fh = format_img(np.abs(Fh)**(1/4))

        cv2.namedWindow('visualize_roi', cv2.WINDOW_NORMAL)
        cv2.imshow('visualize_roi', Fh)
        cv2.waitKey(1500)
        if self._ref is None:
            Fh *= self._masks[0].astype('uint8')
        else:
            Fh *= self._masks[1].astype('uint8')
        cv2.imshow('visualize_roi', Fh)
        cv2.waitKey(4000)
        if cv2.getWindowProperty('visualize_roi', cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow('visualize_roi')
        return True

    def forwards(self, a):
        a = np.array(a).astype('complex128', casting='safe')
        if self._nb > 0:
            a = self.pad_arr(a)
        a *= self._window
        return np.fft.fftshift(self._fft(a))

    def backwards(self, b):
        b = np.array(b).astype('complex128', casting='safe')
        a = self._ifft(np.fft.ifftshift(b))
        return self.unpad_arr(a) if self._nb > 0 else a

    def add_window(self, window_fcn, **kwargs):
        if callable(window_fcn):
            try:
                self._window = window_fcn(self.input_shape, **kwargs)
                return True
            except TypeError:
                warn("Invalid window function provided.")
                return False
        return False

    def construct_reference(self, f, M, N):
        tilt = self._calc_tilt(f)
        X, Y = gridspace(N, M, 0, True)
        return ref_phase_shift((X, Y), tilt, self._k0, self._sz)

    def calibrate(self, fringes, mask_radius=None, optimize=True, use_ref=True, visualize=False, **kwargs):

        self.reset()
        Fh = self.forwards(fringes)
        _, f10 = select_mask_roi(Fh, True)
        self._masks = off_axis_masks(Fh, f10=f10, mask_radius=mask_radius)
        freq = [f10[1] - Fh.shape[1] / 2, f10[0] - Fh.shape[0] / 2]

        if optimize:
            try:
                freq = self._optimize_tilt(freq, fringes, **kwargs)
            except RuntimeError:
                warn("Could not align to tilt.")

        if use_ref:
            self._ref = self.construct_reference(freq, self._dims[0], self._dims[1])
        if visualize:
            self._visualize_roi(fringes)
        return True

    def reset(self):
        self._masks = None
        self._ref = None
        self._window = np.ones(self.input_shape)
        self._ifft.refactor(self.input_shape)
        return self
