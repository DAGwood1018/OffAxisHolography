from abc import ABC
import numpy as np
import logging
import cv2

from scipy.optimize import minimize
from skimage.feature import peak_local_max
from py_off_axis_holo.discrete_transforms import DFT
from py_off_axis_holo.holography_helpers import ref_phase_shift, gridspace, format_img
from warnings import warn

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('hologram_filtering.log'),
                              logging.StreamHandler()])

class OffAxisMasking:

    def __init__(self):
        self._masks = None

    def __call__(self, Fh, scale_input=False, mask_radius=None):
        if scale_input:
            Fh = np.abs(Fh)**(1/4)
        _, f1 = self.select_roi(Fh)

        r = np.sqrt(np.sum(np.array(f1) ** 2))
        if mask_radius is None:
            r_1 = r / 3
            r_DC = 2 * r_1
        else:
            assert mask_radius < r, "Radius of mask can not exceed distance from image center."
            r_1 = mask_radius
            r_DC = r - r_1

        M, N = np.meshgrid(Fh.shape[0], Fh.shape[1], indexing='xy')
        mask10 = np.sqrt((M - f1[0]) ** 2 + (N - f1[1]) ** 2) < r_1
        mask01 = np.sqrt((M + f1[0]) ** 2 + (N + f1[1]) ** 2) < r_1
        mask00 = np.sqrt(M ** 2 + N ** 2) < r_DC
        self._masks = mask10, mask01, mask00
        return self.masks

    @property
    def masks(self):
        return self._masks

    @classmethod
    def select_roi(cls, Fh):
        Fh_img = format_img(Fh)

        cv2.namedWindow('Masking', cv2.WINDOW_NORMAL)
        cv2.setWindowTitle('Masking', 'Select Off-axis Order of Interest')
        ROI = cv2.selectROI('Masking', Fh_img, fromCenter=True)

        roi_mask = np.zeros(Fh_img.shape)
        roi_mask[int(ROI[1]):int(ROI[1] + ROI[3]), int(ROI[0]): int(ROI[0] + ROI[2])] = 1

        peaks = peak_local_max(Fh_img * roi_mask, min_distance=min([ROI[2], ROI[3]]))
        assert len(peaks) == 1, "Failed to identify a single peak. Try a different ROI."
        f1 = peaks[0]

        cv2.destroyWindow('Masking')
        logging.info(f'Selected ROI centered at a tilt of ({f1[0]}, {f1[1]}).')
        return roi_mask, f1

    @classmethod
    def crop_to_mask(cls, a, mask):
        coords = np.argwhere(mask)
        m_min, n_min = coords.min(axis=0)
        m_max, n_max = coords.max(axis=0)
        return a[m_min:m_max + 1, n_min:n_max + 1]


class OffAxisFilter(DFT):

    def __init__(self, M, N, wl, sz, nb=0, threads=1, dtype='complex128', **kwargs):
        """
        *Using un-normalized FFTs is fine since we are only concerned with the phase information.
        """

        super().__init__(M, N, nb, threads=threads, dtype=dtype, **kwargs)
        self._k0 = 2 * np.pi / wl
        self._sz = sz
        self._crop = False
        self._ref = None
        self._filter_radius = 0
        self._window = np.ones(self._dims)

    def __call__(self, fringes):
        if not self.masked:
            self.calibrate(fringes)

        Fh = self.forwards(fringes) if self._ref is None else self.forwards(fringes * self._ref)
        Fh *= self._mask
        if not self._crop and self._ref is None:
            for k in range(Fh.ndim):
                nonempty = np.nonzero(np.any(Fh, axis=1 - k))[0]
                i1, i2 = nonempty.min(), nonempty.max()
                shift = (Fh.shape[k] - i1 - i2) // 2
                Fh = np.roll(Fh, shift, axis=k)
        return self.backwards(Fh)

    def _calc_tilt(self, f1):
        """
        Calculates tilt from frequency coordinates.

        :param f1: Frequency coords of +/-1 order in the form [fx, fy]
        :type f1: array
        :return: Tilt in the form [x_tilt, y_tilt]
        :rtype: ndarray
        """

        M, N = self._dims
        kx = 2 * np.pi * f1[0] / (N * self._sz)
        ky = 2 * np.pi * f1[1] / (M * self._sz)
        return - np.arcsin(np.array([kx, ky]) / self._k0)

    def _calc_freq(self, tilt):
        """
        Calculates frequency coordinates from tilt.

        :param tilt: Tilt in the form [x_tilt, y_tilt]
        :type tilt: array
        :return: Frequency coords of +/-1 order in the form [fx, fy]
        :rtype: ndarray
        """

        M, N = self._dims
        k = - self._k0 * np.sin(np.array(tilt))
        fx = N * self._sz * k[0] / (2 * np.pi)
        fy = M * self._sz * k[1] / (2 * np.pi)
        return np.array([fx, fy])

    def _visualize_roi(self, fringes):
        if not self._ref is None:
            fringes = fringes * self._ref
        Fh = self.forwards(fringes)

        Fh_scaled = 2 * np.log(np.abs(Fh) + 1e-10)
        Fh_img = format_img(Fh_scaled)
        cv2.namedWindow('visualize_roi', cv2.WINDOW_NORMAL)

        cv2.imshow('visualize_roi', Fh_img)
        cv2.waitKey(1500)
        Fh_img *= self._mask.astype('uint8')
        cv2.imshow('visualize_roi', Fh_img)
        cv2.waitKey(4000)
        if cv2.getWindowProperty('visualize_roi', cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow('visualize_roi')
        return True

    def _min_k(self, f1, phi, X, Y):
        tilt = self._calc_tilt(f1)
        R = ref_phase_shift((X, Y), tilt, self._k0, self._sz)

        f = R * np.exp(1j * phi)
        xvec = -1j * np.gradient(f, axis=0)
        yvec = -1j * np.gradient(f, axis=1)
        xvec *= np.conjugate(f)
        yvec *= np.conjugate(f)

        x = xvec.sum()
        y = yvec.sum()
        return np.absolute(x) ** 2 + np.absolute(y) ** 2

    def _min_ksqr(self, f1, phi, X, Y):
        tilt = self._calc_tilt(f1)
        R = ref_phase_shift((X, Y), tilt, self._k0, self._sz)

        f = R * np.exp(1j * phi)
        xvec = -1j * np.gradient(f, axis=0)
        yvec = -1j * np.gradient(f, axis=1)
        mag = np.absolute(xvec) ** 2 + np.absolute(yvec) ** 2
        return mag.sum()

    def _optimize_tilt(self, f1, phase, method='TNC', tol=1e-6, step=10, opts=None):
        logging.info("Aligning Mask to Inferred Tilt...")
        M, N = self._dims
        X, Y = gridspace(N, M, 0, True)

        if self.masked:
            phase = self.crop_to_mask(phase, self._mask)
            X = self.crop_to_mask(X, self._mask)
            Y = self.crop_to_mask(Y, self._mask)

        if opts is None:
            opts = {}
        bounds = [(f1[0]-step, f1[0]+step), (f1[1]-step, f1[1]+step)]
        res = minimize(self._min_ksqr, f1, args=(phase, X, Y),
                       method=method, bounds=bounds, tol=tol, options=opts)
        logging.info("Optimization Routine Complete.")
        return np.array(res.x)

    def _set_tilt_compensation(self, f1, radius=None):
        M, N = self._dims
        r = np.sqrt(np.sum(np.array(f1) ** 2))
        if radius is None:
            r_1 = r / 3
        else:
            assert radius < r, "Radius of mask can not exceed distance from image center."
            r_1 = radius

        self._mask = np.sqrt(self._Fx ** 2 + self._Fy ** 2) < r_1
        self._ref = self.construct_reference(f1, M, N)
        self._filter_radius = int(np.floor(r_1))
        return True

    def _crop_filter(self, b):
        m, n = b.shape
        cy, cx = m / 2, n / 2
        y_min = max(int(np.floor(cy - self._filter_radius)), 0)
        y_max = min(int(np.floor(cy + self._filter_radius)), m)
        x_min = max(int(np.floor(cx - self._filter_radius)), 0)
        x_max = min(int(np.floor(cx + self._filter_radius)), n)
        return b[y_min:y_max, x_min:x_max]

    def forwards(self, a):
        a = a * self._window
        return np.fft.fftshift(self._fft(a))

    def backwards(self, b):
        if self._crop and self._filter_radius:
            b = self._crop_filter(b)
        a = self._ifft(np.fft.ifftshift(b))
        return self.unpad_arr(a) if self._nb > 0 else a

    def add_window(self, window_fcn, **kwargs):
        if callable(window_fcn):
            try:
                self._window = window_fcn(self.shape, **kwargs)
                return True
            except TypeError:
                warn("Invalid window function provided.")
                return False
        return False

    def construct_reference(self, f1, M, N):
        tilt = self._calc_tilt(f1)
        X, Y = gridspace(N, M, 0, True)
        return ref_phase_shift((X, Y), tilt, self._k0, self._sz)

    def calibrate(self, fringes, radius=None, auto=True, optimize=False, crop=False,
                  visualize=False, **kwargs):

        if not auto and optimize:
            warn("No optimization can be done if the manually selected mask is to be used.")
            optimize = False

        self.reset()
        Fh = self.forwards(fringes)
        Fh_scaled = 2 * np.log(np.abs(Fh) + 1e-10)
        roi_mask, f1 = self._select_roi(Fh_scaled, auto)

        if auto:
            self._mask, _, _ = self._calc_mask(f1, radius)
        else:
            self._mask = roi_mask

        if optimize:
            try:
                Fh = self.forwards(fringes) * self._mask
                phi = np.angle(self.backwards(Fh))
                f1 = self._optimize_tilt(f1, phi, **kwargs)
                self._mask, _, _ = self._calc_mask(f1, radius)
            except RuntimeError:
                warn("Could not align to tilt.")

        self._crop = crop
        self._set_tilt_compensation(f1, radius)
        if self._crop and self._filter_radius > 0:
            self._ifft.refactor((self.roi[2], self.roi[3]))

        if visualize:
            self._visualize_roi(fringes)
        return f1

    def reset(self):
        self._mask = None
        self._crop = False
        self._ref = None
        self._window = np.ones(self._dims)
        self._ifft.refactor(self.shape)
        return self
