from abc import ABC
import numpy as np
from scipy.optimize import minimize
from py_off_axis_holo.discrete_transforms import DFT
from py_off_axis_holo.holo_utils import reference_wave, gridspace, format_img
from warnings import warn
import cv2

class OffAxisMask(ABC, DFT):

    def __init__(self, M, N, nb=1, threads=1, dtype='complex128', **kwargs):
        super().__init__((M, N), nb, threads=threads, dtype=dtype, ortho=False, **kwargs)
        self._Fx, self._Fy = gridspace(N, M, nb, True)
        self._mask = None

    @property
    def masked(self):
        if self._mask is None:
            return False
        else:
            return True

    @property
    def roi(self):
        if self._mask is None:
            return 0, 0, self._dims[0], self._dims[1]
        coords = np.argwhere(self._mask)
        m_min, n_min = coords.min(axis=0)
        m_max, n_max = coords.max(axis=0)
        return m_min, n_min, m_max - m_min + 1, n_max - n_min + 1

    def _calc_mask(self, f1, radius=None):
        """

        :param f1: Frequency coords of +/-1 order in the form [fx, fy]
        :return:
        """

        r = np.sqrt(np.sum(np.array(f1) ** 2))
        if radius is None:
            r_1 = r / 3
            r_DC = 2 * r_1
        else:
            assert radius < r, "Radius of mask can not exceed distance from image center."
            r_1 = radius
            r_DC = r - r_1

        mask10 = np.sqrt((self._Fx - f1[0]) ** 2 + (self._Fy - f1[1]) ** 2) < r_1
        mask01 = np.sqrt((self._Fx + f1[0]) ** 2 + (self._Fy + f1[1]) ** 2) < r_1
        mask00 = np.sqrt(self._Fx ** 2 + self._Fy ** 2) < r_DC
        return mask10, mask01, mask00

    def _find_max(self, Fh):
        """
        Finds maximum value of a masked Fourier spectrum

        :param Fh: Masked Fourier spectrum
        :type Fh: array
        :return: Indices of maximum in the form [fx, fy]
        :rtype: ndarray
        """

        fy, fx = np.where(Fh == np.amax(Fh))
        if len(fx) == 0:
            raise RuntimeError("No maximum identified.")
        if len(fx) > 1:
            warn("Multiple maximums detected. Using median value.")
            fx = np.median(fx)
            fy = np.median(fy)

        M, N = self.shape
        f = np.array([fx - N / 2, fy - M / 2])
        return f.reshape(f.size, ) / self._nb

    def _select_roi(self, Fh, auto=False):
        Fh_img = format_img(Fh)

        print('Select Order of Interest.')
        cv2.namedWindow('spfilter_roi_selector', cv2.WINDOW_NORMAL)
        ROI = cv2.selectROI('spfilter_roi_selector', Fh_img, fromCenter=True)

        roi_mask = np.zeros(self.shape)
        roi_mask[int(ROI[1]):int(ROI[1] + ROI[3]), int(ROI[0]): int(ROI[0] + ROI[2])] = 1

        M, N = self._dims
        fx = int(ROI[0] + ROI[2] // 2) - N / 2
        fy = int(ROI[1] + ROI[3] // 2) - M / 2
        f1 = np.array([fx, fy]) / self._nb

        if auto:
            try:
                f1 = self._find_max(Fh * roi_mask)
            except RuntimeError:
                warn("Failed to automatically find peak. Resorting to chosen center.")

        cv2.destroyWindow('spfilter_roi_selector')
        print(f'Selected ROI centered at a tilt of ({f1[0]}, {f1[1]}).')
        return roi_mask, f1

    def _crop_to_mask(self, a):
        if self._mask is None:
            raise RuntimeError("No stored mask found.")
        coords = np.argwhere(self._mask)
        m_min, n_min = coords.min(axis=0)
        m_max, n_max = coords.max(axis=0)
        return a[m_min:m_max + 1, n_min:n_max + 1]

    def alignment_mask(self, right=True):
        M, N = self._dims
        fx = (8 * N + M - np.sqrt(8 * (N ** 2 + M ** 2) + 2 * N * M)) / 14
        fy = (N + 8 * M - np.sqrt(8 * (N ** 2 + M ** 2) + 2 * N * M)) / 14

        sign = -1 if right else 1
        f1_opt = (fx, sign * fy)
        mask10, mask01, mask00 = self._calc_mask(f1_opt)
        return mask10 + mask01 + mask00, f1_opt


class OffAxisFilter(OffAxisMask):

    def __init__(self, M, N, wl, sz, nb=1, threads=1, dtype='complex128', **kwargs):
        """
        *Using un-normalized FFTs is fine since we are only concerned with the phase information.

        :param M:
        :param N:
        :param wl:
        :param sz:
        :param nb:
        :param threads:
        :param dtype:
        :param kwargs:
        """

        super().__init__(M, N, nb, threads=threads, dtype=dtype, **kwargs)
        self._k0 = 2 * np.pi / wl
        self._sz = sz
        self._crop = False
        self._ref = None
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
        R = reference_wave((X, Y), tilt, self._k0, self._sz)

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
        R = reference_wave((X, Y), tilt, self._k0, self._sz)

        f = R * np.exp(1j * phi)
        xvec = -1j * np.gradient(f, axis=0)
        yvec = -1j * np.gradient(f, axis=1)
        mag = np.absolute(xvec) ** 2 + np.absolute(yvec) ** 2
        return mag.sum()

    def _optimize_tilt(self, f1, phase, method='TNC', step=1, tol=1e-6, opts=None):
        print("Aligning Mask to Inferred Tilt...")
        M, N = self._dims
        X, Y = gridspace(N, M, 1, True)

        phase = self._crop_to_mask(phase)
        X = self._crop_to_mask(X)
        Y = self._crop_to_mask(Y)

        if opts is None:
            opts = {}
        bounds = [(f1[0]-step, f1[0]+step), (f1[1]-step, f1[1]+step)]
        res = minimize(self._min_ksqr, f1, args=(phase, X, Y),
                       method=method, bounds=bounds, tol=tol, options=opts)

        print("Optimization Routine Complete.")
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
        return True

    def forwards(self, a):
        a = self.pad_arr(a * self._window) if self._nb > 1 else a * self._window
        return self._fft(a)

    def backwards(self, b):
        if self._crop and not self._mask is None:
            b = self._crop_to_mask(b)
        a = self._ifft(b)
        return self.unpad_arr(a) if self._nb > 1 else a

    def add_window(self, window_fcn, **kwargs):
        if callable(window_fcn):
            try:
                self._window = window_fcn(self._dims, **kwargs)
                return True
            except TypeError:
                warn("Invalid window function provided.")
                return False
        return False

    def construct_reference(self, f1, M, N):
        tilt = self._calc_tilt(f1)
        X, Y = gridspace(N, M, 1, True)
        return reference_wave((X, Y), tilt, self._k0, self._sz)

    def align_to_freq(self, f1, tilt_compensate_in='FP'):
        self.reset()
        if tilt_compensate_in == 'IP':
            self._set_tilt_compensation(f1)
        else:
            self._mask, _, _ = self._calc_mask(f1)

        if self._crop and not self._mask is None:
            self._ifft.refactor(self.roi)
        return True

    def calibrate(self, fringes, radius=None, tilt_compensate_in='FP', auto=True, optimize=False, crop=False,
                  visualize=False, **kwargs):

        if tilt_compensate_in.upper() != 'IP' and tilt_compensate_in.upper() != 'FP':
            warn("Defaulting to compensating for tilt in Fourier plane.")
            tilt_compensate_in = 'FP'
        if not auto and tilt_compensate_in == 'IP':
            warn("To use a manually selected mask, tilt compensation must be done in Fourier plane.")
            tilt_compensate_in = 'FP'
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

        if tilt_compensate_in.upper() == 'IP':
            self._set_tilt_compensation(f1, radius)

        self._crop = crop
        if self._crop and not self._mask is None:
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
