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
                    handlers=[logging.StreamHandler()])

def select_mask_roi(Fh):
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

def off_axis_masks(Fh, f10=None, mask_radius=None, sqr_masks=False):
    f10 = np.array(f10)
    if f10 is None:
        _, f10 = select_mask_roi(np.abs(Fh)**(1/4))
    else:
        assert len(f10) == 2, "Peak coordinates have the wrong dimensionality."
        assert f10[0] < Fh.shape[0] and f10[1] < Fh.shape[1], "Invalid peak position given."

    cy, cx = Fh.shape[0] / 2, Fh.shape[1] / 2
    f01 = Fh.shape - f10
    Fy, Fx = np.ogrid[:Fh.shape[0], :Fh.shape[1]]
    r = np.sqrt(np.sum(np.array([f10[0] - cy, f10[1] - cx]) ** 2))
    if mask_radius is None:
        r_1 = r / 3
    else:
        assert mask_radius < r, "Radius of mask can not exceed distance from image center."
        r_1 = mask_radius

    r_1 = np.round(r_1)
    if sqr_masks:
        mask10 = r_1 <= (Fx - f10[1]) <= r_1 <= (Fy - f10[0]) <= r_1
        mask01 = r_1 <= (Fx - f01[1]) <= r_1 <= (Fy - f01[0]) <= r_1
        mask00 = r_1 <= (Fx - cx) <= r_1 <= (Fy - cy) <= r_1
    else:
        mask10 = np.sqrt((Fx - f10[1]) ** 2 + (Fy - f10[0]) ** 2) <= r_1
        mask01 = np.sqrt((Fx - f01[1]) ** 2 + (Fy - f01[0]) ** 2) <= r_1
        mask00 = np.sqrt((Fx - cx) ** 2 + (Fy - cy) ** 2) <= r_1
    return r_1, (mask10, mask00, mask01)


class OffAxisFilter(DFT):

    def __init__(self, M, N, wl, sz, nb=0, threads=1, dtype='complex128', sqr_masks=False, **kwargs):
        """
        *Using un-normalized FFTs is fine since we are only concerned with the phase information.
        """

        super().__init__((M, N), nb, threads=threads, dtype=dtype, **kwargs)
        self._dims = np.array([M, N])
        self._k0 = 2 * np.pi / wl
        self._sz = sz

        self._masks = None
        self._ref_wave = None
        self._filter_radius = None
        self._filter_center = None
        self._sqr_masks = sqr_masks
        self._window = np.ones(self._dims)

    def __call__(self, fringes, only_crop_to_mask=False, self_RCH=False, RCH_r_factor=0.1):
        if not self.masked:
            self.calibrate(fringes)

        if only_crop_to_mask is None:
            Fh = self.forwards(fringes)
            if self_RCH:
                RCH = Fh.copy()
                _, RCH_masks = off_axis_masks(np.zeros(self._dims), f10=self._filter_center,
                                              mask_radius=np.round(0.1*self._filter_radius), sqr_masks=self._sqr_masks)
                RCH *= np.tile(RCH_masks[0], self.input_shape[-1]) if self._stacked else RCH_masks[0]
                RCH = self.crop_to_mask(RCH, self._masks[0])
            Fh *= np.tile(self._masks[0], self.input_shape[-1]) if self._stacked else self._masks[0]
            Fh = self.crop_to_mask(Fh, self._masks[0])
        else:
            Fh = self.forwards(fringes * self._ref_wave)
            if self_RCH:
                RCH = Fh.copy()
                _, RCH_masks = off_axis_masks(np.zeros(self._dims), f10=self._filter_center,
                                              mask_radius=np.round(0.1*self._filter_radius), sqr_masks=self._sqr_masks)
                RCH *= np.tile(RCH_masks[1], self.input_shape[-1]) if self._stacked else RCH_masks[1]
                RCH = self.crop_to_mask(RCH, self._masks[1])
            Fh *= np.tile(self._masks[1], self.input_shape[-1]) if self._stacked else self._masks[1]
            Fh = self.crop_to_mask(Fh, self._masks[1])

        if tuple(self.output_shape) != tuple(Fh.shape):
            self._ifft.refactor(Fh.shape, axes={0, 1})
        field = self.backwards(Fh)
        if self_RCH:
            RCH_field = self.backwards(RCH)
            field *= np.exp(-1j*np.angle(RCH_field))
        return field

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

    def _min_k(self, freq, phi, X, Y):
        tilt = self._calc_tilt(freq)
        R = ref_phase_shift((X, Y), tilt, self._k0, self._sz)

        f = R * np.exp(1j * phi)
        xvec = -1j * np.gradient(f, axis=0)
        yvec = -1j * np.gradient(f, axis=1)
        xvec *= np.conjugate(f)
        yvec *= np.conjugate(f)

        x = xvec.sum()
        y = yvec.sum()
        return np.absolute(x) ** 2 + np.absolute(y) ** 2

    def _min_ksqr(self, freq, phi, X, Y):
        tilt = self._calc_tilt(freq)
        R = ref_phase_shift((X, Y), tilt, self._k0, self._sz)

        f = R * np.exp(1j * phi)
        xvec = -1j * np.gradient(f, axis=0)
        yvec = -1j * np.gradient(f, axis=1)
        mag = np.absolute(xvec) ** 2 + np.absolute(yvec) ** 2
        return mag.sum()

    def _optimize_tilt(self, freq, phase, method='TNC', cost='kqr', tol=1e-8, step=None, opts=None):
        logging.info("Aligning Mask to Inferred Tilt...")
        M, N = self._dims
        X, Y = gridspace(N, M, 0, True)

        if self.masked:
            phase = crop_to_mask(phase, self._masks[0])
            X = crop_to_mask(X, self._masks[0])
            Y = crop_to_mask(Y, self._masks[0])

        if opts is None:
            opts = {}
        if type(step) is int:
            bounds = [(freq[0] - step, freq[0] + step), (freq[1] - step, freq[1] + step)]
        else:
            bounds = None
        if cost == 'kqr':
            cost_func = self._min_ksqr
        elif cost == 'k':
            cost_func = self._min_k
        else:
            warn('Invalid cost function method given.')
            cost_func = self._min_ksqr
        res = minimize(cost_func, freq, args=(phase, X, Y),
                       method=method, bounds=bounds, tol=tol, options=opts)
        logging.info("Optimization Routine Complete.")
        return np.array(res.x)

    def crop_to_mask(self, a, mask):
        coords = np.argwhere(mask)
        m_min, n_min = coords.min(axis=0)
        m_max, n_max = coords.max(axis=0)
        return a[m_min:m_max + 1, n_min:n_max + 1, :] if self._stacked else a[m_min:m_max + 1, n_min:n_max + 1]

    def _visualize_roi(self, fringes):
        if not self._ref_wave is None:
            fringes = fringes * self._ref_wave
        Fh = self.forwards(fringes)
        Fh = format_img(np.abs(Fh)**(1/4))

        cv2.namedWindow('visualize_roi', cv2.WINDOW_NORMAL)
        cv2.imshow('visualize_roi', Fh)
        cv2.waitKey(1500)
        Fh *= self._masks[1].astype('uint8')
        cv2.imshow('visualize_roi', Fh)
        cv2.waitKey(4000)
        if cv2.getWindowProperty('visualize_roi', cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow('visualize_roi')
        return True

    def forwards(self, a):
        window = np.tile(self._window, self.input_shape[:-1]) if self._stacked else self._window
        return super().forwards(a * window)

    def add_window(self, window_fcn, **kwargs):
        if callable(window_fcn):
            try:
                self._window = window_fcn(self.input_shape[:-1], **kwargs) \
                    if self._stacked else window_fcn(self.input_shape, **kwargs)
                return True
            except TypeError:
                warn("Invalid window function provided.")
                return False
        return False

    def construct_reference(self, f, M, N):
        tilt = self._calc_tilt(f)
        X, Y = gridspace(N, M, 0, True)
        return ref_phase_shift((X, Y), tilt, self._k0, self._sz)

    def calibrate(self, fringes, mask_radius=None, optimize=True, visualize=False, **kwargs):
        """
        :param fringes: Hologram (typically of background) to calibrate phase filtering to.
        :param mask_radius: Override the automatic radius of the mask if not None.
        :param optimize: Optimize the mask center according to some cost function.
        :param visualize: Show image of mask.
        :param kwargs: Optional arguments to optimization routine.
        :return:
        """

        self.reset()
        Fh = self.forwards(fringes)
        _, f10 = select_mask_roi(np.abs(Fh)**(1/4))
        self._filter_radius, self._masks = off_axis_masks(Fh, f10=f10, mask_radius=mask_radius, sqr_masks=self._sqr_masks)
        freq = [f10[1] - Fh.shape[1] / 2, f10[0] - Fh.shape[0] / 2]

        if optimize:
            try:
                Fh = self.forwards(fringes) * self._masks[0]
                phi = np.angle(self.backwards(Fh))
                freq = self._optimize_tilt(freq, phi, **kwargs)
            except RuntimeError:
                warn("Could not align to tilt.")

        self._filter_center = [freq[1] + Fh.shape[1] / 2, freq[0] + Fh.shape[0] / 2]
        self._ref_wave = self.construct_reference(freq, self._dims[0], self._dims[1])
        if visualize:
            self._visualize_roi(fringes)
        return True

    def reset(self):
        self.unstack_arrays()
        self._masks = None
        self._ref_wave = None
        self._window = np.ones(self._dims)
        self._ifft.refactor(self.input_shape)
        return self

