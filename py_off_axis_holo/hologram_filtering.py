import numpy as np
import cv2

from scipy.optimize import minimize
from skimage.feature import peak_local_max
from py_off_axis_holo.discrete_transforms import DFT, DiscreteTransform
from py_off_axis_holo.holography_helpers import ref_phase_shift, gridspace, format_holo, crop_to_mask, unpad_arr
from warnings import warn


def select_mask_roi(Fh):
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

    f1 = np.array([(peaks[0][0]), (peaks[0][1])])
    return f1

def off_axis_masks(Fh, f10=None, mask_radius=None):
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

    r_1 = int(np.floor(r_1))
    mask10 = np.sqrt((Fx - f10[1]) ** 2 + (Fy - f10[0]) ** 2) <= r_1
    mask01 = np.sqrt((Fx - f01[1]) ** 2 + (Fy - f01[0]) ** 2) <= r_1
    mask00 = np.sqrt((Fx - cx) ** 2 + (Fy - cy) ** 2) <= r_1
    return r_1, (mask10, mask00, mask01)


class OffAxisFilter(DFT):

    def __init__(self, M, N, wl, sz, nb=0, threads=1, dtype='complex128', **kwargs):
        """
        *Using un-normalized FFTs is fine since we are only concerned with the phase information.
        """

        # Ensure a 1:1 aspect ratio
        assert type(M) is int and type(N) is int, "Shape of array must be given in ints."
        pad_m, pad_n = 0, 0
        if M > N:
            pad_n = M-N
        elif N > M:
            pad_m = N-M
        M += pad_m
        N += pad_n

        super().__init__((M, N), nb, threads=threads, dtype=dtype, **kwargs)
        self._dims = np.array([M, N])
        self._k0 = 2 * np.pi / wl
        self._sz = sz

        self._masks = None
        self._ref_wave = None
        self._filter_radius = None
        self._window = None
        self._sqr_padding = (pad_m, pad_n)

    def __call__(self, fringes, only_crop_to_mask=False):
        """
        :param fringes: Interferogram corresponding to an off-axis hologram
        :type fringes: ndarray
        :param only_crop_to_mask: If True the field is not multiplied by a reference wave.
        :return: The +/-1 (off-axis) term of the hologram
        :rtype: ndarray
        """

        if not self.masked:
            self.calibrate(fringes)

        a = self.fix_aspect_ratio(fringes)
        if only_crop_to_mask is None:
            Fh = self.forwards(a)
            Fh *= np.stack([self._masks[0] for _ in range(self.input_shape[0])]) if self._stacked else self._masks[0]
            Fh = self.crop_to_mask(Fh, self._masks[0])
        else:
            Fh = self.forwards(a * self._ref_wave)
            Fh *= np.stack([self._masks[1] for _ in range(self.input_shape[0])]) if self._stacked else self._masks[1]
            Fh = self.crop_to_mask(Fh, self._masks[1])

        field = self.backwards(Fh)
        return field

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
        print(">>> Aligning Mask to Inferred Tilt...")
        M, N = self._dims
        X, Y = gridspace(M, N, 0, True)

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
        print(">>> Optimization Routine Complete.")
        return np.array(res.x)

    def _visualize_roi(self, fringes):
        a = self.fix_aspect_ratio(fringes)
        if not self._ref_wave is None:
            a = a*self._ref_wave
        Fh = self.forwards(a)
        Fh = format_holo(np.abs(Fh) ** (1 / 4))

        cv2.namedWindow('visualize_roi', cv2.WINDOW_NORMAL)
        cv2.imshow('visualize_roi', Fh)

        cv2.waitKey(1500)
        Fh *= self._masks[1].astype('uint8')
        cv2.imshow('visualize_roi', Fh)
        cv2.waitKey(4000)

        if cv2.getWindowProperty('visualize_roi', cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow('visualize_roi')
        return True

    @property
    def masked(self):
        if self._masks is None:
            return False
        return True

    def filter_dc(self, fringes):
        """
        :param fringes: Interferogram corresponding to an off-axis hologram
        :type fringes: ndarray
        :return: The DC term of the hologram
        :rtype: ndarray
        """

        if not self.masked:
            self.calibrate(fringes)

        a = self.fix_aspect_ratio(fringes)
        Fh = self.forwards(a)
        Fh *= np.stack([self._masks[1] for _ in range(self.input_shape[0])]) if self._stacked else self._masks[1]
        Fh = self.crop_to_mask(Fh, self._masks[1])
        return self.backwards(Fh)

    def crop_to_mask(self, a, mask):
        coords = np.argwhere(mask)
        m_min, n_min = coords.min(axis=0)
        m_max, n_max = coords.max(axis=0)
        return a[:, m_min:m_max, n_min:n_max] if self._stacked else a[m_min:m_max, n_min:n_max]

    def fix_aspect_ratio(self, a):
        pad_m, pad_n = self._sqr_padding
        if pad_m == pad_n:
            return a

        padding = ((0, 0),(0, pad_m),(0, pad_n)) if self._stacked else ((0, pad_m), (0, pad_n))
        a_padded = np.pad(
            a, padding,
            mode='constant',
            constant_values=0
        )
        return a_padded

    def restore_aspect_ratio(self, a):
        pad_m, pad_n = self._sqr_padding
        if pad_m == 0 and pad_n == 0:
            return a

        axes = [1, 2] if self._stacked else [0, 1]
        mm = self.output_shape[axes[0]] - int((pad_m / self.input_shape[axes[0]]) * self.output_shape[axes[0]])
        nn = self.output_shape[axes[0]] - int((pad_n / self.input_shape[axes[1]]) * self.output_shape[axes[1]])

        if self._stacked:
            return a[:, 0:mm, 0:nn]
        else:
            return a[0:mm, 0:nn]

    def unpad_array(self, a):
        """
        Removes array padding.

        :param a: Input array.
        :type a: ndarray<complex>
        :return: Padding array.
        :rtype: ndarray<complex>
        """

        nb = [0] if self._stacked else []
        axes = [1, 2] if self._stacked else [0, 1]
        for i in axes:
            cropped_nb = (2 * self._nb / self.input_shape[i]) * self.output_shape[i]
            cropped_nb // 2
            nb.append(int(cropped_nb))
        return unpad_arr(a, nb)

    def backwards(self, b):
        """
        Inverse Fourier transform

        :param b: Array to transform
        :type b: ndarray<complex>
        :return: Transformed array.
        :rtype: ndarray<complex>
        """

        if self._window is not None:
            window = np.stack([self._window for _ in range(self.output_shape[0])]) if self._stacked else self._window
            a = super().backwards(b*window)
        else:
            a = super().backwards(b)
        return self.restore_aspect_ratio(a)

    def add_window(self, window_fcn, **kwargs):
        if not self.masked:
            warn("Must align spatial filter before adding window function.")
            return False
        if callable(window_fcn):
            try:
                out_dims = self.output_shape[1:] if self._stacked else self.output_shape
                self._window = window_fcn(out_dims, **kwargs)
                return True
            except TypeError:
                warn("Invalid window function provided.")
                return False
        return False

    def construct_reference(self, f, M, N):
        tilt = self._calc_tilt(f)
        X, Y = gridspace(M, N, 0, True)
        return ref_phase_shift((X, Y), tilt, self._k0, self._sz)

    def calibrate(self, fringes, roi=None, mask_radius=None, optimize=True, visualize=False, **kwargs):
        """
        :param fringes: Hologram (typically of background) to calibrate phase filtering to.
        :param roi: A pre-selected ROI to calibrate with.
        :param mask_radius: Override the automatic radius of the mask if not None.
        :param optimize: Optimize the mask center according to some cost function.
        :param visualize: Show image of mask.
        :param kwargs: Optional arguments to optimization routine.
        :return: The ROI mask you select for calibration.
        :rtype: list
        """

        self.reset()
        a = self.fix_aspect_ratio(fringes)
        Fh = self.forwards(a)
        if roi is None:
            roi, f10 = select_mask_roi(np.abs(Fh)**(1/4))
        else:
            f10 = find_peak_in_roi(np.abs(Fh)**(1/4), roi)
        self._filter_radius, self._masks = off_axis_masks(Fh, f10=f10, mask_radius=mask_radius)
        freq = [f10[1] - Fh.shape[1] / 2, f10[0] - Fh.shape[0] / 2]

        # Attempt to optimize the spatial filter alignment
        if optimize:
            try:
                Fh = self.forwards(a) * self._masks[0]
                phi = np.angle(self.backwards(Fh))
                freq = self._optimize_tilt(freq, phi, **kwargs)
                f10 = [freq[1] + Fh.shape[1] / 2, freq[0] + Fh.shape[0] / 2]
                self._filter_radius, self._masks = off_axis_masks(Fh, f10=f10, mask_radius=mask_radius)
            except RuntimeError:
                warn("Could not align to tilt.")

        # Construct digital reference wave
        self._ref_wave = self.construct_reference(freq, self._dims[0], self._dims[1])

        # Crop ifft to spatial filter
        nstack = self.input_shape[0] if self._stacked else 0
        nthreads = self._ifft.threads
        flags = self._ifft.flags
        out_dir = self._ifft.direction
        dtype = self._ifft.dtype
        self._ifft = DiscreteTransform((2*self._filter_radius, 2*self._filter_radius), out_dir, dtype,
                                       threads=nthreads, nstack=nstack, flags=flags)
        if visualize:
            self._visualize_roi(fringes)
        return roi, f10
        

    def reset(self):
        """
        :return: Object reset to its initial state.
        """

        dims = self.input_shape[1:] if self._stacked else self.input_shape
        self._masks = None
        self._ref_wave = None
        self._window = None
        self._stacked = False

        self._fft = DiscreteTransform(dims, self._fft.direction, self._fft.dtype, threads=self._fft.threads,
                                      nstack=0, flags=self._ifft.flags)
        self._ifft = DiscreteTransform(dims, self._ifft.direction, self._ifft.dtype, threads=self._ifft.threads,
                                       nstack=0, flags=self._ifft.flags)
        return self



