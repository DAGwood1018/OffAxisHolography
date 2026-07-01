from warnings import warn
import numpy as np
import cv2

from py_off_axis_holo.discrete_transforms import DFT, DiscreteTransform
from py_off_axis_holo.fourier_propagators import AngularSpectrum
from py_off_axis_holo.holography_helpers import format_holo, select_mask_roi, find_peak_in_roi, off_axis_masks


class OffAxisFilter(DFT):

    def __init__(self, M, N, wl, sz, nb=0, threads=1, dtype='complex128', **kwargs):
        """
        *Using un-normalized FFTs is fine since we are only concerned with the phase information.

        :param M, N: Image dimensions.
        :type M, N: int
        :param wl: The wavelength.
        :type wl: float
        :param sz: The pixel size.
        :type sz: float
        :param nb: Padding to use. Default is 0.
        :type nb: int
        :param threads: Number of threads to use. Default is 1.
        :type threads: int
        :param dtype: The dtype to use. Default is 'complex128'.
        :type dtype: string
        :param kwargs: Additional parameters to create discrete transforms.
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

        self._sqr_padding = (pad_m, pad_n)
        self._ref = np.ones((M, N), dtype=dtype)
        self._mask = np.ones((self.input_shape[0], self.input_shape[1]), dtype=dtype)
        self._window = None
        self._kNA = -1

    def __call__(self, fringes):
        """
        :param fringes: Interferogram corresponding to an off-axis hologram
        :type fringes: ndarray
        :return: The +/-1 (off-axis) term of the hologram
        :rtype: ndarray
        """

        if self._stacked:
            assert fringes.ndim == 3, "ndim must be 3 if working with multiple arrays."
        else:
            assert fringes.ndim == 2, "ndim must be 2 if working with a single array."
        if not self.calibrated:
            if self._stacked:
                self.unstack_arrays()
                raise RuntimeError("Must calibrate filter before stacking arrays.")
            self.calibrate(fringes)

        fringes = fringes.astype(self._fft.dtype)
        b = self.forwards(fringes)
        bf = self.filter_spectrum(b)
        a = self.backwards(bf)
        return a

    def _get_tilt(self, px, py):
        """
        Calculates tilt from frequency coordinates. f is
        assumed to be relative to the center coordinates.

        :param fx, fy: Frequency coords of +/-1 order.
        :type fx, fy: float
        :return: Tilt in radians.
        :rtype: ndarray
        """

        M, N = self._dims + 2*self._nb_pad
        fx, fy = (px - N/2) / (N*self._sz), (py - M/2) / (M*self._sz)

        kx = 2 * np.pi * fx
        ky = 2 * np.pi * fy
        return np.arcsin(np.array([kx, ky]) / self._k0)

    def _get_peak(self, a, b):
        """
        Calculates position of peak coordinates from tilt.

        :param a, b: Tilt of the reference beam in radians.
        :type a, b: float
        :return: Coords of +/-1 order.
        :rtype: ndarray
        """

        M, N = self._dims + 2 * self._nb_pad
        k = self._k0 * np.sin(np.array([a, b]))
        f = k/(2*np.pi)

        px = N*self._sz*f[0] + N/2
        py = M*self._sz*f[1] + M/2
        return np.array([px, py])

    def _fit_ref(self, U, weights=None, tol=1e-12):
        """
        Using the finite difference of the phase, find a fit to the tilt and quadratic phase shift.
        Phase fit as phi(x, y) = a*x + b*y + c*(x^2 + y^2).

        :param U: The complex field spatially filtered around estimate of tilt.
        :type U: ndarray
        :param weights: Weights to use in fit. No weights are applied if None.
        :type weights: ndarray
        :param tol: Tolerance parameter used to prevent divide by 0 error.
        :return: a, b, c fit parameters.
        :rtype: tuple<floats>
        """

        print(">>> Fitting Inferred Tilt...")
        U /= (np.abs(U) + tol)
        y, x = np.indices(U.shape, dtype=float)
        x -= x.mean()
        y -= y.mean()

        # wrapped phase differences
        gx = np.angle(U[:, 1:] * np.conj(U[:, :-1]))
        gy = np.angle(U[1:, :] * np.conj(U[:-1, :]))

        # midpoint coordinates
        xgx = 0.5 * (x[:, 1:] + x[:, :-1])
        ygy = 0.5 * (y[1:, :] + y[:-1, :])

        # gx = a + 2*c*x
        Ax = np.column_stack([
            np.ones(gx.size),
            np.zeros(gx.size),
            2 * xgx.ravel()
        ])

        # gy = b + 2*c*y
        Ay = np.column_stack([
            np.zeros(gy.size),
            np.ones(gy.size),
            2 * ygy.ravel()
        ])

        A = np.vstack((Ax, Ay)) * self._k0
        bvec = np.concatenate((gx.ravel(), gy.ravel()))
        if not weights is None:
            assert weights.shape == U.shape, "Weights must have same shape as U."
            wx = 0.5 * (weights[:, 1:] + weights[:, :-1])
            wy = 0.5 * (weights[1:, :] + weights[:-1, :])
            w = np.concatenate((wx.ravel(), wy.ravel()))
            Aw = A * w[:, None]
            bw = bvec * w

            coeffs, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
        else:
            coeffs, *_ = np.linalg.lstsq(A, bvec, rcond=None)

        print(">>> Fitting Complete.")
        a, b, c = coeffs
        return a, b, c

    def _crop_ifft(self, M, N):
        """
        Crops ifft to smaller size.

        :param M, N: Dimensions to crop to.
        :type M, N: int
        """

        # Crop ifft to spatial filter
        nthreads = self._ifft.threads
        flags = self._ifft.flags
        out_dir = self._ifft.direction
        dtype = self._ifft.dtype
        self._ifft = DiscreteTransform((M, N), out_dir, dtype,
                                       threads=nthreads, nstack=0, flags=flags)
        self._nb_unpad = (2 * self._nb_pad / self.input_shape[0]) * self.output_shape[0]
        self._nb_unpad = int(self._nb_unpad // 2)

    def _view_filter(self, fringes):
        """
        Visualize the calibrated filter applied to given interferogram.

        :param fringes: An interferogram.
        :type fringes: ndarray
        """

        Fh = self.forwards(fringes)
        Fh = format_holo(np.abs(Fh) ** (1 / 4))

        cv2.namedWindow('visualize_roi', cv2.WINDOW_NORMAL)
        cv2.imshow('visualize_roi', Fh)

        cv2.waitKey(1500)
        Fh *= self._mask.astype('uint8')
        cv2.imshow('visualize_roi', Fh)
        cv2.waitKey(4000)

        if cv2.getWindowProperty('visualize_roi', cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow('visualize_roi')

    @property
    def calibrated(self):
        """
        :return: True if filter is calibrated.
        :rtype: bool
        """

        if self._kNA < 0:
            return False
        return True

    def fix_aspect_ratio(self, a):
        """
        Adds padding as need to make the aspect ratio 1:1, preventing distortions.

        :param a: The original interferogram.
        :type a: ndarray
        :return: Interferogram with guaranteed 1:1 aspect ratio.
        :rtype: ndarray
        """

        pad_m, pad_n = self._sqr_padding
        if pad_m == pad_n == 0:
            return a

        padding = ((0, 0),(0, pad_m),(0, pad_n)) if self._stacked else ((0, pad_m),(0, pad_n))
        a_padded = np.pad(
            a, padding,
            mode='constant',
            constant_values=0
        )
        return a_padded

    def restore_aspect_ratio(self, a):
        """
        Removes padding as needed to restore original aspect ratio.

        :param a: The filtered interferogram.
        :type a: ndarray
        :return: Interferogram with original aspect ratio.
        :rtype: ndarray
        """

        pad_m, pad_n = self._sqr_padding
        if pad_m == 0 and pad_n == 0:
            return a

        axes = [1, 2] if self._stacked else [0, 1]
        input_shape = np.array(self.input_shape) - 2*self._nb_pad
        output_shape = np.array(self.output_shape) - 2*self._nb_unpad
        mm = output_shape[axes[0]] - int((pad_m / input_shape[axes[0]]) * output_shape[axes[0]])
        nn = output_shape[axes[1]] - int((pad_n / input_shape[axes[1]]) * output_shape[axes[1]])

        return a[:, 0:mm, 0:nn] if self._stacked else a[0:mm, 0:nn]

    def forwards(self, a):
        """
        Forward Fourier transform

        :param a: Array to transform
        :type a: ndarray<complex>
        :return: Transformed array.
        :rtype: ndarray<complex>
        """

        ref = np.broadcast_to(self._ref, (self.input_shape[0], self._dims[0], self._dims[1])) \
            if self._stacked else self._ref
        a = self.fix_aspect_ratio(a)
        return super().forwards(a*ref)

    def filter_spectrum(self, b):
        """
        Crop Fourier plane to the spatial filter mask.

        :param b: The Fourier plane to crop.
        :type b: ndarray
        :return: Cropped Fourier plane.
        :rtype: ndarray
        """

        mask = np.broadcast_to(self._mask, self.input_shape) \
            if self._stacked else self._mask
        bf = b*mask

        coords = np.argwhere(self._mask)
        m_min, n_min = coords.min(axis=0)
        m_max, n_max = coords.max(axis=0)
        return bf[:, m_min:m_max, n_min:n_max] \
            if self._stacked else b[m_min:m_max, n_min:n_max]

    def backwards(self, b):
        """
        Inverse Fourier transform

        :param b: Array to transform
        :type b: ndarray<complex>
        :return: Transformed array.
        :rtype: ndarray<complex>
        """

        if self._window is not None:
            window = np.broadcast_to(self._window, self.output_shape) \
                if self._stacked else self._window
            b *= window
        a = super().backwards(b)
        return self.restore_aspect_ratio(a)

    def add_window(self, window_fcn, **kwargs):
        """
        Add window to use as bandpass filter in Fourier plane.

        :param window_fcn: Window function that takes dimensions as argument.
        :param kwargs: Any optiona arguments for the window function.
        :return:
        """

        if not self.calibrated:
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

    def create_ref(self, M, N, a, b, c=0):
        """
        Create reference plane wave for tilt compensation.

        :param M, N: Dimensions of reference wave.
        :type M, N: int
        :param a, b, c: The phase fit parameters
        :type a, b, c: float
        :return: The reference wave.
        :rtype: ndarray
        """

        x = (np.arange(N) - N / 2) * self._sz
        y = (np.arange(M) - M / 2) * self._sz
        X, Y = np.meshgrid(x, y)
        k = self._k0 * np.sin(np.array([a, b]))
        return np.exp(-1j * (k[0] * X + k[1] * Y + self._k0*c*(X**2 + Y**2)))

    def intensity(self, fringes):
        """
        Filters for |O|^2+|R|^2 (the DC term).

        :param fringes: Interferogram corresponding to an off-axis hologram.
        :type fringes: ndarray
        :return: The DC term of the hologram.
        :rtype: ndarray
        """

        ref = self._ref
        self._ref = np.ones_like(ref)
        dc = self.__call__(fringes)
        self._ref = ref
        return np.real(dc)

    def contrast(self, fringes):
        """
        Estimates the contrast of the hologram.

        :param fringes: Interferogram corresponding to an off-axis hologram.
        :type fringes: ndarray
        :return: An estimate of the contrast of the hologram.
        :rtype: ndarray
        """

        return 2 * np.abs(self.__call__(fringes)) / self.intensity(fringes)

    def calibrate(self, fringes, dz=0, kNA=-1, roi=None, optimize=True, visualize=False, **kwargs):
        """
        :param fringes: Hologram (typically of background) to calibrate phase filtering to.
        :param dz: A small distance to propagate the fringes by. By assuming telecentricity this is used to isolate tilt + aberations from signal.
        :param kNA: The known numerical aperture of the system in terms of pixels. Overrides the automatic radius of the mask if not None.
        :param roi: A pre-selected ROI to calibrate with.
        :param optimize: Optimize the mask center according to some cost function.
        :param visualize: Show image of mask.
        :param kwargs: Optional arguments to optimization routine.
        :return: The ROI mask you select for calibration and reference beam fit parameters
        :rtype: ndarray, float, float, float
        """

        if self.calibrated:
            self.reset()
        if self._stacked:
            self.unstack_arrays()
        assert fringes.ndim == 2, "Calibration requires input interferogram of ndim==2."

        fringes = fringes.astype(self._fft.dtype)
        Fh = self.forwards(fringes)
        if roi is None:
            roi, fp1 = select_mask_roi(np.abs(Fh) ** (1 / 4))
        else:
            fp1 = find_peak_in_roi(np.abs(Fh) ** (1 / 4), roi)
        self._kNA, self._mask, _, _ = off_axis_masks(Fh, fp1=fp1, kNA=kNA)

        # Construct digital reference wave
        c = 0
        a, b = self._get_tilt(fp1[1], fp1[0])
        self._ref = self.create_ref(self._dims[0], self._dims[1], a, b, c)
        self._crop_ifft(2 * self._kNA, 2 * self._kNA)

        # Attempt to optimize the spatial filter alignment
        if optimize:
            try:
                U = self(fringes)
                if dz > 0 or dz < 0:
                    prop = AngularSpectrum(U.shape[0], U.shape[1],
                                           2 * np.pi / self._k0, self._sz, nb=self._nb_pad)
                    Uz = prop(U, dz)
                    U = Uz * U.conj()
                    del prop

                da, db, c = self._fit_ref(U, **kwargs)
                a += da
                b += db

                # Reconstruct mask and digital reference wave
                fp1 = self._get_peak(a, b)
                self._kNA, self._mask, _, _ = off_axis_masks(Fh, fp1=fp1, kNA=kNA)
                self._ref = self.create_ref(self._dims[0], self._dims[1], a, b, c)
                self._crop_ifft(2 * self._kNA, 2 * self._kNA)
            except RuntimeError:
                warn("Could not align to tilt.")

        if visualize:
            self._view_filter(fringes)
        return roi, a, b, c
        
    def set_calibration(self, a, b, c=0, kNA=-1):
        """
        :param a, b: Known tilt in radians.
        :param c: Coefficient of quadratic phase shift of reference beam. 0 by default.
        :param kNA: The numerical aperture of the system (in terms of pixels). This determines the radius of the mask.
        If not given, it will be assumed that you aligned system in the most optimal way given location of peak.
        """

        if self.calibrated:
            self.reset()
        if self._stacked:
            self.unstack_arrays()

        fp1 = self._get_peak(a, b)
        self._kNA, self._mask, _, _ = off_axis_masks(np.zeros(self.input_shape), fp1=fp1, kNA=kNA)

        # Construct digital reference wave
        self._ref = self.create_ref(self._dims[0], self._dims[1], a, b, c)
        self._crop_ifft(2 * self._kNA, 2 * self._kNA)

    def reset(self):
        """
        Reset object to original uncalibrated state.
        """

        dims = self.input_shape[1:] if self._stacked else self.input_shape
        self._ref = np.ones((self._dims[0], self._dims[1]), dtype=float)
        self._mask = np.ones((self._dims[0], self._dims[1]))
        self._window = None
        self._kNA = -1
        self._stacked = False

        self._fft = DiscreteTransform(dims, self._fft.direction, self._fft.dtype, threads=self._fft.threads,
                                      nstack=0, flags=self._ifft.flags)
        self._ifft = DiscreteTransform(dims, self._ifft.direction, self._ifft.dtype, threads=self._ifft.threads,
                                       nstack=0, flags=self._ifft.flags)
        self._nb_unpad = self._nb_pad
