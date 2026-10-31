from warnings import warn
import numpy as np

from py_off_axis_holo.holography.discrete_transforms import DFT, DiscreteTransform
from py_off_axis_holo.holography.holography_helpers import select_mask_roi, find_peak_in_roi, find_centroid_in_roi, \
    off_axis_masks, view_filter


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
        self._fit_params = (0, 0, 0)
        self._quad = (0.0, 0.0, 0.0)   # (c, xc, yc): quadratic phase removed after filtering, see calibrate
        self._post = None              # cached exp(-i*k0*c*|r - rc|^2) on the output grid

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
        if self._quad[0] != 0:
            a = a * self._post_phase(a.shape[-2:])
        return a

    @property
    def curvature(self):
        """
        :return: (c, xc, yc): the quadratic phase k0*c*|r - rc|^2 removed from the filtered field
            (c in 1/m, centre in m, camera-plane coordinates with the same origin as create_ref).
        :rtype: tuple<float>
        """

        return self._quad

    def _out_coords(self, shape):
        """Camera-plane coordinates (m) of the filtered output grid, origin as in create_ref."""

        m, n = shape
        p = self.out_pxl_sz()
        x = np.arange(n) * p - 0.5 * self._dims[1] * self._sz
        y = np.arange(m) * p - 0.5 * self._dims[0] * self._sz
        return np.meshgrid(x, y)

    def _post_phase(self, shape):
        """Quadratic phase correction applied to the filtered field (cached per output shape)."""

        if self._post is None or self._post.shape != tuple(shape):
            c, xc, yc = self._quad
            X, Y = self._out_coords(shape)
            self._post = np.exp(-1j * self._k0 * c * ((X - xc) ** 2 + (Y - yc) ** 2))
        return self._post

    def _get_tilt(self, fp1):
        """
        Calculates tilt from the position of the +1 order on the padded grid.

        :param fp1: (row, col) of the +1 order.
        :type fp1: array-like
        :return: Tilt in radians (x, y).
        :rtype: ndarray
        """

        py, px = fp1
        M, N = self._dims + 2*self._nb_pad
        fx, fy = (px - N/2) / (N*self._sz), (py - M/2) / (M*self._sz)

        kx = 2 * np.pi * fx
        ky = 2 * np.pi * fy
        return np.arcsin(np.array([kx, ky]) / self._k0)

    def _get_peak(self, a, b):
        """
        Calculates position of peak coordinates from tilt.

        :param a, b: Tilt of the reference beam in radians (x, y).
        :type a, b: float
        :return: (row, col) of the +1 order.
        :rtype: ndarray
        """

        M, N = self._dims + 2 * self._nb_pad
        k = self._k0 * np.sin(np.array([a, b]))
        f = k/(2*np.pi)

        px = N*self._sz*f[0] + N/2
        py = M*self._sz*f[1] + M/2
        return np.array([py, px])

    def _fit_ref(self, U):
        """
        Fit the phase of the filtered order as phi = dk.(r - rc) + q*|r - rc|^2 (+ const) using
        wrapped finite differences, so no unwrapping is needed. rc is the |U|^2-weighted centre
        of the field, so dk is the carrier that centres the order's spectrum in the mask.

        :param U: The complex field spatially filtered around the current estimate of the tilt.
        :type U: ndarray
        :return: dkx, dky (rad/m) to add to the reference wavevector, q (rad/m^2), and the
            centre (xc, yc) in m.
        :rtype: float, float, float, tuple<float>
        """

        p = self.out_pxl_sz()                     # camera-plane pitch of U
        X, Y = self._out_coords(U.shape)
        wU = np.abs(U) ** 2
        xc, yc = (wU * X).sum() / wU.sum(), (wU * Y).sum() / wU.sum()
        X, Y = (X - xc) / p, (Y - yc) / p          # pixel units for conditioning

        # wrapped phase increments and their weights
        dUx = U[:, 1:] * np.conj(U[:, :-1])
        dUy = U[1:, :] * np.conj(U[:-1, :])
        gx, gy = np.angle(dUx), np.angle(dUy)
        wx, wy = np.sqrt(np.abs(dUx)), np.sqrt(np.abs(dUy))
        xm = 0.5 * (X[:, 1:] + X[:, :-1])
        ym = 0.5 * (Y[1:, :] + Y[:-1, :])

        # per-pixel model:  gx = alpha + 2*gamma*x,  gy = beta + 2*gamma*y   (pixel units)
        A = np.column_stack([np.concatenate((np.ones(gx.size), np.zeros(gy.size))),
                             np.concatenate((np.zeros(gx.size), np.ones(gy.size))),
                             np.concatenate((2 * xm.ravel(), 2 * ym.ravel()))])
        bvec = np.concatenate((gx.ravel(), gy.ravel()))
        w = np.concatenate((wx.ravel(), wy.ravel()))
        w = w / w.max()
        coeffs, *_ = np.linalg.lstsq(A * w[:, None], bvec * w, rcond=None)

        alpha, beta, gamma = coeffs
        return alpha / p, beta / p, gamma / p ** 2, (xc, yc)

    def _set_filter(self, Fh, fp1, kNA, a, b):
        """Build mask, tilt-only reference wave and cropped ifft for a given +1 order position."""

        self._kNA, self._mask, _, _ = off_axis_masks(Fh, fp1=fp1, kNA=kNA)
        self._ref = self.create_ref(self._dims[0], self._dims[1], a, b, 0)
        self._crop_ifft(2 * self._kNA, 2 * self._kNA)
        self._post = None

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
        return bf[..., m_min:m_max, n_min:n_max]

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
        :return: True if sucessful. False otherwise.
        :rtype: bool
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

    def out_pxl_sz(self, mag=1.0):
        """
        Size of pixels after filtering.

        :param mag: The magnification of the imaging system.
        :type mag: float
        :return: Output pixel size.
        :rtype: float
        """

        # last axis: works for stacked arrays too (grids are square)
        n_in = self.input_shape[-1]
        n_out = self.output_shape[-1]
        return self._sz * n_in / n_out / mag

    def num_aperture(self, mag=1.0):
        """
        Size of numerical aperture.

        :param mag: The magnification of the imaging system.
        :type mag: float
        :return: The numerical aperture size.
        :rtype: float, None if not calibrated.
        """

        if not self.calibrated:
            return None
        return self.out_pxl_sz(mag=mag) * self._kNA

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

        ref, quad = self._ref, self._quad
        self._ref, self._quad = np.ones_like(ref), (0.0, 0.0, 0.0)
        dc = self.__call__(fringes)
        self._ref, self._quad = ref, quad
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

    def calibrate(self, fringes, kNA=-1, roi=None, optimize=True, visualize=False, method='centroid',
                  n_iter=10, tol=0.05, **kwargs):
        """
        :param fringes: Hologram (typically of background) to calibrate phase filtering to.
        :param kNA: The known numerical aperture of the system in terms of pixels. Overrides the automatic radius of the mask if not None.
        :param roi: A pre-selected ROI (x, y, w, h) to calibrate with.
        :param optimize: Refine the tilt from the phase of the filtered order (iterating until the order
            moves less than tol pixels, at most n_iter times), then fit its quadratic phase (curvature)
            and remove it from every filtered field. The curvature is removed after filtering rather than
            with the reference wave: a chirped reference would smear the DC and -1 orders into the mask.
            Removing it acts like a lens on the field (shifts the focus found by propagation).
        :param visualize: Show image of mask.
        :param method: Initial estimate of the order position: 'centroid' (power-weighted centroid of
            the lobe; robust to defocus) or 'peak' (the brightest pixel; old behaviour).
        :param n_iter: Maximum number of refinement iterations.
        :param tol: Convergence tolerance on the order position (spectral pixels).
        :param centroid_kw: Optional dict passed to find_order_centroid (smooth, rel_thresh).
        :param kwargs: Optional arguments to the fit (_fit_ref), e.g. weights.
        :return: The ROI you selected, fp1 = (row, col) of the +1 order on the padded spectrum
            (can be passed to set_calibration), and the curvature c removed from the filtered field.
            The full (c, xc, yc) is in self.curvature; pass c and centre=curvature[1:] to set_calibration.
        :rtype: tuple, ndarray, float
        """

        if self.calibrated:
            self.reset()
        if self._stacked:
            self.unstack_arrays()
        assert fringes.ndim == 2, "Calibration requires input interferogram of ndim==2."
        assert method in ('centroid', 'peak'), "method must be 'centroid' or 'peak'."

        fringes = fringes.astype(self._fft.dtype)
        Fh = self.forwards(fringes)
        if roi is None:
            roi, fp1 = select_mask_roi(Fh, require_peak=(method == 'peak'))
        elif method == 'peak':
            fp1 = find_peak_in_roi(Fh, roi)
        if method == 'centroid':
            fp1 = find_centroid_in_roi(Fh, roi, **kwargs)

        a, b = self._get_tilt(fp1)
        self._quad = (0.0, 0.0, 0.0)
        self._set_filter(Fh, fp1, kNA, a, b)

        if optimize:
            try:
                for it in range(n_iter):
                    U = self(fringes)               # tilt-only filtering (no curvature removed yet)
                    dkx, dky, q, centre = self._fit_ref(U, **kwargs)
                    k = self._k0 * np.sin(np.array([a, b])) + np.array([dkx, dky])
                    a, b = np.arcsin(k / self._k0)

                    fp1_new = self._get_peak(a, b)
                    shift = np.hypot(*(fp1_new - fp1))
                    fp1 = fp1_new
                    self._set_filter(Fh, fp1, kNA, a, b)
                    if shift < tol:
                        break
                else:
                    warn(f"Tilt refinement did not converge in {n_iter} iterations (last step {shift:.2f} px).")

                # final curvature fit on the converged filter, then remove it from the filtered field
                _, _, q, centre = self._fit_ref(self(fringes))
                self._quad = (q / self._k0, centre[0], centre[1])
                self._post = None
                print(f">>> Order at ({fp1[0]:.2f}, {fp1[1]:.2f}) after {it + 1} refinement(s); "
                      f"removed curvature c = {self._quad[0]:.4g} 1/m about "
                      f"({1e3 * centre[0]:.3f}, {1e3 * centre[1]:.3f}) mm.")
            except RuntimeError:
                warn("Could not align to tilt.")

        self._fit_params = (a, b, self._quad[0])
        if visualize:
            view_filter(self.forwards(fringes), self._mask)
        return roi, fp1, self._quad[0]

    def set_calibration(self, fp1, c=0, kNA=-1, centre=(0.0, 0.0)):
        """
        :param fp1: Location (row, col) of the +1 order on the padded spectrum, as returned by calibrate().
        :param c: Coefficient of the quadratic phase removed from the filtered field (1/m), as returned
            by calibrate(). 0 by default.
        :param kNA: The numerical aperture of the system (in terms of pixels). This determines the radius of the mask.
        If not given, it will be assumed that you aligned system in the most optimal way given location of peak.
        :param centre: (xc, yc) in m of the quadratic phase; calibrate() stores it in curvature[1:].
        """

        if self.calibrated:
            self.reset()
        if self._stacked:
            self.unstack_arrays()

        a, b = self._get_tilt(fp1)
        self._set_filter(np.zeros(self.input_shape), fp1, kNA, a, b)
        self._quad = (float(c), float(centre[0]), float(centre[1]))
        self._fit_params = (a, b, c)

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
        self._fit_params = (0, 0, 0)
        self._quad = (0.0, 0.0, 0.0)
        self._post = None

        self._fft = DiscreteTransform(dims, self._fft.direction, self._fft.dtype, threads=self._fft.threads,
                                      nstack=0, flags=self._ifft.flags)
        self._ifft = DiscreteTransform(dims, self._ifft.direction, self._ifft.dtype, threads=self._ifft.threads,
                                       nstack=0, flags=self._ifft.flags)
        self._nb_unpad = self._nb_pad
