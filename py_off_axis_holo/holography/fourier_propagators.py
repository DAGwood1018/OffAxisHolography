from abc import ABC, abstractmethod
import numpy as np

from py_off_axis_holo.holography.discrete_transforms import DFT, DiscreteTransform


class Convolve(DFT, ABC):

    @abstractmethod
    def _kernel(self, *args):
        ...

    def __call__(self, u, *args):
        """
        :param u: Input array.
        :type u: ndarray
        :param args: Arguments for convolution kernel.
        :return: Convolution.
        :rtype: ndarray
        """

        U = self.forwards(u)
        V = self._kernel(*args)
        if self._stacked:
            V = np.broadcast_to(V, U.shape)
        return self.backwards(U*V)


class AngularSpectrum(Convolve):

    def __init__(self, M, N, wl, sz, nb=0, threads=1, dtype='complex128', **kwargs):
        """
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

        super().__init__((M, N), nb, threads=threads, dtype=dtype, **kwargs)

        # Frequency coordinates on padded grid
        fx = np.fft.fftshift(np.fft.fftfreq(self.input_shape[1], sz))
        fy = np.fft.fftshift(np.fft.fftfreq(self.input_shape[0], sz))

        FX = fx.reshape(1, self.input_shape[1])
        FY = fy.reshape(self.input_shape[0], 1)
        k = 2 * np.pi / wl
        self._kz = k * np.sqrt(1.0 - wl**2 * (FX**2 + FY**2))

    def _kernel(self, z):
        """
        Creates angular spectrum transfer function

        :param z: Distance to propagate.
        :type z: float
        :return: The transfer function.
        :rtype: ndarray
        """

        return np.exp(1j * self._kz * z)

class ParaxialAngularSpectrum(Convolve):

    def __init__(self, M, N, wl, sz, nb=0, threads=1, dtype='complex128', **kwargs):
        """
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

        super().__init__((M, N), nb, threads=threads, dtype=dtype, **kwargs)

        # Frequency coordinates on padded grid
        fx = np.fft.fftshift(np.fft.fftfreq(self.input_shape[1], sz))
        fy = np.fft.fftshift(np.fft.fftfreq(self.input_shape[0], sz))

        FX = fx.reshape(1, self.input_shape[1])
        FY = fy.reshape(self.input_shape[0], 1)
        k = 2 * np.pi / wl
        self._kz_approx = k * (1 - wl**2 * (FX**2 + FY**2)/2)

    def _kernel(self, z):
        """
        Creates angular spectrum transfer function

        :param z: Distance to propagate.
        :type z: float
        :return: The transfer function.
        :rtype: ndarray
        """

        return np.exp(1j * self._kz_approx * z)


class PairPropagator:

    def __init__(self, M, N, wl, sz, nb=0, upsample=1, paraxial=True, threads=1,
                 dtype='complex128', **kwargs):
        """
        Propagates object fields S and background fields B of shape (M, N), or
        stacks of them with shape (K, M, N), and returns t(z) = P_z{S} / P_z{B}.
        Uses a single ParaxialAngularSpectrum / AngularSpectrum whose stack size is
        switched (stack_arrays / unstack_arrays) to match the input. Switching is
        cheap after the first time a size is used, since FFTW reuses its plans.

        Typical use:
            prop = PairPropagator(M, N, wl, sz)
            t = prop(S, B, z)                     # S, B: (M, N) or (K, M, N)

        For scans over many z, compute the spectra once:
            FS, FB = prop.forwards(S), prop.forwards(B)
            t = prop.ratio(FS, FB, z)

        :param M, N: Image dimensions.
        :type M, N: int
        :param wl: The wavelength.
        :type wl: float
        :param sz: The pixel size of the input fields.
        :type sz: float
        :param nb: Padding to use. Default is 0.
        :type nb: int
        :param upsample: Integer band-limited (sinc) upsampling factor of the output. Adds no
            information but reduces the per-pixel phase gradient, which helps phase unwrapping.
            Default is 1.
        :type upsample: int
        :param paraxial: Use ParaxialAngularSpectrum (True) or AngularSpectrum (False).
        :type paraxial: bool
        :param threads: Number of threads to use. Default is 1.
        :type threads: int
        :param dtype: The dtype to use. Default is 'complex128'.
        :type dtype: string
        :param kwargs: Additional parameters to create discrete transforms.
        """

        M, N = int(M), int(N)
        assert isinstance(upsample, (int, np.integer)) and upsample >= 1, "upsample must be an integer >= 1."

        prop_cls = ParaxialAngularSpectrum if paraxial else AngularSpectrum
        self._prop = prop_cls(M, N, wl, sz, nb=nb, threads=threads, dtype=dtype, **kwargs)

        self._dims = np.array([M, N])
        self._k0 = 2 * np.pi / wl
        self._sz = sz
        self._nb_pad = nb
        self._up = int(upsample)
        self._paraxial = paraxial
        self._threads = threads
        self._dtype = dtype

        self._nstack = 0                     # current stack size (0 = unstacked)
        self._pad_dims = np.array(self._prop.input_shape[-2:], dtype=int)
        self._up_dims = self._pad_dims * self._up
        self._ifft_up = None                 # inverse transform on the upsampled grid
        self._nstack_up = None               # its stack size

    def __call__(self, S, B, z, eps=1e-3):
        """
        Object field divided by background field, both propagated to z:
        t(z) = P_z{S} / P_z{B}, regularised where |B| is small.

        :param S: Object field(s), shape (M, N) or (K, M, N).
        :type S: ndarray
        :param B: Background field(s), shape (M, N) or (K, M, N). A single (M, N)
            background is used for every array of a stacked S.
        :type B: ndarray
        :param z: Distance to propagate: scalar, or one value per stacked array.
        :type z: float or ndarray
        :param eps: Regularisation relative to mean |B|^2 (per array). Default is 1e-3.
        :type eps: float
        :return: t(z), same stacking as S.
        :rtype: ndarray
        """

        k_s, k_b = self._stack_size(S), self._stack_size(B)
        assert k_b in (0, k_s), "B must be a single field or have the same stack size as S."
        return self.ratio(self.forwards(S), self.forwards(B), z, eps=eps)

    def _stack_size(self, a):
        """
        :param a: Field of shape (M, N) or stack (K, M, N).
        :type a: ndarray
        :return: 0 for a single array, K for a stack.
        :rtype: int
        """

        a = np.asarray(a)
        M, N = self._dims
        assert a.ndim in (2, 3) and a.shape[-2:] == (M, N), \
            f"Shape of array must be ({M}, {N}) or (K, {M}, {N}), got {a.shape}."
        return 0 if a.ndim == 2 else a.shape[0]

    def _kernel(self, z, n):
        """
        Transfer function for scalar z, or one z per stacked array.

        :param z: Distance to propagate.
        :type z: float or ndarray
        :param n: Stack size of the spectrum it multiplies (0 = unstacked).
        :type n: int
        :return: The transfer function.
        :rtype: ndarray
        """

        z = np.asarray(z, dtype=float)
        if z.ndim == 0:
            return self._prop._kernel(float(z))
        assert n > 0 and z.shape == (n,), "z must be a scalar or have one value per stacked array."
        return self._prop._kernel(z[:, None, None])

    def _upsampled_backwards(self, b):
        """
        Inverse transform on the upsampled grid (zero-padded spectrum), then remove padding.

        :param b: Fftshift-ed, padded spectrum.
        :type b: ndarray
        :return: Upsampled field.
        :rtype: ndarray
        """

        n = 0 if b.ndim == 2 else b.shape[0]
        if self._nstack_up != n:
            self._ifft_up = DiscreteTransform(tuple(self._up_dims), 'FFTW_BACKWARD', self._dtype,
                                              threads=self._threads, nstack=int(n))
            self._nstack_up = n

        P, U = self._pad_dims, self._up_dims
        g = np.zeros(b.shape[:-2] + tuple(U), dtype=b.dtype)
        # keep DC (index P//2 after fftshift) at the centre of the larger grid
        oy, ox = U // 2 - P // 2
        g[..., oy:oy + P[0], ox:ox + P[1]] = b
        # forwards already applies the 1/N scaling (normalise_idft=False), so the
        # inverse here is unscaled, as in DFT.backwards
        a = self._ifft_up(np.fft.ifftshift(g, axes=(-2, -1)), ortho=False, normalise_idft=False)
        nb = self._nb_pad * self._up
        return a[..., nb:a.shape[-2] - nb, nb:a.shape[-1] - nb] if nb > 0 else a

    @property
    def wl(self):
        """
        :return: The wavelength.
        :rtype: float
        """

        return 2 * np.pi / self._k0

    @property
    def stacked(self):
        """
        :return: Current stack size (0 if unstacked).
        :rtype: int
        """

        return self._nstack

    @property
    def input_shape(self):
        """
        :return: Shape of the padded arrays transformed by forwards (includes stack axis).
        :rtype: tuple<int>
        """

        return self._prop.input_shape

    @property
    def output_shape(self):
        """
        :return: Shape of the fields returned (includes stack axis and upsampling).
        :rtype: tuple<int>
        """

        M, N = (int(d) * self._up for d in self._dims)
        return (self._nstack, M, N) if self._nstack else (M, N)

    def out_pxl_sz(self):
        """
        :return: Pixel size of the returned fields (input pixel size / upsample).
        :rtype: float
        """

        return self._sz / self._up

    def stack_arrays(self, n):
        """
        Process n arrays at once. Called automatically by forwards/backwards.

        :param n: How many arrays you want to process.
        :type n: int
        :return: bool
        """

        if n == self._nstack:
            return True
        ok = self._prop.stack_arrays(int(n))
        if ok:
            self._nstack = int(n)
        return ok

    def unstack_arrays(self):
        """
        Return to processing a single array at a time.

        :return: bool
        """

        if self._nstack == 0:
            return True
        ok = self._prop.unstack_arrays()
        self._nstack = 0
        return ok

    def forwards(self, a):
        """
        :param a: Field of shape (M, N) or stack (K, M, N).
        :type a: ndarray
        :return: Padded, fftshift-ed spectrum (same stacking as a).
        :rtype: ndarray
        """

        n = self._stack_size(a)
        self.stack_arrays(n) if n else self.unstack_arrays()
        return self._prop.forwards(a)

    def backwards(self, b):
        """
        :param b: Padded, fftshift-ed spectrum (from forwards).
        :type b: ndarray
        :return: Field, upsampled if requested, with padding removed.
        :rtype: ndarray
        """

        if self._up > 1:
            return self._upsampled_backwards(b)
        n = 0 if b.ndim == 2 else b.shape[0]
        self.stack_arrays(n) if n else self.unstack_arrays()
        return self._prop.backwards(b)

    def propagate(self, a, z):
        """
        :param a: Field of shape (M, N) or stack (K, M, N).
        :type a: ndarray
        :param z: Distance to propagate: scalar, or one value per stacked array.
        :type z: float or ndarray
        :return: P_z{a}.
        :rtype: ndarray
        """

        b = self.forwards(a)
        return self.backwards(b * self._kernel(z, 0 if b.ndim == 2 else b.shape[0]))

    def ratio(self, FS, FB, z, eps=1e-3):
        """
        t(z) = P_z{S} / P_z{B} from precomputed spectra. FB may be a single
        spectrum shared by every array of a stacked FS (common background).

        :param FS: Spectrum of S from forwards().
        :type FS: ndarray
        :param FB: Spectrum of B from forwards().
        :type FB: ndarray
        :param z: Distance to propagate: scalar, or one value per stacked array.
        :type z: float or ndarray
        :param eps: Regularisation relative to mean |B|^2 (per array). Default is 1e-3.
        :type eps: float
        :return: t(z).
        :rtype: ndarray
        """

        n = 0 if FS.ndim == 2 else FS.shape[0]
        H = self._kernel(z, n)
        s = self.backwards(FS * H)
        if FB.ndim == 2 and n > 0 and np.ndim(z) == 0:
            b = self.backwards(FB * H)[None]               # shared background, same z
        else:
            if FB.ndim == 2 and n > 0:
                FB = np.broadcast_to(FB, FS.shape)
            b = self.backwards(FB * H)
        b2 = np.abs(b) ** 2
        return s * np.conj(b) / (b2 + eps * b2.mean(axis=(-2, -1), keepdims=True))
