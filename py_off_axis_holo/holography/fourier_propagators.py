from abc import ABC, abstractmethod
import numpy as np

from py_off_axis_holo.holography.discrete_transforms import DFT, DiscreteTransform
from py_off_axis_holo.holography.holography_helpers import unpad_arr


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
            FS, FB = prop.spectrum(S), prop.spectrum(B)
            t = prop.ratio_from_spectra(FS, FB, z)

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

        assert isinstance(upsample, (int, np.integer)) and upsample >= 1, "upsample must be an integer >= 1."
        self.shape = (int(M), int(N))
        self.wl = wl
        self.paraxial = paraxial
        self.up = int(upsample)
        self.sz = sz / self.up              # pixel size of the returned fields
        self._nb = nb
        self._threads = threads
        self._dtype = dtype

        prop_cls = ParaxialAngularSpectrum if paraxial else AngularSpectrum
        self._prop = prop_cls(M, N, wl, sz, nb=nb, threads=threads, dtype=dtype, **kwargs)
        self._k = 0                          # current stack size of self._prop (0 = unstacked)
        self._ifft_up = None                 # inverse transform on the upsampled grid
        self._k_up = None                    # its stack size
        self._pad_shape = (int(self._prop.input_shape[0]), int(self._prop.input_shape[1]))
        self._up_shape = (self._pad_shape[0] * self.up, self._pad_shape[1] * self.up)

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

        kS, kB = self._stack_size(S), self._stack_size(B)
        if kB not in (0, kS):
            raise ValueError("B must be a single field or have the same stack size as S.")
        return self.ratio_from_spectra(self.spectrum(S), self.spectrum(B), z, eps=eps)

    def _stack_size(self, a):
        """0 for a single (M, N) array, K for a (K, M, N) stack."""

        a = np.asarray(a)
        if a.shape[-2:] != self.shape or a.ndim not in (2, 3):
            raise ValueError(f"Expected shape {self.shape} or (K, {self.shape[0]}, {self.shape[1]}), "
                             f"got {a.shape}.")
        return 0 if a.ndim == 2 else a.shape[0]

    def _set_stack(self, k):
        """Switch the propagator's transforms to stack size k (0 = unstacked)."""

        if k == self._k:
            return
        if k == 0:
            self._prop.unstack_arrays()
        else:
            self._prop.stack_arrays(int(k))
        self._k = k

    def _kernel(self, z, k):
        """Transfer function for scalar z, or one z per stacked array."""

        z = np.asarray(z, dtype=float)
        if z.ndim == 0:
            return self._prop._kernel(float(z))
        if k == 0 or z.shape != (k,):
            raise ValueError("z must be a scalar or have one value per stacked array.")
        return self._prop._kernel(z[:, None, None])

    def _back(self, F):
        """
        Inverse transform a (fftshift-ed, padded) spectrum, upsampling if requested, and
        remove padding.
        """

        k = 0 if F.ndim == 2 else F.shape[0]
        if self.up == 1:
            self._set_stack(k)
            return self._prop.backwards(F)

        if self._k_up != k:
            self._ifft_up = DiscreteTransform(self._up_shape, 'FFTW_BACKWARD', self._dtype,
                                              threads=self._threads, nstack=int(k))
            self._k_up = k
        P = self._pad_shape
        G = np.zeros(F.shape[:-2] + self._up_shape, dtype=F.dtype)
        # keep DC (index P//2 after fftshift) at the centre of the larger grid
        oy = (self.up * P[0]) // 2 - P[0] // 2
        ox = (self.up * P[1]) // 2 - P[1] // 2
        G[..., oy:oy + P[0], ox:ox + P[1]] = F
        # DFT.forwards already applies the 1/N scaling (normalise_idft=False), so the
        # inverse here is unscaled, as in DFT.backwards
        u = self._ifft_up(np.fft.ifftshift(G, axes=(-2, -1)), ortho=False, normalise_idft=False)
        b = self._nb * self.up
        return u[..., b:u.shape[-2] - b, b:u.shape[-1] - b] if b > 0 else u

    def spectrum(self, u):
        """
        :param u: Field of shape (M, N) or stack (K, M, N).
        :type u: ndarray
        :return: Padded, fftshift-ed spectrum (same stacking as u).
        :rtype: ndarray
        """

        self._set_stack(self._stack_size(u))
        return self._prop.forwards(u)

    def propagate_spectrum(self, F, z):
        """
        :param F: Spectrum from spectrum().
        :type F: ndarray
        :param z: Distance to propagate: scalar, or one value per stacked array.
        :type z: float or ndarray
        :return: P_z{u}.
        :rtype: ndarray
        """

        k = 0 if F.ndim == 2 else F.shape[0]
        return self._back(F * self._kernel(z, k))

    def propagate(self, u, z):
        """
        :param u: Field of shape (M, N) or stack (K, M, N).
        :type u: ndarray
        :param z: Distance to propagate: scalar, or one value per stacked array.
        :type z: float or ndarray
        :return: P_z{u}.
        :rtype: ndarray
        """

        return self.propagate_spectrum(self.spectrum(u), z)

    def ratio_from_spectra(self, FS, FB, z, eps=1e-3):
        """
        t(z) = P_z{S} / P_z{B} from precomputed spectra. FB may be a single
        spectrum shared by every array of a stacked FS (common background).

        :param FS: Spectrum of S from spectrum().
        :type FS: ndarray
        :param FB: Spectrum of B from spectrum().
        :type FB: ndarray
        :param z: Distance to propagate: scalar, or one value per stacked array.
        :type z: float or ndarray
        :param eps: Regularisation relative to mean |B|^2 (per array). Default is 1e-3.
        :type eps: float
        :return: t(z).
        :rtype: ndarray
        """

        k = 0 if FS.ndim == 2 else FS.shape[0]
        H = self._kernel(z, k)
        s = self._back(FS * H)
        if FB.ndim == 2 and k > 0 and np.ndim(z) == 0:
            b = self._back(FB * H)[None]                   # shared background, same z
        else:
            if FB.ndim == 2 and k > 0:
                FB = np.broadcast_to(FB, FS.shape)
            b = self._back(FB * H)
        b2 = np.abs(b) ** 2
        return s * np.conj(b) / (b2 + eps * b2.mean(axis=(-2, -1), keepdims=True))
