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

    def __init__(self, S, B, wl=633e-9, sz=1.0, nb=128, upsample=1, paraxial=True,
                 threads=1, dtype='complex128', **kwargs):
        """
        Propagates an object field S and a background field B on a
        common grid using AngularSpectrum / ParaxialAngularSpectrum. The spectra
        are computed once, so scanning many planes only costs one inverse FFT per
        field per plane.

        :param S: Object field.
        :type S: ndarray<complex>
        :param B: Background field with the same shape as S.
        :type B: ndarray<complex>
        :param wl: The wavelength.
        :type wl: float
        :param sz: The pixel size of S and B.
        :type sz: float
        :param nb: Padding to use. Default is 128.
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

        assert isinstance(upsample, int) and upsample >= 1, "upsample must be an integer >= 1."
        assert B.shape == S.shape, "S and B must have the same shape."

        M, N = S.shape
        prop_cls = ParaxialAngularSpectrum if paraxial else AngularSpectrum
        self._prop = prop_cls(M, N, wl, sz, nb=nb, threads=threads, dtype=dtype, **kwargs)

        self.wl = wl
        self.paraxial = paraxial
        self.shape = (M, N)
        self.up = upsample
        self.sz = sz / upsample              # pixel size of the returned fields
        self._nb = nb

        # Spectra (fftshift-ed, padded grid) computed once
        self._FS = self._prop.forwards(S)
        self._FB = self._prop.forwards(B)

        # Inverse transform on the upsampled grid
        self._ifft_up = None
        if upsample > 1:
            P = np.array(self._prop.input_shape)
            self._up_shape = tuple(int(p) * upsample for p in P)
            self._ifft_up = DiscreteTransform(self._up_shape, 'FFTW_BACKWARD', dtype,
                                              threads=threads)

    def __call__(self, z, eps=1e-3):
        """
        Object field divided by background field, both propagated to z:
        t(z) = P_z{S} / P_z{B}, regularised where |B| is small.

        :param z: Distance to propagate.
        :type z: float
        :param eps: Regularisation relative to mean |B|^2. Default is 1e-3.
        :type eps: float
        :return: t(z).
        :rtype: ndarray
        """

        H = self._kernel(z)
        s = self._back(self._FS * H)
        b = self._back(self._FB * H)
        b2 = np.abs(b) ** 2
        return s * np.conj(b) / (b2 + eps * b2.mean())

    def _kernel(self, z):
        return self._prop._kernel(z)

    def _back(self, F):
        """
        Inverse transform a (fftshift-ed, padded) spectrum, upsampling if requested, and
        remove padding.
        """

        if self.up == 1:
            return self._prop.backwards(F)

        P = F.shape
        G = np.zeros(self._up_shape, dtype=F.dtype)
        # keep DC (index P//2 after fftshift) at the centre of the larger grid
        oy = (self.up * P[0]) // 2 - P[0] // 2
        ox = (self.up * P[1]) // 2 - P[1] // 2
        G[oy:oy + P[0], ox:ox + P[1]] = F
        # DFT.forwards already applies the 1/N scaling (normalise_idft=False), so the
        # inverse here is unscaled, as in DFT.backwards
        u = self._ifft_up(np.fft.ifftshift(G), ortho=False, normalise_idft=False)
        return unpad_arr(u, self._nb * self.up)

    def object(self, z):
        """
        :param z: Distance to propagate.
        :type z: float
        :return: P_z{S}.
        :rtype: ndarray
        """

        return self._back(self._FS * self._kernel(z))

    def background(self, z):
        """
        :param z: Distance to propagate.
        :type z: float
        :return: P_z{B}.
        :rtype: ndarray
        """

        return self._back(self._FB * self._kernel(z))
