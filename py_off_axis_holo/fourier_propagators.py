from abc import ABC, abstractmethod
import numpy as np

from py_off_axis_holo.discrete_transforms import DFT


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

        super().__init__((M, N), nb, threads=threads, dtype=dtype, **kwargs)

        # Frequency coordinates on padded grid
        fx = np.fft.fftshift(np.fft.fftfreq(self.input_shape[1], sz))
        fy = np.fft.fftshift(np.fft.fftfreq(self.input_shape[0], sz))

        FX = fx.reshape(1, self.input_shape[1])
        FY = fy.reshape(self.input_shape[0], 1)
        self._kz = 2*np.pi * np.sqrt(1.0 - (wl * FX) ** 2 - (wl * FY) ** 2) / wl

    def _kernel(self, z, evanescent=True):
        """
        Creates angular spectrum transfer function

        :param z: Distance to propagate.
        :type z: float
        :return: The transfer function.
        :rtype: ndarray
        """

        kz = self._kz if evanescent else np.maximum(self._kz, 0)
        return np.exp(1j * kz * z)
