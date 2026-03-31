import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Iterable
from py_off_axis_holo.holography_helpers import freqspace
from py_off_axis_holo.discrete_transforms import DFT


def prop_angular_spectrum(field, z, wl, sz, **kwargs):
    """
    Performs propagation via angular spectrum method. Consistent units assumed.

    :param field: Complex field to propagate.
    :type field: ndarray<complex>
    :param z: Propagation distance.
    :type z: float
    :param wl: Wavelength of light.
    :type wl: float
    :param sz: Size of each grid point.
    :type sz: float
    :param kwargs: Optional arguments for AngularSpectrum class object
    :return: Propagated field.
    :rtype: ndarray<complex>
    """

    Nx, Ny = field.shape
    prop = AngularSpectrum(Nx, Ny, wl, sz, **kwargs)
    return prop(field, z)

class Propagate(DFT, ABC):

    def __init__(self, M, N, wl, sz, nb=0, threads=1, dtype='complex128', **kwargs):
        """
        :param M, N: Shape of 2D arrays to operate on.
        :type M, N: int
        :param wl: Wavelength of light.
        :type wl: float
        :param sz: Size of each grid point. Assumes sames dimension as wavelength.
        :type sz: float
        :param nb: Padding to add to each array axis. Default is '0'.
        :type nb: int
        :param threads: Number of threads to use. Default is '1'.
        :type threads: int
        :param dtype: Numpy dtype (complex) of arrays to operate on. Default is 'complex128'.
        :type dtype: string
        :param kwargs: Optional arguments for DFT object.
        """

        super().__init__((M, N), nb, threads=threads, dtype=dtype, ortho=False, **kwargs)
        self._Fx, self._Fy = freqspace(N, M, sz, nb)
        self._wl = wl
        self._sz = sz

    def __call__(self, field, z):
        if not isinstance(z, Iterable) and self._stacked:
            z = np.array([z])
        else:
            z = np.array(z)
        Fh = self.forwards(field)
        return self.backwards(Fh * self.propagator(z))

    @abstractmethod
    def propagator(self, z):
        ...


class AngularSpectrum(Propagate):

    def propagator(self, z):
        """
        :param z: Propagation distance.
        :type z: float
        :return: Angular spectrum field propagator.
        :rtype: ndarray<complex>
        """

        fx, fy = self._Fx, self._Fy
        k0 = 2 * np.pi / self._wl
        kz = k0 * np.sqrt((1 - (self._wl * fx)**2 - (self._wl * fy)**2))

        if self._stacked:
            return np.exp(1j * kz[None, :, :] * z[:, None, None])
        else:
            return np.exp(1j * kz * z)

