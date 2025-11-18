import numpy as np
from abc import ABC, abstractmethod
from copy import copy
from py_off_axis_holo.holography_helpers import freqspace
from py_off_axis_holo.discrete_transforms import DFT


def fresnel(field, z, wl, sz, **kwargs):
    """
    Performs Fresnel propagation. Consistent units assumed.

    :param field: Complex field to propagate.
    :type field: ndarray<complex>
    :param z: Propagation distance.
    :type z: float
    :param wl: Wavelength of light.
    :type wl: float
    :param sz: Size of each grid point.
    :type sz: float
    :param kwargs: Optional arguments for Frensel class object
    :return: Propagated field.
    :rtype: ndarray<complex>
    """

    Nx, Ny = field.shape
    prop = Fresnel(Nx, Ny, wl, sz, **kwargs)
    return prop(field, z)


def angular_spectrum(field, z, wl, sz, **kwargs):
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


def rep_propagation(field, dists, wl, sz, method='Fresnel', **kwargs):
    """
    Performs numerical propagation independently for a number of distances.

    :param field: Complex field to propagate.
    :type field: ndarray<complex>
    :param dists: Distances to propagate to.
    :type dists: list<float>
    :param wl: Wavelength of light.
    :type wl: float
    :param sz: Size of each grid point. Assumes sames dimension as wavelength.
    :type sz: float
    :param method: Method to use. If not 'Fresnel' then defaults to 'AngularSpectrum'.
    :type method: str
    :param kwargs: Method specific optional arguments.
    :return: Distances and propagated fields.
    :rtype: list<float>, list<ndarray>
    """

    Nx, Ny = field.shape
    prop = Fresnel(Nx, Ny, wl, sz, **kwargs) if method == 'Fresnel' else AngularSpectrum(Nx, Ny, wl, sz, **kwargs)

    recs = []
    for i, dist in enumerate(dists):
        f = copy(prop(field, dist))
        recs.append(f)
    return dists, recs


def chain_propagation(field, dists, wl, sz, method='Fresnel', direction=1, **kwargs):
    """
    Performs numerical propagation sequentially in a direction for a number of distances.

    :param field: Complex field to propagate.
    :type field: ndarray<complex>
    :param dists: Distances to propagate to.
    :type dists: list<float>
    :param wl: Wavelength of light.
    :type wl: float
    :param sz: Size of each grid point. Assumes sames dimension as wavelength.
    :type sz: float
    :param direction: Direction to propagate. +1 for z or -1 for -z. Default is '1'.
    :rtype direction: int
    :param method: Method to use. If not 'Fresnel' then defaults to 'AngularSpectrum'.
    :type method: str
    :param kwargs: Method specific optional arguments.
    :return: Distances and propagated fields.
    :rtype: list<float>, list<ndarray>
    """

    dists = set(np.abs(dists))
    direction = 1 if direction >= 0 else -1
    dists = direction * np.sort(list(dists))

    Nx, Ny = field.shape
    prop = Fresnel(Nx, Ny, wl, sz, **kwargs) if method == 'Fresnel' else AngularSpectrum(Nx, Ny, wl, sz, **kwargs)

    d0 = 0
    recs = []
    for dist in dists:
        field = prop(field, dist - d0)
        d0 = dist
        recs.append(field)
    return dists, recs


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
        self._k0 = 2 * np.pi / wl
        self._sz = sz

    def __call__(self, field, z):
        return self.propagate(field, z)

    @abstractmethod
    def propagate(self, field, z):
        ...


class Fresnel(Propagate):

    def propagate(self, field, z):
        """
        Performs Fresnel propagation via FFT. Note the paraxial approximation to the
        Helmholtz wave equation is used.

        :param field: Complex field to propagate.
        :type field: ndarray<complex>
        :param z: Propagation distance.
        :type z: float
        :return: Propagated field.
        :rtype: ndarray<complex>
        """

        kx, ky = 2 * np.pi * self._Fx, 2 * np.pi * self._Fy
        Fh = self.forwards(field)
        H = np.exp(1j * z * self._k0) * np.exp(-1j * z * (kx ** 2 + ky ** 2) / (2 * self._k0))
        if self._stacked:
            H = np.stack([H for _ in range(self.output_shape[0])])
        return self.backwards(Fh * H)


class AngularSpectrum(Propagate):

    def propagate(self, field, z):
        """
        Performs angular spectrum method via FFT. Note the paraxial approximation to the
        Helmholtz wave equation is used.

        :param field: Complex field to propagate.
        :type field: ndarray<complex>
        :param z: Propagation distance.
        :type z: float
        :return: Propagated field.
        :rtype: ndarray<complex>
        """

        kx, ky = 2 * np.pi * self._Fx, 2 * np.pi * self._Fy
        Fh = self.forwards(field)
        kz = self._k0 - (kx ** 2 + ky ** 2) / (2 * self._k0)
        H = np.exp(1j * z * kz)
        if self._stacked:
            H = np.stack([H for _ in range(self.output_shape[0])])
        return self.backwards(Fh * H)
