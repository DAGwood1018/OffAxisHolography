import numpy as np
from abc import ABC, abstractmethod
from copy import copy
from off_axis_utils.manage_arrays import gridspace
from py_offaxis_holo.discrete_transforms import DFT

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

def rep_propagation(method, field, dists, *args, **kwargs):
    """
    Performs numerical propagation independently for a number of distances.

    :param method: Method to use.
    :type method: Propagate
    :param field: Complex field to propagate.
    :type field: ndarray<complex>
    :param dists: Distances to propagate to.
    :type dists: list<float>
    :param args: Method specific arguments.
    :param kwargs: Method specific optional arguments.
    :return: Distances and propagated fields.
    :rtype: list<float>, list<ndarray>
    """

    assert issubclass(method, Propagate), "Method must implement Propagate abstract class."
    Nx, Ny = field.shape
    prop = method(Nx, Ny, *args, **kwargs)

    recs = []
    for i, dist in enumerate(dists):
        f = copy(prop(field, dist))
        recs.append(f)
    return dists, recs

def chain_propagation(method, field, dists, *args, direction=1, **kwargs):
    """
    Performs numerical propagation sequentially in a direction for a number of distances.

    :param method: Method to use.
    :type method: Propagate
    :param field: Complex field to propagate.
    :type field: ndarray<complex>
    :param dists: Distances to propagate to.
    :type dists: list<float>
    :param args: Method specific arguments.
    :param direction: Direction to propagate. +1 for z or -1 for -z. Default is '1'.
    :rtype direction: int
    :param kwargs: Method specific optional arguments.
    :return: Distances and propagated fields.
    :rtype: list<float>, list<ndarray>
    """

    assert issubclass(method, Propagate), "Method must implement Propagate abstract class."
    dists = set(np.abs(dists))
    direction = 1 if direction >= 0 else -1
    dists = direction * np.sort(list(dists))

    Nx, Ny = field.shape
    prop = method(Nx, Ny, *args, **kwargs)

    d0 = 0
    recs = []
    for dist in dists:
        field = prop(field, dist - d0)
        d0 = dist
        recs.append(field)
    return dists, recs

class Propagate(DFT, ABC):

    def __init__(self, M, N, wl, sz, nb=1, threads=1, dtype='complex128', **kwargs):
        """
        :param M, N: Shape of 2D arrays to operate on.
        :type M, N: int
        :param wl: Wavelength of light.
        :type wl: float
        :param sz: Size of each grid point. Assumes sames dimension as wavelength.
        :type sz: float
        :param nb: Padding factor to use. i.e. nb * dims = padded array shape. Default is '1'.
        :type nb: float
        :param threads: Number of threads to use. Default is '1'.
        :type threads: int
        :param dtype: Numpy dtype (complex) of arrays to operate on. Default is 'complex128'.
        :type dtype: string
        :param kwargs: Optional arguments for DFT object.
        """

        super().__init__((M, N), nb, threads=threads, dtype=dtype, ortho=False, **kwargs)
        self._X, self._Y = gridspace(N, M, 1, True)
        self._Fx, self._Fy = gridspace(N, M, nb, True)  # Can replace with freqs
        self._k0 = 2 * np.pi / wl
        self._sz = sz

    @abstractmethod
    def __call__(self, field, z):
        ...

    def _image_plane(self):
        """
        Defines length scale of image plane.

        :return: Meshgrids of X and Y
        :rtype: ndarray<float>, ndarray<float>
        """

        x1 = self._X * self._sz
        y1 = self._Y * self._sz
        return x1, y1

    def _object_plane(self, z):
        """
        Defines length scale of object plane given a distance z.

        :param z: Propagation distance. Assumes same units as wavelength.
        :type z: float
        :return: Meshgrids of X and Y
        :rtype: ndarray<float>, ndarray<float>
        """

        dy2, dx2 = tuple(2 * np.pi * z / (np.array(self._dims) * self._sz * self._k0))
        x2 = self._X * dx2
        y2 = self._Y * dy2
        return x2, y2

    def _fourier_plane(self):
        """
        Defines length scale of Fourier plane.

        :return: Meshgrids of X and Y
        :rtype: ndarray<float>, ndarray<float>
        """

        dky, dkx = tuple(2 * np.pi / (np.array(self._dims) * self._sz))
        kx = self._Fx * dkx
        ky = self._Fy * dky
        return kx, ky


class Fresnel(Propagate):

    def __call__(self, field, z):
        """
        Performs Fresnel propagation via FFT. Note that the pixel size of the reconstruction is
        wl * z / ( N * sz ).

        :param field: Complex field to propagate.
        :type field: ndarray<complex>
        :param z: Propagation distance.
        :type z: float
        :return: Propagated field.
        :rtype: ndarray<complex>
        """

        assert field.shape == tuple(self._dims), "Dimensions must match."
        x2, y2 = self._object_plane(z)
        x1, y1 = self._image_plane()

        z_phase = (-1j * self._k0 / (2 * np.pi * z)) * np.exp(1j * self._k0 * z)
        out_phase = np.exp((1j * self._k0 / (2 * z)) * (x2**2 + y2**2))
        in_phase = np.exp((1j * self._k0 / (2 * z)) * (x1**2 + y1**2))
        return z_phase * out_phase * self.forwards(field * in_phase)

class AngularSpectrum(Propagate):

    def __call__(self, field, z):
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

        assert field.shape == tuple(self._dims), "Dimensions must match."
        kx, ky = self._fourier_plane()
        Fh = self.forwards(field)
        kz = self._k0 - (kx**2 + ky**2) / (2 * self._k0)
        H = np.exp(1j * z * kz)
        return self.backwards(Fh * H)