import numpy as np
from off_axis_holography.utilities import reference_wave, object_wave
from off_axis_holography.field_propagation import Propagate

def spherical_surface(X, Y, r):
    X2Y2 = X ** 2 + Y ** 2
    indcs = np.where(X2Y2 <= r ** 2)
    Z = np.zeros(X.shape)
    Z[indcs] = np.sqrt(r ** 2 - X2Y2[indcs])
    return Z

def gaussian_surface(X, Y, mu, sigma, h=1):
    Z = np.exp(-(X - mu[0])**2 / (2 * sigma[0]**2) - (Y - mu[1])**2 / (2 * sigma[1]**2))
    return h * Z

def flat_surface(X, Y, H, xrange, yrange):
    Z = np.zeros(X.shape)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if xrange[0] <= X[i, j] <= xrange[1] and \
                    yrange[0] <= Y[i, j] < yrange[1]:
                Z[i, j] = 1
    return H * Z


class DigiHoloModel(Propagate):

    def __init__(self, M, N, d, tilt, wl, sz, nb=1, threads=1, dtype='complex128', **kwargs):
        super().__init__(M, N, wl, sz, nb=nb, threads=threads, dtype=dtype, **kwargs)
        self._tilt = tilt
        self._z = d

    def __call__(self, phi, *args):
        #assert callable(obj), "Height profile of object must be callable."
        x2, y2 = self._image_plane()
        #phase = self._k0 * n * obj(x2, y2, *args)
        psi = object_wave([x2, y2], phi, self._k0, self._sz)

        kx, ky = tuple(self._k0 * self._sz * np.array([self._Fx, self._Fy]) / self._z)
        E = np.exp(-1j * self._k0 * self._z) * \
            self.backwards(self.forwards(psi) * np.exp(1j * self._z / (2 * self._k0) * (kx ** 2 + ky ** 2)))
        return self._interferogram(E)

    def _interferogram(self, obj):
        ref = reference_wave((self._X, self._Y), self._tilt, self._k0, self._sz)
        return np.abs(obj + ref) ** 2

    def extract_hologram(self, E):
        R = reference_wave((self._X, self._Y), self._tilt, self._k0, self._sz)
        E = E.astype(R.dtype)
        E *= R
        Fh = self.forwards(E)
        k = -self._k0 * np.sin(self._tilt)
        f1 = np.array(E.shape) * self._sz * k / (2 * np.pi)
        r = np.sqrt(np.sum(np.array(f1) ** 2)) / 3
        mask = np.sqrt(self._Fx ** 2 + self._Fy ** 2) < r
        Fh *= mask
        return self.backwards(Fh)

class DigiHolo4fModel(DigiHoloModel):

    def __init__(self, M, N, f1, f2, tilt, wl, sz, threads=1, nb=1, dtype='complex128', **kwargs):
        super().__init__(M, N, 2 * (f1 + f2), tilt, wl, sz, nb=nb, threads=threads, dtype=dtype, **kwargs)
        self._M = - f2 / f1

    def __call__(self, obj, n,  *args):
        assert callable(obj), "Height profile of object must be callable."
        x2, y2 = self._image_plane()
        phase = self._k0 * n * obj(x2, y2, *args)
        psi = object_wave([x2 / self._M, y2 / self._M], phase, self._k0, self._sz)

        E = self._M * np.exp(-1j * self._k0 * self._z) * psi
        return self._interferogram(E)

