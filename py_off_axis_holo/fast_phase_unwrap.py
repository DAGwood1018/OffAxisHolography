import numpy as np

from py_off_axis_holo.discrete_transforms import DFT


class FastPhaseUnwrap(DFT):

    def __init__(self, M, N, nb=0, threads=1, dtype='complex128', **kwargs):
        """
        Based off of method explained in -> doi: 10.1364/BOE.10.001365
        """

        super().__init__((M, N), nb, threads=threads, dtype=dtype, **kwargs)
        kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(M, d=1.0))
        ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=1.0))
        self._KX, self._KY = np.meshgrid(kx, ky, indexing='ij')

    def __call__(self, phase):
        P = np.exp(1j*phase)
        FT1 = self.forwards(np.fft.fftshift(P))
        K2 = self._KX**2 + self._KY**2
        IFT1 = np.fft.ifftshift(self.backwards(K2*FT1))
        Im = np.imag(IFT1/P)
        FT2 = self.forwards(np.fft.fftshift(Im))
        psi_est = np.fft.ifftshift(self.backwards(FT2/K2))
        Q = np.round(psi_est-phase)
        return phase + Q