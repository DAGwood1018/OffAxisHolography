import numpy as np
from py_off_axis_holo.discrete_transforms import DFT


def wrap_to_pi(phi):
    phi = phi - 2 * np.pi * np.floor((phi + np.pi) / (2 * np.pi))
    return phi

def discrete_residue(F):
    """
    Vectorized computation of:
    np.round(((F[i, j+1]-F[i,j]) + (F[i+1,j+1]-F[i,j+1]) +
              (F[i+1,j]-F[i+1,j+1]) + (F[i,j]-F[i+1,j])) / (2*np.pi))
    """
    F = np.asarray(F)

    # compute the terms for interior points
    term1 = F[:-1, 1:] - F[:-1, :-1]
    term2 = F[1:, 1:] - F[:-1, 1:]
    term3 = F[1:, :-1] - F[1:, 1:]
    term4 = F[:-1, :-1] - F[1:, :-1]

    val = (term1 + term2 + term3 + term4) / (2 * np.pi)
    res = np.zeros_like(F, dtype=float)
    res[:-1, :-1] = np.round(val)
    return res

class BranchCut:

    def __init__(self, x1, y1, x2, y2):
        assert x1 is int, "x1 must be an integer."
        assert y1 is int, "y1 must be an integer."
        assert x2 is int, "x2 must be an integer."
        assert y2 is int, "y2 must be an integer."

        start = np.array([x1, y1])
        end = np.array([x2, y2])
        abs_diff = np.abs(end - start)
        if abs_diff[1] < abs_diff[0]:
            if start[0] > end[0]:
                self._start = end
                self._end = start
            else:
                self._start = start
                self._end = end
            self._points = self._line_low()
        else:
            if start[1] > end[1]:
                self._start = end
                self._end = start
            else:
                self._start = start
                self._end = end
            self._points = self._line_high()

    def __call__(self):
        return self._points

    @property
    def length(self):
        return np.sqrt(np.sum((self._start-self._end)**2))

    def _line_high(self):
        dx = self._end[0] - self._start[0]
        dy = self._end[1] - self._start[1]
        xi = 1
        if dx < 0:
            xi = -1
            dx *= -1
        x = self._start[0]
        D = 2*dx-dy

        points = []
        for y in range(self._start[1], self._end[1]):
            points.append(np.array([x, y]))
            if D > 0:
                x += xi
                D += 2*(dx-dy)
            else:
                D += 2*dx
        return points

    def _line_low(self):
        dx = self._end[0] - self._start[0]
        dy = self._end[1] - self._start[1]
        yi = 1
        if dy < 0:
            yi = -1
            dy *= -1
        y = self._start[1]
        D = 2*dy-dx

        points = []
        for x in range(self._start[0], self._end[0]):
            points.append(np.array([x, y]))
            if D > 0:
                y += yi
                D += -2*(dx-dy)
            else:
                D += 2*dy
        return points

class PhaseUnwrapTIE(DFT):
    """
    This class is adapted from the method described in https://doi.org/10.1088/1361-6501/aaec5c.
    The author's Matlab implementation can be found here https://www.mathworks.com/matlabcentral/fileexchange/68493-robust-2d-phase-unwrapping-algorithm.
    """

    def __init__(self, Nx, Ny, nb=1, threads=1, dtype='float64', zero=False, **kwargs):
        super().__init__((Nx, Ny), nb, threads=threads, ortho=True, re=True, dtype=dtype, **kwargs)
        self._zero = zero

    def __call__(self, phase_wrap):
        phi1 = self.unwrap(phase_wrap)
        phi1 += np.mean(phase_wrap) - np.mean(phi1)  # adjust piston
        K1 = np.round((phi1 - phase_wrap) / (2 * np.pi))  # calculate integer K
        phase_unwrap = phase_wrap + 2 * K1 * np.pi
        residue = wrap_to_pi(phase_unwrap - phi1)

        phi1 += self.unwrap(residue)
        phi1 += np.mean(phase_wrap) - np.mean(phi1)  # adjust piston
        K2 = np.round((phi1 - phase_wrap) / (2 * np.pi))  # calculate integer K
        phase_unwrap = phase_wrap + 2 * K2 * np.pi
        residue = wrap_to_pi(phase_unwrap - phi1)

        i = 0
        while np.sum(np.sum(np.abs(K2 - K1))) > 0:
            K1 = K2
            phic = self.unwrap(residue)
            phi1 += phic
            phi1 += np.mean(phase_wrap) - np.mean(phi1)  # adjust piston
            K2 = np.round((phi1 - phase_wrap) / (2 * np.pi))  # calculate integer K
            phase_unwrap = phase_wrap + 2 * K2 * np.pi
            residue = wrap_to_pi(phase_unwrap - phi1)
            i += 1
        return phase_unwrap - phase_unwrap.min() if self._zero else phase_unwrap

    def _solve(self, rho):
        # solve the Poisson equation using DCT
        dct_rho = self.forwards(rho)
        N, M = dct_rho.shape
        I, J = np.meshgrid(np.arange(M), np.arange(N))
        with np.errstate(divide='ignore', invalid='ignore'):
            dct_phi = dct_rho / 2 / (np.cos(np.pi * I / M) + np.cos(np.pi * J / N) - 2)
            dct_phi[0, 0] = 0  # handling the inf/nan value
            # now invert to get the result
        phi = self.backwards(dct_phi)
        return phi

    def unwrap(self, phase_wrap):
        edx = np.concatenate((np.zeros((phase_wrap.shape[0], 1)), wrap_to_pi(np.diff(phase_wrap, axis=1, n=1)),
                              np.zeros((phase_wrap.shape[0], 1))), axis=1)
        edy = np.concatenate((np.zeros((1, phase_wrap.shape[1])), wrap_to_pi(np.diff(phase_wrap, axis=0, n=1)),
                              np.zeros((1, phase_wrap.shape[1]))), axis=0)
        lap = np.diff(edx, axis=1, n=1) + np.diff(edy, axis=0, n=1)  # calculate Laplacian using the finite difference
        return self._solve(lap)