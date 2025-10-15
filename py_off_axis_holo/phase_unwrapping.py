import numpy as np
from branch_cut_annealing import make_branch_cuts
from py_off_axis_holo.discrete_transforms import DFT
from py_off_axis_holo.holography_helpers import wrap_to_pi, discrete_residue

#temp function
def visualize_branch_cuts(pole_matrix):
    branch_cuts, branch_points = make_branch_cuts(pole_matrix)
    branch_cut_matrix = np.zeros(pole_matrix.shape)
    for cut in branch_cuts:
        x1, y1, x2, y2 = cut[0, 0], cut[0, 1], cut[1, 0], cut[1, 1]
        points = bresenham_line(x1, y1, x2, y2)
        for point in points:
            branch_cut_matrix[point[0], point[1]] = 1
    return branch_cut_matrix

class BranchCutUnwrap:

    def __init__(self):
        '''
        Initialize with annealing parameters
        '''
        self._cooling_schedule = None

    def __call__(self, wrapped_phase):
        residues = discrete_residue(wrapped_phase)
        branch_cuts, branch_points = make_branch_cuts(residues)

        '''
        Optimize branch cuts
        '''

        '''
        Find path to traverse and unwrap
        '''

        unwrapped_phase = None
        return unwrapped_phase

    @classmethod
    def line_high(cls, p1, p2):
        assert len(p1) == 2, "p1 is not a 2d coordinate."
        assert len(p2) == 2, "p2 is not a 2d coordinate."

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        xi = 1
        if dx < 0:
            xi = -1
            dx *= -1
        x = p1[0]
        D = 2 * dx - dy

        points = []
        for y in range(p1[1], p2[1]):
            points.append(np.array([x, y]))
            if D > 0:
                x += xi
                D += 2 * (dx - dy)
            else:
                D += 2 * dx
        return np.array(points, dtype='int32')

    @classmethod
    def line_low(cls, p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        yi = 1
        if dy < 0:
            yi = -1
            dy *= -1
        y = p1[1]
        D = 2 * dy - dx

        points = []
        for x in range(p1[0], p2[0]):
            points.append(np.array([x, y]))
            if D > 0:
                y += yi
                D += -2 * (dx - dy)
            else:
                D += 2 * dy
        return np.array(points, dtype='int32')

    @classmethod
    def pxls_on_branch_cut(cls, x1, y1, x2, y2):
        """
        This implements Bresenham's line algorithm.
        """

        A = np.array([x1, y1], dtype='int32')
        B = np.array([x2, y2], dtype='int32')
        abs_diff = np.abs(B - A)
        if abs_diff[1] < abs_diff[0]:
            if A[0] > B[0]:
                start, end = B, A
            else:
                start, end = A, B
            line_points = cls.line_low(start, end)
        else:
            if A[1] > B[1]:
                start, end = B, A
            else:
                start, end = A, B
            line_points = cls.line_high(start, end)
        return line_points

#======================================================================================================================#

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