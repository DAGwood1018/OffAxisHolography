import numpy as np
from py_off_axis_holo.branch_cut_annealing import make_branch_cuts, draw_branch_cut
from py_off_axis_holo.discrete_transforms import DFT
from py_off_axis_holo.holography_helpers import wrap_to_pi, discrete_residue

#temp function
def visualize_branch_cuts(pole_matrix):
    pole_matrix = np.pad(pole_matrix, pad_width=1, mode='constant', constant_values=np.nan)
    branch_cuts, branch_points = make_branch_cuts(pole_matrix)
    branch_cut_matrix = np.zeros(pole_matrix.shape)
    for cut in branch_cuts:
        x1, y1, x2, y2 = cut[0, 0], cut[0, 1], cut[1, 0], cut[1, 1]
        points = draw_branch_cut(x1, y1, x2, y2)
        for point in list(points):
            branch_cut_matrix[point[0], point[1]] = 1
    return branch_cut_matrix

class BranchCutUnwrap:

    def __init__(self):
        """
        Initialize with annealing parameters
        """

        self._cooling_schedule = None

    def __call__(self, wrapped_phase):
        residues = np.pad(discrete_residue(wrapped_phase), pad_width=1, mode='constant', constant_values=np.nan)
        branch_cuts, branch_points = make_branch_cuts(residues)

        '''
        Optimize branch cuts
        '''

        '''
        Find path to traverse and unwrap
        '''

        return branch_cuts, branch_points



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