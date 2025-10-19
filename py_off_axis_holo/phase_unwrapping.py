import numpy as np
from py_off_axis_holo.branch_cut_annealing import (make_branch_cuts, branch_cut_dists, init_temperature,
                                                   draw_branch_cut, heat_init_state, anneal_branch_cuts, check_for_pole_repeats)
from py_off_axis_holo.discrete_transforms import DFT
from py_off_axis_holo.holography_helpers import wrap_to_pi, discrete_residue

#temp function
def visualize_branch_cuts(wrapped_phase):
    branch_cut_unwrapper = BranchCutUnwrap()
    branch_cuts, T, E = branch_cut_unwrapper(wrapped_phase)
    branch_cut_matrix = np.zeros((wrapped_phase.shape[0]+1, wrapped_phase.shape[1]+1))
    pole_coords = branch_cut_unwrapper._pole_coords
    for cut in np.argwhere(branch_cuts == 1):
        p1, p2 = cut[0], cut[1]
        x1, y1, x2, y2 = pole_coords[cut[0], 0], pole_coords[cut[0], 1], pole_coords[cut[1], 0], pole_coords[cut[1], 1]
        points = draw_branch_cut(x1, y1, x2, y2)
        for point in list(points):
            branch_cut_matrix[point[0], point[1]] = 1
    return branch_cut_matrix, T, E

class BranchCutUnwrap:

    def __init__(self, heating_cycles=100, cooling_cycles=10000, n_samples=1000, heating_rate=1.1, cooling_rate=0.8):
        """
        Initialize with annealing parameters
        """

        self._sample_size = n_samples
        self._heating_cycles = heating_cycles
        self._cooling_cycles = cooling_cycles
        self._pos_poles = np.array([], dtype='int32')
        self._neg_poles = np.array([], dtype='int32')
        self._virtual_poles = np.array([], dtype='int32')
        self._pole_coords = np.array([[]], dtype='int32')
        self._pole_vals = np.array([], dtype='float64')
        self._heating_rate = heating_rate
        self._cooling_rate = cooling_rate

    def __call__(self, wrapped_phase):
        residues = np.pad(discrete_residue(wrapped_phase), pad_width=1, mode='constant', constant_values=np.nan)
        branch_points = self._identify_branch_points(residues)
        if len(branch_points['-']) == 0 and len(branch_points['+']) == 0:
            raise RuntimeError("There is no need to make branch cuts")

        # Index to pole coord and val.
        pole_coords, pole_vals = [], []
        self._neg_poles = np.array(range(len(branch_points['-'])), dtype='int32')
        for i, neg_pole in enumerate(branch_points['-']):
            pole_coords.append(tuple(neg_pole))
            pole_vals.append(-1)
        self._pos_poles = np.array(range(len(branch_points['-'])), dtype='int32') + len(pole_vals)
        for i, pos_pole in enumerate(branch_points['+']):
            pole_coords.append(tuple(pos_pole))
            pole_vals.append(1)
        self._virtual_poles = np.array(range(len(branch_points['i'])), dtype='int32') + len(pole_vals)
        for i, virtual_pole in enumerate(branch_points['i']):
            pole_coords.append(tuple(virtual_pole))
            pole_vals.append(np.nan)
        self._pole_coords = np.array(pole_coords, dtype='int32')
        self._pole_vals = np.array(pole_vals, dtype='float64')
        dist_matrix = branch_cut_dists(self._neg_poles, self._pos_poles, self._virtual_poles, self._pole_coords)
        branch_cuts = make_branch_cuts(self._neg_poles, self._pos_poles, self._pole_vals, dist_matrix)

        branch_cuts, T, E = self._reverse_annealing(branch_cuts, dist_matrix)
        '''
        Find path to traverse and unwrap
        '''

        return branch_cuts, T, E

    def _identify_branch_points(self, pole_matrix):
        # Get coords of residues

        neg_coords = np.argwhere(pole_matrix == -1)
        pos_coords = np.argwhere(pole_matrix == 1)
        virtual_poles = np.argwhere(np.isnan(pole_matrix))

        # If no residues stop
        if len(neg_coords) == 0 and len(pos_coords) == 0:
            return {}

        # Label all points based on type
        branch_points = {'-': {tuple(coord) for coord in neg_coords}, '+': {tuple(coord) for coord in pos_coords},
                         'i': {tuple(coord) for coord in virtual_poles}}
        return branch_points

    def _reverse_annealing(self, branch_cuts, dist_matrix):
        mask = np.full(dist_matrix.shape, 0, dtype='float64')
        mask[0:len(self._neg_poles) + len(self._pos_poles), 0:len(self._neg_poles) + len(self._pos_poles)] = np.inf
        virtual_pole_dists = dist_matrix + mask

        T_init = init_temperature(self._sample_size, self._neg_poles, self._pos_poles, branch_cuts,
                                  self._pole_coords, self._pole_vals, virtual_pole_dists)

        branch_cuts, T1, E1 = heat_init_state(self._heating_cycles, T_init, self._neg_poles, self._pos_poles,
                                                         branch_cuts, self._pole_coords, self._pole_vals, virtual_pole_dists, self._heating_rate)
        #branch_cuts, T2, E2 = anneal_branch_cuts(self._cooling_cycles, T_init, self._neg_poles, self._pos_poles,
        #                                                 branch_cuts, self._pole_coords, self._pole_vals, virtual_pole_dists, self._cooling_rate)

        return branch_cuts, T1, E1 #np.concatenate((T1, T2)), np.concatenate((E1, E2))



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