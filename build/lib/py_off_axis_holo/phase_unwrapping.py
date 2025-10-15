import numpy as np
from numba import njit
from py_off_axis_holo.discrete_transforms import DFT
from py_off_axis_holo.holography_helpers import wrap_to_pi

@njit("float64(int32, int32, int32, int32)")
def dist(x1, y1, x2, y2):
    a, b = np.array([x1, y1], dtype='float64'), np.array([x2, y2], dtype='float64')
    return np.linalg.norm(a - b)

@njit("float64(int32[:, :, :])")
def branch_cut_cost(branch_cuts):
    cost = 0
    for cut in branch_cuts:
        x1, y1, x2, y2 = cut[0, 0], cut[0, 1], cut[1, 0], cut[1, 1]
        cost += dist(x1, y1, x2, y2)
    return cost

def bresenham_line(x1, y1, x2, y2):
    def line_low(p1, p2):
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

    def line_high(p1, p2):
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

    A = np.array([x1, y1], dtype='int32')
    B = np.array([x2, y2], dtype='int32')
    abs_diff = np.abs(B - A)
    if abs_diff[1] < abs_diff[0]:
        if A[0] > B[0]:
            start, end = B, A
        else:
            start, end = A, B
        line_points = line_low(start, end)
    else:
        if A[1] > B[1]:
            start, end = B, A
        else:
            start, end = A, B
        line_points = line_high(start, end)
    return line_points

def branch_cut_dists(branch_points, point_to_idx):
    n = len(point_to_idx.keys())
    dist_matrix = np.full((n, n), np.inf)

    # Fill distance matrix valid pairs
    for p1 in branch_points['-']:
        for p2 in branch_points['+']:
            i, j = point_to_idx[tuple(p1)], point_to_idx[tuple(p2)]
            d = dist(p1[0], p1[1], p2[0], p2[1])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    for p1 in branch_points['-']:
        for p2 in branch_points['b']:
            i, j = point_to_idx[tuple(p1)], point_to_idx[tuple(p2)]
            d = dist(p1[0], p1[1], p2[0], p2[1])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    for p1 in branch_points['+']:
        for p2 in branch_points['b']:
            i, j = point_to_idx[tuple(p1)], point_to_idx[tuple(p2)]
            d = dist(p1[0], p1[1], p2[0], p2[1])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix

def identify_branch_points(pole_matrix):
    # Get coords of residues
    neg_coords = np.argwhere(pole_matrix == -1)
    pos_coords = np.argwhere(pole_matrix == 1)
    border_coords = np.argwhere(pole_matrix == np.nan)

    # If no residues stop
    if len(neg_coords) == 0 and len(pos_coords) == 0:
        return {}

    # Label all points based on type
    branch_points = {'-': {tuple(coord) for coord in neg_coords}, '+': {tuple(coord) for coord in pos_coords},
                     'b': {tuple(coord) for coord in border_coords}}
    return branch_points

def make_branch_cuts(pole_matrix):
    branch_points = identify_branch_points(pole_matrix)
    if len(branch_points['-']) == 0 and len(branch_points['+']) == 0:
        raise RuntimeError("There is no need to make branch cuts")

    # Point-to-index maps
    point_to_idx, idx_to_point = {}, {}
    for i, neg_pole in enumerate(branch_points['-']):
        point_to_idx[tuple(neg_pole)] = i
        idx_to_point[i] = tuple(neg_pole)
    n = len(point_to_idx.keys())
    for i, pos_pole in enumerate(branch_points['+']):
        point_to_idx[tuple(pos_pole)] = i + n
        idx_to_point[i+n] = tuple(pos_pole)
    n = len(point_to_idx.keys())
    for i, border_point in enumerate(branch_points['b']):
        point_to_idx[tuple(border_point)] = i + n
        idx_to_point[i+n] = tuple(border_point)
    dist_matrix = branch_cut_dists(branch_points, point_to_idx)

    # Border points will always remain available b/c the assumption is that there is some matching pole outside the
    # boundary not that the border point is the pole.
    available_poles = branch_points['-'].union(branch_points['+'])
    availability_mask = np.full((n, n), 0, dtype='float64')

    # Iterate through poles at random.
    rng = np.random.default_rng()
    poles = rng.permutation(list(available_poles))  # random iteration order

    # Find heuristic solution by making branch cuts with a greedy nearest neighbor search.
    branch_cuts = []
    for p1 in poles:
        p1 = tuple(p1)
        if p1 in available_poles:
            i = point_to_idx[p1]
            masked_dist_matrix = dist_matrix + availability_mask
            dists = masked_dist_matrix[i, :]

            j = np.argmin(dists)
            p2 = tuple(idx_to_point[j])
            branch_cuts.append((p1, p2))

            # Remove used points from availability
            available_poles.discard(p1)
            availability_mask[i, :], availability_mask[:, i] = np.inf, np.inf
            if p2 in branch_points['b']:
                continue
            available_poles.discard(p2)
            availability_mask[j, :], availability_mask[:, j] = np.inf, np.inf

    return np.array(branch_cuts, dtype='int32'), branch_points

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