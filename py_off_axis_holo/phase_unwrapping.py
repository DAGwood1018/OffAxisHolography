import numpy as np
from py_off_axis_holo.discrete_transforms import REDFT
import heapq

'''
def wrap_to_pi(phi):
    phi = phi - 2 * np.pi * np.floor((phi + np.pi) / (2 * np.pi))
    return phi
'''

def wrap_to_2pi(angle):
    angle = np.remainder(angle, 2 * np.pi).astype(angle.dtype)
    is_pos = (angle > 0)
    angle[(angle == 0) & is_pos] = 2 * np.pi
    return angle


def wrap_to_pi(angle):
    q = np.where((angle < -np.pi) | (np.pi < angle))
    angle[q] = wrap_to_2pi(angle[q] + np.pi) - np.pi
    return angle


def phase_residues(phi):
    d1 = phi[:-1, :-1] - phi[1:, :-1]
    d2 = phi[1:, :-1] - phi[1:, 1:]
    d3 = phi[1:, 1:] - phi[:-1, 1:]
    d4 = phi[:-1, 1:] - phi[:-1, :-1]

    # Calculate residue
    r = wrap_to_pi(d1) + wrap_to_pi(d2) + wrap_to_pi(d3) + wrap_to_pi(d4)
    return r

def unwrapping_operator(p1, p2):
    return np.floor((p1 - p2 + np.pi) / (2 * np.pi))

def quality_measure(phi):
    # Returns the quality measure. Probability a residue is introduced from a
    # linear phase shift monotonically increases with respect to this quantity.
    d1 = phi[1:, :-1] - phi[:-1, :-1]
    d2 = phi[:-1, 1:] - phi[1:, 1:]
    t = np.abs(wrap_to_pi(d1 + d2))

    # Quality of each pixel will be the average of the values in t it contributes to
    qmap = np.zeros_like(phi)
    counts = np.zeros_like(phi)

    qmap[:-1, :-1] += t
    qmap[1:, :-1] += t
    qmap[:-1, 1:] += t
    qmap[1:, 1:] += t
    counts[:-1, :-1] += 1
    counts[1:, :-1] += 1
    counts[:-1, 1:] += 1
    counts[1:, 1:] += 1
    return qmap

def sort_edges(phi):
    qmap = quality_measure(phi)
    rows, cols = qmap.shape
    num_edges = (rows * (cols - 1)) + ((rows - 1) * cols)  # Total edges (horizontal + vertical)

    # Initialize arrays
    edges = np.zeros((num_edges, 2, 2), dtype=int)  # Store edge indices as ((i1, j1), (i2, j2))
    values = np.zeros(num_edges)  # Store edge sums

    index = 0
    # Horizontal edges (left-right neighbors)
    for i in range(rows):
        for j in range(cols - 1):
            edges[index] = [(i, j), (i, j + 1)]
            values[index] = qmap[i, j] + qmap[i, j + 1]
            index += 1

    # Vertical edges (top-bottom neighbors)
    for i in range(rows - 1):
        for j in range(cols):
            edges[index] = [(i, j), (i + 1, j)]
            values[index] = qmap[i, j] + qmap[i + 1, j]
            index += 1

    # Sort edges by values
    sorted_indices = np.argsort(values)
    sorted_edges = edges[sorted_indices]
    return sorted_edges

def quality_guided_unwrap(phi):
    sorted_edges = sort_edges(phi)
    pxl_to_group, group_to_pxls, group_num = {}, {}, 0
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            pxl_to_group[(i, j)] = group_num
            group_to_pxls[group_num] = {(i, j)}
            group_num += 1

    for n in range(sorted_edges.shape[0]):
        edge = sorted_edges[n, :, :]
        px1 = (int(edge[0, 0]), int(edge[0, 1]))
        px2 = (int(edge[1, 0]), int(edge[1, 1]))
        p1, group1 = phi[px1[0], px1[1]], pxl_to_group[px1]
        p2, group2 = phi[px2[0], px2[1]], pxl_to_group[px2]
        if group1 == group2:
            continue
        else:
            unwrap_phi = unwrapping_operator(p1, p2)
            for pxl in group_to_pxls.pop(group2):
                phi[pxl[0], pxl[1]] += unwrap_phi
                group_to_pxls[group1].add(pxl)
                pxl_to_group[pxl] = group1
    return phi

class PhaseUnwrapTIE(REDFT):
    """
    This class is adapted from the method described in https://doi.org/10.1088/1361-6501/aaec5c.
    The author's Matlab implementation can be found here https://www.mathworks.com/matlabcentral/fileexchange/68493-robust-2d-phase-unwrapping-algorithm.
    """

    def __init__(self, Nx, Ny, nb=1, threads=1, dtype='float64', zero=False, **kwargs):
        super().__init__((Nx, Ny), nb, threads=threads, ortho=True, dtype=dtype, **kwargs)
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
