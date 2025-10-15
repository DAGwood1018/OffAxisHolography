import numpy as np
from numba import njit

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

@njit()
def swap_branch_cuts():
    pass

@njit()
def break_branch_cut():
    pass

@njit()
def init_temperature():
    pass

@njit()
def heat_init_state():
    pass

@njit()
def anneal_branch_cuts():
    pass

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