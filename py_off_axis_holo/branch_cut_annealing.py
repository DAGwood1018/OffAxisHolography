import numpy as np
from numba import njit, int32, types
from random import random
from numba.typed import List


@njit("float64(int32, int32, int32, int32)")
def dist(x1, y1, x2, y2):
    """
    Returns cut length. Assumes the cut is a straight line.

    :param x1: x coord 1
    :param y1: y coord 1
    :param x2: x coord 2
    :param y2: y coord 2
    :return: Distance
    """

    a, b = np.array([x1, y1], dtype='float64'), np.array([x2, y2], dtype='float64')
    return np.linalg.norm(a - b)


@njit("float64(int32[:, :], int32[:, :])")
def branch_cut_cost(branch_cuts, pole_coords):
    """
    Computes sum of all branch cut lengths to use as cost function.

    :param branch_cuts: Branch cut matrix. Each index represents a +/-1 pole or a border point.
                        If there is a branch cut between them then a 1 will be found at the corresponding matrix entrry.
                        Formatted so that the -1 pole is always first while the +1 pole is always second.
    :param pole_coords: Matrix that gives the coords of all +/- 1 poles and border points.
    :return: Total "cost" of all branch cuts
    """

    cost = 0
    cuts = np.argwhere(branch_cuts == 1)
    dists = np.zeros((len(cuts),), dtype='float64')
    for i, cut in enumerate(cuts):
        x1, y1, x2, y2 = pole_coords[cut[0], 0], pole_coords[cut[0], 1], pole_coords[cut[1], 0], pole_coords[cut[1], 1]
        dists[i] = dist(x1, y1, x2, y2)
        cost += dists[i]**2
    for i, cut1 in enumerate(cuts):
        for j, cut2 in enumerate(cuts[i:]):
            x11, y11, x21, y21 = pole_coords[cut1[0], 0], pole_coords[cut1[0], 1], pole_coords[cut1[1], 0], pole_coords[cut1[1], 1]
            x12, y12, x22, y22 = pole_coords[cut2[0], 0], pole_coords[cut2[0], 1], pole_coords[cut2[1], 0], pole_coords[cut2[1], 1]
            vec1, vec2 = np.array([x21-x11, y21-y11], dtype='float64'), np.array([x22-x12, y22-y12], dtype='float64')
            mu_ij = dists[i] + dists[i+j]
            cost += mu_ij * (1 - np.abs(vec1.dot(vec2) / dists[i] / dists[i+j]))
    return cost


@njit("float64[:,:](int32[:], int32[:], int32[:], int32[:,:])")
def branch_cut_dists(neg_poles, pos_poles, virtual_poles, pole_coords):
    """
    :param neg_poles: Array of the indices labeling -1 poles.
    :param pos_poles: Array of the indices labeling +1 poles.
    :param virtual_poles: Array of the indices labeling virtual poles (value depends on what is needed and are always available).
    :param pole_coords: Matrix that gives the coords of all +/- 1 poles and border points.
    :return: Matrix containing the lengths of all possible branch cuts.
    """

    n = len(pole_coords)
    dist_matrix = np.full((n, n), np.inf, dtype='float64')

    # Fill distance matrix valid pairs. The border is assumed on average to intersect a virtual pair midway.
    for p1 in neg_poles:
        for p2 in pos_poles:
            xy1, xy2 = pole_coords[p1], pole_coords[p2]
            d = dist(xy1[0], xy1[1], xy2[0], xy2[1])
            dist_matrix[p1, p2] = d
            dist_matrix[p2, p1] = d
    for p1 in neg_poles:
        for p2 in virtual_poles:
            xy1, xy2 = pole_coords[p1], pole_coords[p2]
            d = dist(xy1[0], xy1[1], xy2[0], xy2[1])
            dist_matrix[p1, p2] = 2 * d
            dist_matrix[p2, p1] = 2 * d
    for p1 in pos_poles:
        for p2 in virtual_poles:
            xy1, xy2 = pole_coords[p1], pole_coords[p2]
            d = dist(xy1[0], xy1[1], xy2[0], xy2[1])
            dist_matrix[p1, p2] = 2 * d
            dist_matrix[p2, p1] = 2 * d
    return dist_matrix


#@njit("int32[:, :](int32[:], int32[:], float64[:], float64[:, :])")
def make_branch_cuts(neg_poles, pos_poles, pole_vals, dist_matrix):
    # Border points will always remain available b/c the assumption is that there is some matching pole outside the
    # boundary not that the border point is the pole.
    all_poles = np.concatenate((neg_poles, pos_poles))
    branch_cuts = np.full(dist_matrix.shape, 0, dtype='int32')
    pole_availability = np.full(all_poles.shape, True, dtype='bool')
    availability_mask = np.full(dist_matrix.shape, 0, dtype='float64')

    # Iterate through poles at random.
    rng = np.random.default_rng()
    poles = rng.permutation(all_poles)  # random iteration order

    # Find heuristic solution by making branch cuts with a greedy nearest neighbor search.
    for p1 in poles:
        if pole_availability[p1]:
            masked_dist_matrix = dist_matrix + availability_mask
            dists = masked_dist_matrix[p1, :]
            p2 = np.argmin(dists)

            branch_cuts[p1, p2] = 1
            branch_cuts[p2, p1] = 1

            if ((pole_vals[p1] == -1 and pole_vals[p2] == -1) or
                    (pole_vals[p1] == 1 and pole_vals[p2] == 1) or (
                            np.isnan(pole_vals[p1]) and np.isnan(pole_vals[p2]))):
                raise ValueError(f"Improper branch cut detected ({pole_vals[p1]} <-> {pole_vals[p2]})")

            # Remove used points from availability
            pole_availability[p1] = False
            availability_mask[p1, :], availability_mask[:, p1] = np.inf, np.inf
            if np.isnan(pole_vals[p2]):
                continue
            pole_availability[p2] = False
            availability_mask[p2, :], availability_mask[:, p2] = np.inf, np.inf
    if len(np.argwhere(pole_availability == True)) > 0:
        raise RuntimeError("Not enough branch cuts were made.")
    return branch_cuts


def rand_branch_cuts(neg_poles, pos_poles, virtual_poles):
    # Border points will always remain available b/c the assumption is that there is some matching pole outside the
    # boundary not that the border point is the pole.
    all_poles = np.concatenate((neg_poles, pos_poles))
    N = len(neg_poles) + len(pos_poles) + len(virtual_poles)
    branch_cuts = np.full((N, N), 0, dtype='int32')
    pole_availability = np.full(all_poles.shape, True, dtype='bool')
    availability_mask = np.full(dist_matrix.shape, 0, dtype='float64')

    # Iterate through poles at random.
    rng = np.random.default_rng()
    poles = rng.permutation(all_poles)  # random iteration order

    # Find heuristic solution by making branch cuts with a greedy nearest neighbor search.
    for p1 in poles:
        if pole_availability[p1]:
            masked_dist_matrix = dist_matrix + availability_mask
            dists = masked_dist_matrix[p1, :]
            p2 = np.argmin(dists)

            branch_cuts[p1, p2] = 1
            branch_cuts[p2, p1] = 1

            if ((pole_vals[p1] == -1 and pole_vals[p2] == -1) or
                    (pole_vals[p1] == 1 and pole_vals[p2] == 1) or (
                            np.isnan(pole_vals[p1]) and np.isnan(pole_vals[p2]))):
                raise ValueError(f"Improper branch cut detected ({pole_vals[p1]} <-> {pole_vals[p2]})")

            # Remove used points from availability
            pole_availability[p1] = False
            availability_mask[p1, :], availability_mask[:, p1] = np.inf, np.inf
            if np.isnan(pole_vals[p2]):
                continue
            pole_availability[p2] = False
            availability_mask[p2, :], availability_mask[:, p2] = np.inf, np.inf
    if len(np.argwhere(pole_availability == True)) > 0:
        raise RuntimeError("Not enough branch cuts were made.")
    return branch_cuts


@njit("int32[:, :](int32, int32, int32[:, :], float64[:], float64[:, :])")
def mix_branch_cuts(p00, p11, branch_cuts, pole_vals, virtual_pole_dists):
    """
    :param p00: -1 pole
    :param p11: +1 pole
    :param branch_cuts:
    :param pole_vals:
    :param virtual_pole_dists:
    :return:
    """

    # The branch cuts are (p00, p01) and (p10, p11)
    p00_val, p11_val = pole_vals[p00], pole_vals[p11]
    if not (p00_val == -1 and p11_val == 1):
        return branch_cuts

    p01, p10 = np.argwhere(branch_cuts[p00, :] == 1), np.argwhere(branch_cuts[p11, :] == 1)
    if np.size(p01) > 1 or np.size(p10) > 1:
        raise ValueError("Inconsistent branch cut encountered.")
    elif np.size(p01) == 0 or np.size(p10) == 0:
        raise ValueError("A pole has no branch cut.")
    else:
        p01, p10 = p01[0, 0], p10[0, 0]
    p01_val, p10_val = pole_vals[p01], pole_vals[p10]

    # TODO branch cut is being dropped somewhere in this
    # Break the branch cut if the two chosen poles are currently in the same cut.
    if p01 == p11:
        if not p10 == p00:
            raise ValueError("Inconsistent branch cut encountered.")
        p2 = np.argmin(virtual_pole_dists[p00, :])
        p3 = np.argmin(virtual_pole_dists[p11, :])
        branch_cuts[p00, p11] = branch_cuts[p11, p00] = 0
        branch_cuts[p00, p2] = branch_cuts[p2, p00] = 1
        branch_cuts[p11, p3] = branch_cuts[p3, p11] = 1
    if np.isnan(p01_val) and np.isnan(p10_val):
        # Make new branch cut between the non-virtual poles
        branch_cuts[p00, p11] = branch_cuts[p11, p00] = 1
        branch_cuts[p00, p01] = branch_cuts[p01, p00] = 0
        branch_cuts[p10, p11] = branch_cuts[p11, p10] = 0
    elif np.isnan(p01_val) and p10_val == -1:
        pick = random()
        if pick < 0.50:
            # Find the closest virtual pole of p10
            p2 = np.argmin(virtual_pole_dists[p10, :])

            # Break current branch cuts
            branch_cuts[p00, p01] = branch_cuts[p01, p00] = 0
            branch_cuts[p10, p11] = branch_cuts[p11, p10] = 0

            # Swap p01 and p11
            branch_cuts[p10, p2] = branch_cuts[p2, p10] = 1
            branch_cuts[p00, p11] = branch_cuts[p11, p00] = 1
        else:
            # Break real branch cut leave virtual branch cut unchanged
            p2 = np.argmin(virtual_pole_dists[p10, :])
            p3 = np.argmin(virtual_pole_dists[p11, :])
            branch_cuts[p10, p11] = branch_cuts[p11, p10] = 0
            branch_cuts[p10, p2] = branch_cuts[p2, p10] = 1
            branch_cuts[p11, p3] = branch_cuts[p3, p11] = 1
    elif p01_val == 1 and np.isnan(p10_val):
        pick = random()
        if pick < 0.50:
            # Find closest virtual pole of p00
            p2 = np.argmin(virtual_pole_dists[p00, :])

            # Break current branch cuts
            branch_cuts[p00, p01] = branch_cuts[p01, p00] = 0
            branch_cuts[p10, p11] = branch_cuts[p11, p10] = 0

            # Swap p01 and p11
            branch_cuts[p00, p2] = branch_cuts[p2, p00] = 1
            branch_cuts[p01, p11] = branch_cuts[p11, p01] = 1
        else:
            # Break real branch cut leave virtual branch cut unchanged
            p2 = np.argmin(virtual_pole_dists[p00, :])
            p3 = np.argmin(virtual_pole_dists[p01, :])
            branch_cuts[p00, p01] = branch_cuts[p01, p00] = 0
            branch_cuts[p00, p2] = branch_cuts[p2, p00] = 1
            branch_cuts[p01, p3] = branch_cuts[p3, p01] = 1
    else:
        branch_cuts[p00, p01] = branch_cuts[p01, p00] = 0
        branch_cuts[p10, p11] = branch_cuts[p11, p10] = 0

        # Swap -1 poles between branch cuts
        branch_cuts[p00, p11] = branch_cuts[p11, p00] = 1
        branch_cuts[p10, p01] = branch_cuts[p01, p10] = 1
        '''
        pick = random()
        if pick < 0.25:
            # Break current branch cuts
            branch_cuts[p00, p01] = branch_cuts[p01, p00] = 0
            branch_cuts[p10, p11] = branch_cuts[p11, p10] = 0

            # Swap -1 poles between branch cuts
            branch_cuts[p00, p11] = branch_cuts[p11, p00] = 1
            branch_cuts[p10, p01] = branch_cuts[p01, p10] = 1
        elif 0.25 <= pick < 0.50:
            # Break current branch cuts
            branch_cuts[p00, p01] = branch_cuts[p01, p00] = 0
            branch_cuts[p10, p11] = branch_cuts[p11, p10] = 0

            # Swap one of -1 poles between branch cuts
            branch_cuts[p00, p11] = branch_cuts[p11, p00] = 1

            # Reconnect other two poles to virtual poles
            p2 = np.argmin(virtual_pole_dists[p01, :])
            p3 = np.argmin(virtual_pole_dists[p10, :])
            branch_cuts[p01, p2] = branch_cuts[p2, p01] = 1
            branch_cuts[p10, p3] = branch_cuts[p3, p10] = 1
        elif 0.50 <= pick < 0.75:
            # Break current branch cuts
            branch_cuts[p00, p01] = branch_cuts[p01, p00] = 0
            branch_cuts[p10, p11] = branch_cuts[p11, p10] = 0

            # Swap one of -1 poles between branch cuts
            branch_cuts[p10, p01] = branch_cuts[p01, p10] = 1

            # Reconnect other two poles to virtual poles
            p2 = np.argmin(virtual_pole_dists[p00, :])
            p3 = np.argmin(virtual_pole_dists[p11, :])
            branch_cuts[p00, p2] = branch_cuts[p2, p00] = 1
            branch_cuts[p11, p3] = branch_cuts[p3, p11] = 1
        else:
            # Find nearest virtual poles for all real poles
            p2 = np.argmin(virtual_pole_dists[p00, :])
            p3 = np.argmin(virtual_pole_dists[p01, :])
            p4 = np.argmin(virtual_pole_dists[p10, :])
            p5 = np.argmin(virtual_pole_dists[p11, :])

            # Break current branch cuts
            branch_cuts[p00, p01] = branch_cuts[p01, p00] = 0
            branch_cuts[p10, p11] = branch_cuts[p11, p10] = 0

            # Connect all real poles to virtual poles
            branch_cuts[p00, p2] = branch_cuts[p2, p00] = 1
            branch_cuts[p01, p3] = branch_cuts[p3, p01] = 1
            branch_cuts[p10, p4] = branch_cuts[p4, p10] = 1
            branch_cuts[p11, p5] = branch_cuts[p5, p11] = 1
        '''
    return branch_cuts


@njit("float64(int32, int32[:], int32[:], int32[:,:], int32[:,:], float64[:], float64[:,:])")
def init_temperature(N, neg_poles, pos_poles, branch_cuts, pole_coords, pole_vals, virtual_pole_dists):
    dE, test_dE = 0.0, 0.0
    E = branch_cut_cost(branch_cuts, pole_coords)
    for n in range(N):
        i = int(np.floor(random() * neg_poles.shape[0]))
        j = int(np.floor(random() * pos_poles.shape[0]))
        neg_pole = neg_poles[i]
        pos_pole = pos_poles[j]
        neighbor_state = mix_branch_cuts(neg_pole, pos_pole, branch_cuts.copy(), pole_vals, virtual_pole_dists)
        test_dE = branch_cut_cost(neighbor_state, pole_coords)
        if test_dE > dE:
            dE = test_dE
    return dE


@njit
def heat_init_state(N, T_init, neg_poles, pos_poles, branch_cuts, pole_coords, pole_vals, virtual_pole_dists,
                    heating_rate):
    T = T_init
    E = branch_cut_cost(branch_cuts, pole_coords)
    temps, energies = np.empty((N,), dtype='float64'), np.empty((N,), dtype='float64')
    for n in range(N):
        i = int(np.floor(random() * neg_poles.shape[0]))
        j = int(np.floor(random() * pos_poles.shape[0]))
        neg_pole = neg_poles[i]
        pos_pole = pos_poles[j]
        neighbor_state = mix_branch_cuts(neg_pole, pos_pole, branch_cuts.copy(), pole_vals, virtual_pole_dists)
        E_new = branch_cut_cost(neighbor_state, pole_coords)
        if E_new - E > 0 or random() > np.exp(-(E_new - E) / T):
            E = E_new
            branch_cuts = neighbor_state
            for i in range(len(neg_poles) + len(pos_poles)):
                if np.size(np.argwhere(branch_cuts[i, :] == 1)) == 0:
                    print(f'lost branch cut for pole {i}')
        energies[n] = E
        temps[n] = T
        T *= heating_rate

    return branch_cuts, temps, energies


@njit
def anneal_branch_cuts(N, T_init, neg_poles, pos_poles, branch_cuts, pole_coords, pole_vals, virtual_pole_dists,
                       cooling_rate):
    T = T_init
    E = branch_cut_cost(branch_cuts, pole_coords)
    temps, energies = np.empty((N,), dtype='float64'), np.empty((N,), dtype='float64')
    for n in range(N):
        i = int(np.floor(random() * neg_poles.shape[0]))
        j = int(np.floor(random() * pos_poles.shape[0]))
        neg_pole = neg_poles[i]
        pos_pole = pos_poles[j]
        neighbor_state = mix_branch_cuts(neg_pole, pos_pole, branch_cuts.copy(), pole_vals, virtual_pole_dists)
        E_new = branch_cut_cost(neighbor_state, pole_coords)
        if E_new - E < 0 or random() < np.exp(-(E_new - E) / T):
            E = E_new
            branch_cuts = neighbor_state
        energies[n] = E
        temps[n] = T
        T *= cooling_rate

    return branch_cuts, temps, energies


#@njit("types.ListType(types.UniTuple(int32, 2))(int32, int32, int32, int32)")
def draw_branch_cut(x1, y1, x2, y2):
    """
    Uses Bresenham's algorithm to identify pixels along branch cut between the two points

    :param x1: x coord 1
    :param y1: y coord 1
    :param x2: x coord 2
    :param y2: y coord 2
    :return: List of pixels that are part of branch cuts.
    """

    dx = abs(x2 - x1)
    dy = -abs(y2 - y1)
    error = dx + dy
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    points = List.empty_list(types.UniTuple(int32, 2))

    while True:
        points.append((x1, y1))
        e2 = 2 * error
        if e2 >= dy:
            if x2 == x1:
                break
            error += dy
            x1 += sx
        if e2 <= dx:
            if y2 == y1:
                break
            error += dx
            y1 += sy
    return points


def check_for_pole_repeats(branch_cuts, pole_vals):
    poles = set()
    for cut in np.argwhere(branch_cuts == 1):
        p1, p2 = cut[0], cut[1]
        if not np.isnan(pole_vals[p1]):
            if p1 in poles:
                return True
            else:
                poles.add(p1)
        if not np.isnan(pole_vals[p2]):
            if p2 in poles:
                return True
            else:
                poles.add(p2)
    return False
