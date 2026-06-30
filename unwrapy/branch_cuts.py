import numpy as np
from ortools.graph.python import min_cost_flow
from matplotlib import pyplot as plt


def mcf_branch_cuts(residues, cost_x, cost_y, boundary_cost=1, capacity=10 ** 6):
    """
    Fully vectorized OR-Tools MCF builder (no Python loops over edges).
    """

    M, N = residues.shape
    num_nodes = M * N
    boundary = num_nodes

    mcf = min_cost_flow.SimpleMinCostFlow()

    # ------------------------------------------------------------
    # 1. Node supplies (vectorized)
    # ------------------------------------------------------------

    supplies = residues.ravel().astype(int)

    for i in range(num_nodes):
        mcf.set_node_supply(i, int(supplies[i]))

    # balance to boundary
    mcf.set_node_supply(boundary, -int(supplies.sum()))

    # ------------------------------------------------------------
    # 2. Horizontal edges (vectorized)
    # ------------------------------------------------------------

    r = np.repeat(np.arange(M), N - 1)
    c = np.tile(np.arange(N - 1), M)

    u = r * N + c
    v = u + 1

    start_nodes = np.concatenate([u, v])
    end_nodes = np.concatenate([v, u])
    costs = np.concatenate([cost_x.ravel(), cost_x.ravel()])
    caps = np.full_like(costs, capacity)

    # add to solver
    for i in range(len(start_nodes)):
        mcf.add_arc_with_capacity_and_unit_cost(
            int(start_nodes[i]),
            int(end_nodes[i]),
            int(caps[i]),
            int(costs[i])
        )

    # ------------------------------------------------------------
    # 3. Vertical edges (vectorized)
    # ------------------------------------------------------------

    r = np.repeat(np.arange(M - 1), N)
    c = np.tile(np.arange(N), M - 1)

    u = r * N + c
    v = u + N

    start_nodes = np.concatenate([u, v])
    end_nodes = np.concatenate([v, u])
    costs = np.concatenate([cost_y.ravel(), cost_y.ravel()])
    caps = np.full_like(costs, capacity)

    for i in range(len(start_nodes)):
        mcf.add_arc_with_capacity_and_unit_cost(
            int(start_nodes[i]),
            int(end_nodes[i]),
            int(caps[i]),
            int(costs[i])
        )

    # ------------------------------------------------------------
    # 4. Boundary edges (vectorized)
    # ------------------------------------------------------------

    boundary_nodes = []

    # top & bottom rows
    boundary_nodes += list(np.arange(N))
    boundary_nodes += list((M - 1) * N + np.arange(N))

    # left & right columns
    boundary_nodes += list(np.arange(0, M * N, N))
    boundary_nodes += list(np.arange(N - 1, M * N, N))

    boundary_nodes = np.array(boundary_nodes, dtype=int)
    unique_nodes = np.unique(boundary_nodes)

    for u in unique_nodes:
        mcf.add_arc_with_capacity_and_unit_cost(
            int(u),
            boundary,
            capacity,
            boundary_cost
        )

        mcf.add_arc_with_capacity_and_unit_cost(
            boundary,
            int(u),
            capacity,
            boundary_cost
        )

    return mcf


def extract_branch_cuts(mcf, node_id, shape):
    """
    Extract branch cuts from solved OR-Tools flow.

    Returns
    -------
    cut_x : (M, N-1) bool
        Horizontal cuts

    cut_y : (M-1, N) bool
        Vertical cuts
    """

    M, N = shape

    cut_x = np.zeros((M, N - 1), dtype=bool)
    cut_y = np.zeros((M - 1, N), dtype=bool)

    for i in range(mcf.num_arcs()):
        tail = mcf.tail(i)
        head = mcf.head(i)
        flow = mcf.flow(i)

        if flow == 0:
            continue

        # horizontal edges
        for r in range(M):
            for c in range(N - 1):
                u = node_id(r, c)
                v = node_id(r, c + 1)
                if (tail == u and head == v) or (tail == v and head == u):
                    cut_x[r, c] = True

        # vertical edges
        for r in range(M - 1):
            for c in range(N):
                u = node_id(r, c)
                v = node_id(r + 1, c)
                if (tail == u and head == v) or (tail == v and head == u):
                    cut_y[r, c] = True

    return cut_x, cut_y


def plot_branch_cuts(wrapped_phase, cut_x, cut_y):
    M, N = wrapped_phase.shape

    plt.figure(figsize=(6, 6))
    plt.imshow(wrapped_phase, cmap='jet')
    plt.colorbar(label="Wrapped phase")

    # horizontal cuts
    for r in range(M):
        for c in range(N - 1):
            if cut_x[r, c]:
                plt.plot([c, c + 1], [r, r], 'w-', linewidth=2)

    # vertical cuts
    for r in range(M - 1):
        for c in range(N):
            if cut_y[r, c]:
                plt.plot([c, c], [r, r + 1], 'w-', linewidth=2)

    plt.title("Branch cuts from MCF")
    plt.gca().invert
