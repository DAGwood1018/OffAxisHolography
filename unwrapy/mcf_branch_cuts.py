"""
Minimum-Cost-Flow (MCF) phase unwrapping, in the style of Costantini (1998)
and SNAPHU's network-flow formulation.

Pipeline
--------
1. Residues are computed on the dual grid (M-1, N-1) -- one node per 2x2
   pixel loop.
2. A min-cost-flow network is built:
     - one node per residue,
     - one extra "ground" node representing the image border (lets
       branch cuts terminate at the edge of the image instead of forcing
       every residue to pair up with another interior residue),
     - arcs between 4-connected neighboring residue-nodes, cost = the
       primal-grid edge cost that separates them (from calc_costs),
     - arcs from every border residue-node to ground, cost = the
       corresponding boundary primal-edge cost.
3. The MCF solution gives, for every primal-grid edge, an integer number
   of 2*pi branch-cut jumps to add to the wrapped gradient there.
4. The corrected (curl-free) gradient field is integrated with a
   flood fill biased towards areas of low residue density.
"""

from ortools.graph.python.min_cost_flow import SimpleMinCostFlow
import numpy as np

from unwrapy.phase_residues import residue_density
from unwrapy.phase_unwrapping import floodfill, biased_floodfill


# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------

def _node_id(i, j, ncols):
    return i * ncols + j


def build_mcf_arcs(residues, cost_h, cost_v):
    """
    Build the MCF problem for phase unwrapping (vectorized).

    Parameters
    ----------
    residues : (M-1, N-1) int array, values in {-1, 0, 1} (or more generally
        small integers if you allow multi-charge residues).
    cost_h : (M, N-1) int array   -- horizontal primal edge costs.
    cost_v : (M-1, N) int array   -- vertical primal edge costs.

    Returns
    -------
    tails, heads, costs : 1D int64 arrays, one direction only per arc
        (the solver adds the reverse direction too, since flow can go
        either way). Node ids 0 .. R-1 are residue nodes (row-major over
        the (M-1, N-1) grid), node id R is the ground node.
    supplies : (R+1,) int64 array, node supply/demand (sums to zero).
    shape : (Mr, Nr) shape of the residue grid.
    """
    residues = np.asarray(residues, dtype=np.int64)
    Mr, Nr = residues.shape  # Mr = M-1, Nr = N-1
    R = Mr * Nr
    ground_id = R

    node_ids = np.arange(R, dtype=np.int64).reshape(Mr, Nr)

    tails_list, heads_list, costs_list = [], [], []

    # interior horizontal dual edges: node(i,j) -- node(i,j+1)
    # crosses primal vertical edge cost_v[i, j+1]
    a = node_ids[:, :-1].ravel()
    b = node_ids[:, 1:].ravel()
    c = cost_v[:, 1:-1].ravel().astype(np.int64)
    tails_list.append(a); heads_list.append(b); costs_list.append(c)

    # interior vertical dual edges: node(i,j) -- node(i+1,j)
    # crosses primal horizontal edge cost_h[i+1, j]
    a = node_ids[:-1, :].ravel()
    b = node_ids[1:, :].ravel()
    c = cost_h[1:-1, :].ravel().astype(np.int64)
    tails_list.append(a); heads_list.append(b); costs_list.append(c)

    # ground (border) arcs
    g = np.full(Nr, ground_id, dtype=np.int64)
    tails_list.append(node_ids[0, :]); heads_list.append(g); costs_list.append(cost_h[0, :].astype(np.int64))
    tails_list.append(node_ids[-1, :]); heads_list.append(g); costs_list.append(cost_h[-1, :].astype(np.int64))

    g = np.full(Mr, ground_id, dtype=np.int64)
    tails_list.append(node_ids[:, 0]); heads_list.append(g); costs_list.append(cost_v[:, 0].astype(np.int64))
    tails_list.append(node_ids[:, -1]); heads_list.append(g); costs_list.append(cost_v[:, -1].astype(np.int64))

    tails = np.concatenate(tails_list)
    heads = np.concatenate(heads_list)
    costs = np.concatenate(costs_list)

    supplies = np.zeros(R + 1, dtype=np.int64)
    supplies[:R] = -residues.flatten()
    supplies[ground_id] = int(residues.sum())  # balances total supply to 0

    return tails, heads, costs, supplies, (Mr, Nr)


# --------------------------------------------------------------------------
# Solve
# --------------------------------------------------------------------------

def solve_mcf(tails, heads, costs, supplies, cap=None, growth=8):
    """
    Solve the min-cost-flow problem defined by (tails, heads, costs, supplies).

    Each undirected arc (tail, head, cost) is realized as two directed arcs
    (both directions) so flow can go either way, since residues carry sign.

    Capacity
    --------
    Global charge balance (sum of all node supplies = 0, i.e. "net residue
    charge is 0") is enforced through the *supplies* array (build_mcf_arcs
    sets the ground node's supply to sum(residues), exactly canceling the
    interior nodes' -residue supplies) -- it has nothing to do with arc
    capacity. Capacity only needs to be large enough that a feasible flow
    *can* exist; since every arc cost is >= 0, an optimal solver never
    pushes flow around a closed loop just because capacity allows it (that
    strictly adds cost for no benefit), so any capacity at or above the
    minimal *feasible* value is exactly as good as an infinite one.

    A safe (but often very loose) upper bound is sum(|residues|): no single
    arc in an optimal solution ever needs to carry more flow than the total
    charge that exists to move around in the whole image. But for a large
    image with many small, spatially separated residue clusters, that
    global bound can be far bigger than what's actually needed for any
    individual arc, and can slow the solver down or blow up the cost*capacity
    product OR-tools uses internally. So unless `cap` is given explicitly,
    this does an exponential ("doubling") search for the smallest capacity
    that still yields a feasible solve: start small, and if OR-tools reports
    infeasible, multiply by `growth` and retry, capped at the provably
    sufficient sum(|residues|) bound as a last resort. This typically finds
    a much tighter, still-correct capacity in just a couple of extra solves.

    Returns
    -------
    net_flow : 1D int64 array, same length/order as `tails`/`heads`:
        net_flow[k] = flow(tails[k] -> heads[k]) - flow(heads[k] -> tails[k])
    used_cap : int, the capacity that actually produced a feasible solve.
    """
    # ground node is always the last entry in `supplies` (see build_mcf_arcs)
    total_abs_charge = max(int(np.abs(supplies[:-1]).sum()), 1)

    if cap is not None:
        cap_schedule = [int(cap)]
    else:
        # start near the largest single residue's magnitude (almost always
        # enough for the typical case of small, local dipole-like clusters),
        # then grow geometrically up to the provably-sufficient global bound.
        start = max(int(np.abs(supplies[:-1]).max()), 1)
        cap_schedule = []
        c = start
        while c < total_abs_charge:
            cap_schedule.append(c)
            c *= growth
        cap_schedule.append(total_abs_charge)

    n_arcs = len(tails)
    last_status = None

    for c in cap_schedule:

        smcf = SimpleMinCostFlow()
        all_tails = np.concatenate([tails, heads])
        all_heads = np.concatenate([heads, tails])
        all_costs = np.concatenate([costs, costs])
        all_caps = np.full(2 * n_arcs, c, dtype=np.int64)

        smcf.add_arcs_with_capacity_and_unit_cost(all_tails, all_heads, all_caps, all_costs)
        smcf.set_nodes_supplies(np.arange(len(supplies), dtype=np.int64), supplies.astype(np.int64))

        status = smcf.solve()
        last_status = status
        if status == smcf.OPTIMAL:
            flows = smcf.flows(np.arange(2 * n_arcs, dtype=np.int64))
            net_flow = flows[:n_arcs] - flows[n_arcs:]
            return net_flow, c
        # INFEASIBLE (capacity too tight) -> try the next, larger cap
        continue
    raise RuntimeError(
        f"MCF solve failed even at the provably-sufficient capacity "
        f"{total_abs_charge} (last status={last_status}); this should not "
        f"happen unless supplies don't sum to zero."
    )


# --------------------------------------------------------------------------
# Convert flow -> per-primal-edge 2*pi jump counts
# --------------------------------------------------------------------------

def flow_to_jumps(net_flow, residues_shape, M, N):
    """
    Translate the solved dual-graph flow (in the same arc order produced by
    build_mcf_arcs) into integer jump-count arrays aligned with cost_h
    (M, N-1) and cost_v (M-1, N):

        k_h[i, j]  -> number of extra 2*pi cycles to add when integrating
                      from pixel (i, j) to pixel (i, j+1)
        k_v[i, j]  -> number of extra 2*pi cycles to add when integrating
                      from pixel (i, j) to pixel (i+1, j)
    """
    Mr, Nr = residues_shape

    k_h = np.zeros((M, N - 1), dtype=np.int64)
    k_v = np.zeros((M - 1, N), dtype=np.int64)

    pos = 0

    # interior horizontal dual edges -> k_v[:, 1:-1], shape (Mr, Nr-1)
    n = Mr * (Nr - 1)
    k_v[:, 1:-1] = net_flow[pos:pos + n].reshape(Mr, Nr - 1)
    pos += n

    # interior vertical dual edges -> k_h[1:-1, :], shape (Mr-1, Nr)
    n = (Mr - 1) * Nr
    k_h[1:-1, :] = net_flow[pos:pos + n].reshape(Mr - 1, Nr)
    pos += n

    # ground arcs: top, bottom, left, right (must match build_mcf_arcs order)
    n = Nr
    k_h[0, :] = -net_flow[pos:pos + n]
    pos += n
    k_h[-1, :] = net_flow[pos:pos + n]
    pos += n

    n = Mr
    k_v[:, 0] = -net_flow[pos:pos + n]
    pos += n
    k_v[:, -1] = net_flow[pos:pos + n]
    pos += n

    return k_h, k_v

# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------

def mcf_unwrap(wrapped_phase, residues, cost_h, cost_v, ref=(0, 0), cap=None,
               density_sigma=3.0):
    """
    Full MCF phase-unwrapping pipeline.

    Parameters
    ----------
    wrapped_phase : (M, N) float array, radians
    residues : (M-1, N-1) int array (from calc_residues)
    cost_h : (M, N-1) int array (from calc_costs)
    cost_v : (M-1, N) int array (from calc_costs)
    ref : reference pixel to hold fixed at its wrapped value.
    cap : force a specific arc capacity instead of the automatic adaptive
        search (see solve_mcf's docstring).
    density_sigma : Gaussian blur radius (pixels) for the residue-density
        map that orders the flood-fill integration; low-density (clean)
        regions are integrated before high-density (near branch cuts)
        regions. Set to None to fall back to a plain BFS instead.

    Returns
    -------
    unwrapped : (M, N) float array
    info : dict with 'k_h', 'k_v', 'n_cut_edges', 'total_cost', 'cap',
        'density' (the residue-density map used to order the flood fill,
        or None if density_sigma is None).
    """
    M, N = wrapped_phase.shape
    tails, heads, costs, supplies, shape = build_mcf_arcs(residues, cost_h, cost_v)
    net_flow, used_cap = solve_mcf(tails, heads, costs, supplies, cap=cap)
    k_h, k_v = flow_to_jumps(net_flow, shape, M, N)

    if density_sigma is None:
        density = None
        unwrapped = floodfill(wrapped_phase, k_h, k_v, ref=ref)
    else:
        density = residue_density(residues, M, N, sigma=density_sigma)
        unwrapped = biased_floodfill(wrapped_phase, k_h, k_v, density, ref=ref)

    n_cut = int(np.count_nonzero(k_h) + np.count_nonzero(k_v))
    total_cost = int(
        np.sum(np.abs(k_h) * cost_h) + np.sum(np.abs(k_v) * cost_v)
    )
    info = {
        "k_h": k_h, "k_v": k_v, "n_cut_edges": n_cut, "total_cost": total_cost,
        "cap": used_cap, "density": density,
    }
    return unwrapped, info


# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------

def plot_branch_cuts(background, residues, k_h, k_v, ax=None, cmap="gray",
                      cut_color="red", show_residues=True, linewidth_scale=1.5,
                      title="Branch cuts"):
    """
    Plot branch cuts (and, optionally, residue locations) over a background
    image (e.g. the wrapped phase, the contrast map, or the unwrapped
    phase).

    Branch cuts are drawn the standard way: as segments on the *dual* grid
    (the grid whose nodes are residues), since that's what a "cut" actually
    separates -- not as marks on the primal pixel edges themselves. Each
    residue sits at the center of the 2x2 pixel loop it was computed from,
    i.e. residue (i, j) is at continuous image coordinates
    (row=i+0.5, col=j+0.5). A nonzero k_h[i, j] or k_v[i, j] means the
    solver cut the primal pixel-edge at that location, which is the same
    thing as saying the dual edge between the two residue-cells (or a
    residue-cell and the image border) adjacent to that primal edge is
    "on" -- so that's the segment drawn.

    Parameters
    ----------
    background : (M, N) ndarray
        Image to show underneath the cuts (wrapped phase, contrast, etc).
    residues : (M-1, N-1) int array
    k_h : (M, N-1) int array  -- from mcf_unwrap()'s info["k_h"]
    k_v : (M-1, N) int array  -- from mcf_unwrap()'s info["k_v"]
    ax : matplotlib Axes, optional
    linewidth_scale : scales line width by |k| (multiple 2*pi cuts stacked
        on the same edge get a thicker line).

    Returns
    -------
    ax : matplotlib Axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    M, N = background.shape
    Mr, Nr = residues.shape  # Mr = M-1, Nr = N-1

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(background, cmap=cmap, origin="upper",
              extent=[-0.5, N - 0.5, M - 0.5, -0.5])

    segments = []
    widths = []

    # --- interior vertical dual edges, from k_h[1:-1, :] -------------------
    # k_h[r, j] (1 <= r <= Mr-1) came from dual pair (r-1, j) -- (r, j)
    rows, cols = np.nonzero(k_h[1:-1, :])
    for r0, j in zip(rows, cols):
        r = r0 + 1  # index back into k_h
        y0, y1 = (r - 1) + 0.5, r + 0.5
        x = j + 0.5
        segments.append([(x, y0), (x, y1)])
        widths.append(abs(int(k_h[r, j])))

    # --- interior horizontal dual edges, from k_v[:, 1:-1] ------------------
    # k_v[i, c] (1 <= c <= Nr-1) came from dual pair (i, c-1) -- (i, c)
    rows, cols = np.nonzero(k_v[:, 1:-1])
    for i, c0 in zip(rows, cols):
        c = c0 + 1  # index back into k_v
        x0, x1 = (c - 1) + 0.5, c + 0.5
        y = i + 0.5
        segments.append([(x0, y), (x1, y)])
        widths.append(abs(int(k_v[i, c])))

    # --- ground (border) cuts -----------------------------------------------
    # top: k_h[0, j] -> dual node (0, j) to top border
    for j in np.nonzero(k_h[0, :])[0]:
        segments.append([(j + 0.5, 0.5), (j + 0.5, -0.5)])
        widths.append(abs(int(k_h[0, j])))
    # bottom: k_h[-1, j] -> dual node (Mr-1, j) to bottom border
    for j in np.nonzero(k_h[-1, :])[0]:
        segments.append([(j + 0.5, Mr - 0.5), (j + 0.5, M - 0.5)])
        widths.append(abs(int(k_h[-1, j])))
    # left: k_v[i, 0] -> dual node (i, 0) to left border
    for i in np.nonzero(k_v[:, 0])[0]:
        segments.append([(0.5, i + 0.5), (-0.5, i + 0.5)])
        widths.append(abs(int(k_v[i, 0])))
    # right: k_v[i, -1] -> dual node (i, Nr-1) to right border
    for i in np.nonzero(k_v[:, -1])[0]:
        segments.append([(Nr - 0.5, i + 0.5), (N - 0.5, i + 0.5)])
        widths.append(abs(int(k_v[i, -1])))

    if segments:
        widths = np.array(widths, dtype=float)
        lc = LineCollection(segments, colors=cut_color,
                             linewidths=1.0 + linewidth_scale * (widths - 1),
                             zorder=3)
        ax.add_collection(lc)

    if show_residues:
        pos_i, pos_j = np.nonzero(residues > 0)
        neg_i, neg_j = np.nonzero(residues < 0)
        ax.scatter(pos_j + 0.5, pos_i + 0.5, marker="+", s=60,
                   c="yellow", linewidths=2, zorder=4, label="+ residue")
        ax.scatter(neg_j + 0.5, neg_i + 0.5, marker="o", s=30,
                   facecolors="none", edgecolors="cyan", linewidths=1.5,
                   zorder=4, label="- residue")
        if len(pos_i) or len(neg_i):
            ax.legend(loc="upper right", framealpha=0.8, fontsize=8)

    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(M - 0.5, -0.5)
    ax.set_title(title)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    return ax
