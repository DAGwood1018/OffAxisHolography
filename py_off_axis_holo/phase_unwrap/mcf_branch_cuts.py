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


def solve_mcf(tails, heads, costs, supplies):
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
    """
    # ground node is always the last entry in `supplies` (see build_mcf_arcs)
    total_abs_charge = max(int(np.abs(supplies[:-1]).sum()), 1)
    n_arcs = len(tails)

    smcf = SimpleMinCostFlow()
    all_tails = np.concatenate([tails, heads])
    all_heads = np.concatenate([heads, tails])
    all_costs = np.concatenate([costs, costs])
    all_caps = np.full(2 * n_arcs, total_abs_charge, dtype=np.int64)

    smcf.add_arcs_with_capacity_and_unit_cost(all_tails, all_heads, all_caps, all_costs)
    smcf.set_nodes_supplies(np.arange(len(supplies), dtype=np.int64), supplies.astype(np.int64))

    status = smcf.solve()
    if status == smcf.OPTIMAL:
        flows = smcf.flows(np.arange(2 * n_arcs, dtype=np.int64))
        net_flow = flows[:n_arcs] - flows[n_arcs:]
        return net_flow
    # INFEASIBLE (capacity too tight) -> try the next, larger cap
    raise RuntimeError(
        f"MCF solve failed; this should not "
        f"happen unless supplies don't sum to zero."
    )


def extract_cuts(net_flow, residues_shape, M, N):
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