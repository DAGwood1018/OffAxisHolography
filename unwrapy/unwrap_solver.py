from unwrapy.mcf_branch_cuts import build_mcf_arcs, solve_mcf, extract_cuts
from unwrapy.phase_residues import residue_density
from unwrapy.unwrap_methods import floodfill, biased_floodfill


def phase_unwrap(wrapped_phase, residues, cost_h, cost_v, ref=(0, 0), cap=None,
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
        regions. If None, fall back to a plain BFS instead.

    Returns
    -------
    unwrapped : (M, N) float array
    info : dict with 'k_h', 'k_v', 'n_cut_edges', 'total_cost', 'cap',
        'density' (the residue-density map used to order the flood fill,
        or None if density_sigma is None).
    """
    M, N = wrapped_phase.shape
    tails, heads, costs, supplies, shape = build_mcf_arcs(residues, cost_h, cost_v)
    net_flow = solve_mcf(tails, heads, costs, supplies)
    k_h, k_v = extract_cuts(net_flow, shape, M, N)

    if density_sigma is None:
        unwrapped = floodfill(wrapped_phase, k_h, k_v, ref=ref)
    else:
        density = residue_density(residues, sigma=density_sigma)
        unwrapped = biased_floodfill(wrapped_phase, k_h, k_v, density, ref=ref)
    return unwrapped, k_h, k_v
