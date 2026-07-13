import numpy as np
import networkx as nx
from ortools.graph.python import min_cost_flow
from matplotlib import pyplot as plt
from collections import deque

from unwrapy.phase_unwrapping import wrap_diff


def solve_mcf(residues, cost_h, cost_v):
    """
    Solve the minimum-cost flow problem using a single super node
    representing the image boundary.

    Parameters
    ----------
    residues : (R, C) ndarray
        Residue/supply values.

    cost_h : (R, C-1) ndarray
        Horizontal edge costs.

    cost_v : (R-1, C) ndarray
        Vertical edge costs.

    Returns
    -------
    smcf : SimpleMinCostFlow
    arc_indices : ndarray
    metadata : dict
    """

    R, C = residues.shape

    node_ids = np.arange(R * C, dtype=np.int32).reshape(R, C)
    super_node = R * C
    n_nodes = super_node + 1

    INF = np.int64(n_nodes)

    arcs = []

    HORIZONTAL = 0
    VERTICAL = 1
    BORDER = 2

    orientation = []
    row_idx = []
    col_idx = []

    super_edges = []

    # ------------------------------------------------------------
    # Interior horizontal edges
    # ------------------------------------------------------------

    for r in range(R):
        for c in range(C - 1):

            u = node_ids[r, c]
            v = node_ids[r, c + 1]
            w = int(cost_h[r, c])

            arcs.append((u, v, w, INF))
            orientation.append(HORIZONTAL)
            row_idx.append(r)
            col_idx.append(c)

            arcs.append((v, u, w, INF))
            orientation.append(HORIZONTAL)
            row_idx.append(r)
            col_idx.append(c)

    # ------------------------------------------------------------
    # Interior vertical edges
    # ------------------------------------------------------------

    for r in range(R - 1):
        for c in range(C):

            u = node_ids[r, c]
            v = node_ids[r + 1, c]
            w = int(cost_v[r, c])

            arcs.append((u, v, w, INF))
            orientation.append(VERTICAL)
            row_idx.append(r)
            col_idx.append(c)

            arcs.append((v, u, w, INF))
            orientation.append(VERTICAL)
            row_idx.append(r)
            col_idx.append(c)

    # ------------------------------------------------------------
    # Super-node connections
    # ------------------------------------------------------------

    border_nodes = []

    # top and bottom rows
    for c in range(C):
        border_nodes.append((0, c))
        if R > 1:
            border_nodes.append((R - 1, c))

    # left and right columns
    for r in range(1, R - 1):
        border_nodes.append((r, 0))
        if C > 1:
            border_nodes.append((r, C - 1))

    for r, c in border_nodes:
        u = node_ids[r, c]

        # border -> super
        arcs.append((u, super_node, 0, INF))
        orientation.append(BORDER)
        row_idx.append(r)
        col_idx.append(c)

        # super -> border
        arcs.append((super_node, u, 0, INF))
        orientation.append(BORDER)
        row_idx.append(r)
        col_idx.append(c)

        super_edges.append((r, c))

    # ------------------------------------------------------------
    # Convert arc arrays
    # ------------------------------------------------------------

    arcs = np.asarray(arcs, dtype=np.int64)
    tails = arcs[:, 0].astype(np.int32)
    heads = arcs[:, 1].astype(np.int32)
    costs = arcs[:, 2].astype(np.int64)
    capacities = arcs[:, 3].astype(np.int64)

    # ------------------------------------------------------------
    # Supplies
    # ------------------------------------------------------------

    supplies = np.zeros(n_nodes, dtype=np.int64)
    for r in range(R):
        for c in range(C):
            supplies[node_ids[r, c]] = residues[r, c]
    supplies[super_node] = -supplies[:-1].sum()

    # ------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------

    smcf = min_cost_flow.SimpleMinCostFlow()

    arc_indices = smcf.add_arcs_with_capacity_and_unit_cost(
        tails,
        heads,
        capacities,
        costs,
    )

    smcf.set_nodes_supplies(
        np.arange(n_nodes, dtype=np.int32),
        supplies,
    )

    status = smcf.solve()
    if status != smcf.OPTIMAL:
        raise RuntimeError(f"MCF failed (status={status})")

    metadata = {
        "R": R,
        "C": C,
        "node_ids": node_ids,
        "super_node": super_node,
        "super_edges": super_edges,
        "orientation": np.asarray(orientation, dtype=np.int8),
        "row_idx": np.asarray(row_idx, dtype=np.int32),
        "col_idx": np.asarray(col_idx, dtype=np.int32),
    }

    return smcf, arc_indices, metadata

def branch_cut_graph(smcf, arc_indices, metadata, residues=None):
    """
    Construct a NetworkX graph of branch cuts from the solved MCF.

    Parameters
    ----------
    smcf : SimpleMinCostFlow
    arc_indices : ndarray
    metadata : dict
        Returned by solve_mcf().
    residues : ndarray, optional
        Residue array. If given, stored as node attributes.

    Returns
    -------
    G : nx.Graph
        Physical image lattice with edge attribute 'cut'
        and node attribute 'border_cut'.
    """

    R = metadata["R"]
    C = metadata["C"]

    orientation = metadata["orientation"]
    row_idx = metadata["row_idx"]
    col_idx = metadata["col_idx"]

    HORIZONTAL = 0
    VERTICAL = 1
    BORDER = 2

    flows = smcf.flows(arc_indices)

    G = nx.Graph()

    # ------------------------------------------------------------
    # Add nodes
    # ------------------------------------------------------------

    for r in range(R):
        for c in range(C):

            attrs = {
                "row": r,
                "col": c,
                "pos": (c, -r),
                "border_cut": False,
            }

            if residues is not None:
                attrs["residue"] = int(residues[r, c])

            G.add_node((r, c), **attrs)

    # ------------------------------------------------------------
    # Add every physical edge
    # ------------------------------------------------------------

    for r in range(R):
        for c in range(C - 1):
            G.add_edge((r, c), (r, c + 1),
                       cut=False,
                       flow=0)

    for r in range(R - 1):
        for c in range(C):
            G.add_edge((r, c), (r + 1, c),
                       cut=False,
                       flow=0)

    # ------------------------------------------------------------
    # Decode solution
    # ------------------------------------------------------------

    n_arcs = len(flows)

    for k in range(0, n_arcs, 2):

        # Recover the signed flow on this undirected edge
        fwd = flows[k]
        rev = flows[k + 1]

        net_flow = int(fwd) - int(rev)

        if net_flow == 0:
            continue

        typ = orientation[k]
        r = row_idx[k]
        c = col_idx[k]

        if typ == HORIZONTAL:
            G.edges[(r, c), (r, c + 1)]["cut"] = True
            G.edges[(r, c), (r, c + 1)]["flow"] = net_flow
        elif typ == VERTICAL:
            G.edges[(r, c), (r + 1, c)]["cut"] = True
            G.edges[(r, c), (r + 1, c)]["flow"] = net_flow
        elif typ == BORDER:
            G.nodes[(r, c)]["border_cut"] = True
            G.nodes[(r, c)]["border_flow"] = net_flow
    return G

def plot_branch_cuts2(G, cut_color="red", cut_width=2.5, figsize=(8, 8), show_residues=True, residue_size=10):
    """
    Plot branch cuts contained in a branch-cut graph.

    Parameters
    ----------
    G : nx.Graph
        Graph returned by branch_cut_graph().

    cut_color : str
        Color of branch cuts.

    cut_width : float
        Width of branch cuts.

    figsize : tuple
        Figure size.

    show_residues : bool
        Plot residue locations if available.

    residue_size : float
        Marker size for residues.
    """

    fig, ax = plt.subplots(figsize=figsize)

    # ------------------------------------------------------------
    # Node positions
    # ------------------------------------------------------------

    pos = nx.get_node_attributes(G, "pos")

    rows = [d["row"] for _, d in G.nodes(data=True)]
    cols = [d["col"] for _, d in G.nodes(data=True)]

    R = max(rows) + 1
    C = max(cols) + 1

    # ------------------------------------------------------------
    # Interior branch cuts
    # ------------------------------------------------------------

    cut_edges = [
        (u, v)
        for u, v, d in G.edges(data=True)
        if d.get("cut", False)
    ]

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=cut_edges,
        edge_color=cut_color,
        width=cut_width,
        ax=ax,
    )

    # ------------------------------------------------------------
    # Border cuts
    # ------------------------------------------------------------

    for node, data in G.nodes(data=True):

        if not data.get("border_cut", False):
            continue

        r = data["row"]
        c = data["col"]

        x, y = pos[node]

        # Draw to nearest image boundary
        if r == 0:
            ax.plot([x, x], [y, y + 0.5],
                    color=cut_color, lw=cut_width)

        elif r == R - 1:
            ax.plot([x, x], [y, y - 0.5],
                    color=cut_color, lw=cut_width)

        elif c == 0:
            ax.plot([x - 0.5, x], [y, y],
                    color=cut_color, lw=cut_width)

        elif c == C - 1:
            ax.plot([x, x + 0.5], [y, y],
                    color=cut_color, lw=cut_width)

    # ------------------------------------------------------------
    # Plot residues if available
    # ------------------------------------------------------------

    if show_residues:

        has_residues = any(
            "residue" in d for _, d in G.nodes(data=True)
        )

        if has_residues:

            pos_nodes = []
            neg_nodes = []

            for node, data in G.nodes(data=True):

                q = data.get("residue", 0)

                if q > 0:
                    pos_nodes.append(node)
                elif q < 0:
                    neg_nodes.append(node)

            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=pos_nodes,
                node_color="tab:blue",
                node_size=residue_size,
                edgecolors="k",
                ax=ax,
                label="+ residue",
            )

            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=neg_nodes,
                node_color="tab:red",
                node_size=residue_size,
                edgecolors="k",
                ax=ax,
                label="- residue",
            )

    # ------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------

    ax.set_xlim(-0.5, C - 0.5)
    ax.set_ylim(-(R - 0.5), 0.5)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.show()
