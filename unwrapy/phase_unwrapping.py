import numpy as np
from numba import njit


def wrap_to_pi(phase):
    """
    Wrap phase to (-pi, pi].
    """
    return (phase + np.pi) % (2 * np.pi) - np.pi


def calc_residues(wrapped_phase):
    """
    Compute phase residues using the standard 2×2 circulation.

    Parameters
    ----------
    wrapped_phase : (M, N) ndarray
        Wrapped phase image in radians.

    Returns
    -------
    residues : (M-1, N-1) ndarray of int8
        Values are {-1, 0, +1}.
    """

    p00 = wrapped_phase[:-1, :-1]
    p01 = wrapped_phase[:-1, 1:]
    p11 = wrapped_phase[1:, 1:]
    p10 = wrapped_phase[1:, :-1]

    d1 = wrap_to_pi(p01 - p00)
    d2 = wrap_to_pi(p11 - p01)
    d3 = wrap_to_pi(p10 - p11)
    d4 = wrap_to_pi(p00 - p10)

    residues = np.rint((d1 + d2 + d3 + d4) / (2 * np.pi)).astype(np.int8)
    return residues


def calc_costs(wrapped_phase, coherence, cost_scale=1000.0, eps=1e-6):
    """
    Compute SNAPHU-inspired edge costs for MCF phase unwrapping.

    Parameters
    ----------
    wrapped_phase : (M, N) array
        Wrapped phase in radians.

    coherence : (M, N) array
        Coherence or proportional contrast (0..1).

    cost_scale : float
        Scaling factor for integer costs.

    eps : float
        Numerical stability.

    Returns
    -------
    cost_x : (M, N-1) int32
    cost_y : (M-1, N) int32
    """

    # ------------------------------------------------------------
    # 1. coherence → phase variance (single-look model)
    # ------------------------------------------------------------

    gamma = np.clip(coherence, eps, 0.9999)

    sigma2 = (1.0 - gamma**2) / (2.0 * gamma**2)

    # ------------------------------------------------------------
    # 2. edge variances (sum of independent contributions)
    # ------------------------------------------------------------

    sigma2_x = sigma2[:, :-1] + sigma2[:, 1:]
    sigma2_y = sigma2[:-1, :] + sigma2[1:, :]

    # ------------------------------------------------------------
    # 3. wrapped phase differences on edges
    # ------------------------------------------------------------

    dphi_x = wrap_to_pi(wrapped_phase[:, 1:] - wrapped_phase[:, :-1])
    dphi_y = wrap_to_pi(wrapped_phase[1:, :] - wrapped_phase[:-1, :])

    # ------------------------------------------------------------
    # 4. negative log-likelihood cost (k = 0 assumption)
    # ------------------------------------------------------------

    cost_x = (dphi_x ** 2) / (2.0 * sigma2_x + eps)
    cost_y = (dphi_y ** 2) / (2.0 * sigma2_y + eps)

    # ------------------------------------------------------------
    # 5. normalize for solver stability
    # ------------------------------------------------------------

    mean_cost = 0.5 * (cost_x.mean() + cost_y.mean() + eps)

    cost_x /= mean_cost
    cost_y /= mean_cost

    # ------------------------------------------------------------
    # 6. scale to integer costs (OR-Tools requirement)
    # ------------------------------------------------------------

    cost_x = np.maximum(1, np.rint(cost_scale * cost_x)).astype(np.int32)
    cost_y = np.maximum(1, np.rint(cost_scale * cost_y)).astype(np.int32)

    return cost_x, cost_y


@njit
def wrap_diff(a, b):
    """
    Wrap difference to (-pi, pi].
    """
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


@njit
def floodfill_unwrap(wrapped_phase, cut_x, cut_y):
    """
    Flood-fill phase unwrapping using branch cuts.

    Parameters
    ----------
    wrapped_phase : (M, N) float64
    cut_x : (M, N-1) bool
    cut_y : (M-1, N) bool

    Returns
    -------
    unwrapped : (M, N) float64
    """

    M, N = wrapped_phase.shape
    unwrapped = np.zeros((M, N), dtype=np.float64)
    visited = np.zeros((M, N), dtype=np.uint8)

    # ------------------------------------------------------------
    # manual stack (Numba-safe)
    # ------------------------------------------------------------
    stack_r = np.empty(M * N, dtype=np.int32)
    stack_c = np.empty(M * N, dtype=np.int32)
    top = 0

    # start at (0,0)
    stack_r[top] = 0
    stack_c[top] = 0
    top += 1

    unwrapped[0, 0] = wrapped_phase[0, 0]
    visited[0, 0] = 1

    # ------------------------------------------------------------
    # DFS flood-fill
    # ------------------------------------------------------------
    while top > 0:

        top -= 1
        r = stack_r[top]
        c = stack_c[top]

        base_val = unwrapped[r, c]

        # -------------------------
        # RIGHT
        # -------------------------
        if c + 1 < N and visited[r, c + 1] == 0:
            if cut_x[r, c] == 0:
                unwrapped[r, c + 1] = base_val + wrap_diff(
                    wrapped_phase[r, c + 1],
                    wrapped_phase[r, c]
                )
                visited[r, c + 1] = 1
                stack_r[top] = r
                stack_c[top] = c + 1
                top += 1

        # -------------------------
        # LEFT
        # -------------------------
        if c - 1 >= 0 and visited[r, c - 1] == 0:
            if cut_x[r, c - 1] == 0:
                unwrapped[r, c - 1] = base_val + wrap_diff(
                    wrapped_phase[r, c - 1],
                    wrapped_phase[r, c]
                )
                visited[r, c - 1] = 1
                stack_r[top] = r
                stack_c[top] = c - 1
                top += 1

        # -------------------------
        # DOWN
        # -------------------------
        if r + 1 < M and visited[r + 1, c] == 0:
            if cut_y[r, c] == 0:
                unwrapped[r + 1, c] = base_val + wrap_diff(
                    wrapped_phase[r + 1, c],
                    wrapped_phase[r, c]
                )
                visited[r + 1, c] = 1
                stack_r[top] = r + 1
                stack_c[top] = c
                top += 1

        # -------------------------
        # UP
        # -------------------------
        if r - 1 >= 0 and visited[r - 1, c] == 0:
            if cut_y[r - 1, c] == 0:
                unwrapped[r - 1, c] = base_val + wrap_diff(
                    wrapped_phase[r - 1, c],
                    wrapped_phase[r, c]
                )
                visited[r - 1, c] = 1
                stack_r[top] = r - 1
                stack_c[top] = c
                top += 1

    return unwrapped