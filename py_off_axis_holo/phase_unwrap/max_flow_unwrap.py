"""
PUMA — Phase Unwrapping via Max-flow/min-cut Algorithm
Bioucas-Dias & Valadao (2007), implemented with OR-Tools SimpleMaxFlow.

Handles the convex Lp potentials V(x) = |x|**p, p = 1, for which each
"+-1 jump" binary move is provably submodular and solved EXACTLY by
a single max-flow/min-cut call (Ishikawa's construction).

NOT implemented: non-convex potentials (0 < p < 1). Those need QPBO,
since submodularity is no longer guaranteed.

Requires: pip install ortools numpy
"""

import numpy as np
from ortools.graph.python import max_flow

from .unwrap_utils import wrap_to_pi


class UnwrapPUMA:
    """
    2D phase unwrapper using PUMA (graph-cut based) with the L1 potential
    V(x) = |x| (p = 1).

    Usage
    -----
    unwrapper = PumaUnwrapper(connectivity=4, max_outer_iters=200)
    phi, k = unwrapper.unwrap(psi)
    """

    def __init__(self, min_connectivity=True, max_iters=200):
        if min_connectivity:
            self.connectivity = 4
        else:
            self.connectivity = 8
        self.max_iters = max_iters

    def __call__(self, psi):
        unwrapped_phase, _ = self.unwrap(psi)
        return unwrapped_phase

    @staticmethod
    def _total_energy(k, i, j, n_ij):
        return np.abs(k[i] - k[j] + n_ij).sum()

    @staticmethod
    def _build_edges(shape, connectivity=4):
        """(E,2) int array of neighbor pixel pairs (i, j), linear indices."""

        H, W = shape
        idx = np.arange(H * W).reshape(H, W)
        edges = [
            np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], axis=1),   # horiz
            np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], axis=1),   # vert
        ]
        if connectivity == 8:
            edges.append(np.stack([idx[:-1, :-1].ravel(), idx[1:, 1:].ravel()], axis=1))
            edges.append(np.stack([idx[:-1, 1:].ravel(), idx[1:, :-1].ravel()], axis=1))
        return np.concatenate(edges, axis=0)

    def _solve_max_flow(self, k, i_idx, j_idx, n_ij, direction, N):
        """
        One binary move: each pixel may jump by `direction` (+1 or -1).
        Optimal binary assignment found exactly via max-flow, for the
        L1 potential V(x) = |x|.
        Returns (k_new, changed).
        """

        x = k[i_idx] - k[j_idx] + n_ij  # current pairwise state (integer)

        # For V(x) = |x|, A = D = |x| regardless of direction.
        A = np.abs(x)
        if direction == 1:
            B, C = np.abs(x - 1), np.abs(x + 1)
        else:
            B, C = np.abs(x + 1), np.abs(x - 1)
        D = A

        # theta(bi,bj) = A + (B-A)*bj + (D-B)*bi + K'*bi*(1-bj),  K' = B+C-A-D >= 0
        # (K' >= 0 automatically since |.| is convex, i.e. p = 1)
        Kp = B + C - A - D

        unary1 = np.zeros(N, dtype=np.int64)   # cost contribution for bi = 1
        np.add.at(unary1, i_idx, D - B)
        np.add.at(unary1, j_idx, B - A)

        smf = max_flow.SimpleMaxFlow()
        SOURCE, SINK = N, N + 1

        starts, ends, caps = [], [], []

        pos = np.nonzero(unary1 > 0)[0]
        if pos.size:
            starts.append(np.full(pos.size, SOURCE, dtype=np.int32))
            ends.append(pos.astype(np.int32))
            caps.append(unary1[pos])

        neg = np.nonzero(unary1 < 0)[0]
        if neg.size:
            starts.append(neg.astype(np.int32))
            ends.append(np.full(neg.size, SINK, dtype=np.int32))
            caps.append(-unary1[neg])

        pw = np.nonzero(Kp > 0)[0]
        if pw.size:
            starts.append(j_idx[pw].astype(np.int32))   # directed j -> i
            ends.append(i_idx[pw].astype(np.int32))
            caps.append(Kp[pw])

        if starts:
            smf.add_arcs_with_capacity(
                np.concatenate(starts), np.concatenate(ends), np.concatenate(caps)
            )

        status = smf.solve(SOURCE, SINK)
        if status != smf.OPTIMAL:
            raise RuntimeError(f"max-flow solve failed, status={status}")

        source_side = np.asarray(smf.get_source_side_min_cut())
        in_source = np.zeros(N, dtype=bool)
        valid = source_side[source_side < N]
        in_source[valid] = True
        b = (~in_source).astype(np.int64)

        k_new = k + direction * b
        return k_new, not np.array_equal(k_new, k)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def unwrap(self, psi):
        """
        Unwrap a 2D wrapped-phase image.

        Parameters
        ----------
        psi : (H, W) ndarray, wrapped phase assumed in (-pi, pi]

        Returns
        -------
        phi : (H, W) ndarray, unwrapped phase = psi + 2*pi*k
        k   : (H, W) ndarray of int, wrap-count field
        """

        psi = np.asarray(psi, dtype=np.float64)
        H, W = psi.shape
        N = H * W

        edges = self._build_edges((H, W), self.connectivity)
        i_idx, j_idx = edges[:, 0], edges[:, 1]

        raw_diff = psi.ravel()[i_idx] - psi.ravel()[j_idx]
        wrapped_diff = wrap_to_pi(raw_diff)
        n_ij = np.round((raw_diff - wrapped_diff) / (2 * np.pi)).astype(np.int64)

        k = np.zeros(N, dtype=np.int64)
        prev_energy = self._total_energy(k, i_idx, j_idx, n_ij)
        for outer in range(self.max_iters):
            improved = False
            for direction in (1, -1):
                k, changed = self._solve_max_flow(k, i_idx, j_idx, n_ij, direction, N)
                improved = improved or changed
            new_energy = self._total_energy(k, i_idx, j_idx, n_ij)

            if not improved or new_energy >= prev_energy:
                break
            prev_energy = new_energy

        phi = psi + 2 * np.pi * k.reshape(H, W)
        return phi, k.reshape(H, W)

