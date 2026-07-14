import heapq
import itertools
from collections import deque
import numpy as np
from numba import njit


@njit(cache=True)
def wrap_to_pi(x):
    """
    Wrap phase to (-pi, pi].
    """
    return (x + np.pi) % (2 * np.pi) - np.pi


def floodfill(wrapped_phase, k_h, k_v, ref=(0, 0)):
    """
    Flood-fill integration of the curl-free corrected gradient field.
    """

    M, N = wrapped_phase.shape
    unwrapped = np.zeros((M, N), dtype=np.float64)
    visited = np.zeros((M, N), dtype=bool)

    ri, rj = ref
    unwrapped[ri, rj] = wrapped_phase[ri, rj]
    visited[ri, rj] = True

    dq = deque([(ri, rj)])
    while dq:
        i, j = dq.popleft()
        # right neighbor
        if j + 1 < N and not visited[i, j + 1]:
            d = wrap_to_pi(wrapped_phase[i, j + 1] - wrapped_phase[i, j])
            unwrapped[i, j + 1] = unwrapped[i, j] + d + 2 * np.pi * k_h[i, j]
            visited[i, j + 1] = True
            dq.append((i, j + 1))
        # left neighbor
        if j - 1 >= 0 and not visited[i, j - 1]:
            d = wrap_to_pi(wrapped_phase[i, j - 1] - wrapped_phase[i, j])
            unwrapped[i, j - 1] = unwrapped[i, j] + d - 2 * np.pi * k_h[i, j - 1]
            visited[i, j - 1] = True
            dq.append((i, j - 1))
        # down neighbor
        if i + 1 < M and not visited[i + 1, j]:
            d = wrap_to_pi(wrapped_phase[i + 1, j] - wrapped_phase[i, j])
            unwrapped[i + 1, j] = unwrapped[i, j] + d + 2 * np.pi * k_v[i, j]
            visited[i + 1, j] = True
            dq.append((i + 1, j))
        # up neighbor
        if i - 1 >= 0 and not visited[i - 1, j]:
            d = wrap_to_pi(wrapped_phase[i - 1, j] - wrapped_phase[i, j])
            unwrapped[i - 1, j] = unwrapped[i, j] + d - 2 * np.pi * k_v[i - 1, j]
            visited[i - 1, j] = True
            dq.append((i - 1, j))

    return unwrapped


def biased_floodfill(wrapped_phase, k_h, k_v, density, ref=(0, 0)):
    """
    Best-first (priority) flood-fill integration of the curl-free corrected
    gradient field, expanding low residue-density (high-confidence) regions
    before high-density (near branch cut / decorrelated) regions.

    Why this matters despite the field being curl-free (i.e. mathematically
    path-independent): a plain BFS can wander into a noisy/decorrelated
    patch early and grow its unwrapped "frontier" outward from there, so if
    there's ever any imperfection in the correction field -- a masked/NaN
    region, a disconnected component, float accumulation over a very long
    path, or a capacity/tolerance edge case -- the resulting error tends to
    contaminate whatever got visited nearby in BFS order, which is fairly
    arbitrary. Growing outward from the reference through the cleanest
    regions first instead builds a reliable "backbone" via the shortest,
    lowest-risk paths, and only reaches into noisy regions last and from
    the most locally-trusted direction available at that point -- so any
    residual error stays localized to the noisy regions themselves instead
    of spreading through the clean bulk of the image.

    Parameters
    ----------
    wrapped_phase : (M, N) float array
    k_h, k_v : jump-count arrays from flow_to_jumps / mcf_unwrap's info
    density : (M, N) float array from compute_residue_density (or any other
        per-pixel "riskiness" map you want to prioritize by -- e.g. you
        could pass 1/contrast instead).
    ref : reference pixel, held fixed at its wrapped value.

    Returns
    -------
    unwrapped : (M, N) float array
    """
    M, N = wrapped_phase.shape
    unwrapped = np.zeros((M, N), dtype=np.float64)
    visited = np.zeros((M, N), dtype=bool)  # "finalized" / expanded
    queued = np.zeros((M, N), dtype=bool)

    ri, rj = ref
    unwrapped[ri, rj] = wrapped_phase[ri, rj]
    queued[ri, rj] = True

    counter = itertools.count()  # stable tie-breaking, avoids comparing (i,j) tuples
    heap = [(float(density[ri, rj]), next(counter), ri, rj)]

    while heap:
        _, _, i, j = heapq.heappop(heap)
        if visited[i, j]:
            continue
        visited[i, j] = True

        # right
        if j + 1 < N and not queued[i, j + 1]:
            d = wrap_to_pi(wrapped_phase[i, j + 1] - wrapped_phase[i, j])
            unwrapped[i, j + 1] = unwrapped[i, j] + d + 2 * np.pi * k_h[i, j]
            queued[i, j + 1] = True
            heapq.heappush(heap, (float(density[i, j + 1]), next(counter), i, j + 1))
        # left
        if j - 1 >= 0 and not queued[i, j - 1]:
            d = wrap_to_pi(wrapped_phase[i, j - 1] - wrapped_phase[i, j])
            unwrapped[i, j - 1] = unwrapped[i, j] + d - 2 * np.pi * k_h[i, j - 1]
            queued[i, j - 1] = True
            heapq.heappush(heap, (float(density[i, j - 1]), next(counter), i, j - 1))
        # down
        if i + 1 < M and not queued[i + 1, j]:
            d = wrap_to_pi(wrapped_phase[i + 1, j] - wrapped_phase[i, j])
            unwrapped[i + 1, j] = unwrapped[i, j] + d + 2 * np.pi * k_v[i, j]
            queued[i + 1, j] = True
            heapq.heappush(heap, (float(density[i + 1, j]), next(counter), i + 1, j))
        # up
        if i - 1 >= 0 and not queued[i - 1, j]:
            d = wrap_to_pi(wrapped_phase[i - 1, j] - wrapped_phase[i, j])
            unwrapped[i - 1, j] = unwrapped[i, j] + d - 2 * np.pi * k_v[i - 1, j]
            queued[i - 1, j] = True
            heapq.heappush(heap, (float(density[i - 1, j]), next(counter), i - 1, j))
    return unwrapped
