import numpy as np
from warnings import warn
from concurrent.futures import ThreadPoolExecutor
import os

from py_off_axis_holo.field_propagation import AngularSpectrum


def AMP(field):
    """
    Maximized for pure amplitude, minimized for pure phase

    :param field: ndarray of light field
    :return: AMP measure of field
    """

    if np.ndim(field) == 2:
        return np.sum(np.sqrt(np.abs(field)), axis=(0, 1))
    elif np.ndim(field) == 3:
        return np.sum(np.sqrt(np.abs(field)), axis=(1, 2))
    else:
        warn("Invalid field shape.")
        return None

def parallel_propagate(field, z, wl, sz, dz, start, threads=1, phase_only=False):
    m, n = field.shape
    prop = AngularSpectrum(m, n, wl, sz, nb=0, threads=threads)
    score, peaks, z_int = [], [], [z[0] - dz]

    z_int.extend(z)
    z_int.append(z_int[-1] + dz)
    for i in range(len(z_int)):
        fieldz = prop(field, z_int[i])
        score.append(-AMP(fieldz)) if phase_only else score.append(AMP(fieldz))

    min_point = np.argmin(score[1:-1])
    if score[min_point] < score[min_point + 1] and score[min_point] < score[min_point - 1]:
        poi = start + min_point
    else:
        poi = None
    return start, np.array(score)[1:-1], poi

def golden_section_search(field, z_a, z_b, wl, sz, threads=1, tol=1e-5, max_iter=1000, phase_only=False):
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    resphi = 2 - phi  # 1/phi^2

    m, n = field.shape
    prop = AngularSpectrum(m, n, wl, sz, nb=0, threads=threads)
    z_c = z_a + resphi * (z_b - z_a)
    z_d = z_b - resphi * (z_b - z_a)

    iters = 0
    fc, fd = AMP(prop(field, z_c)), AMP(prop(field, z_d))
    if phase_only:
        fc *= -1
        fd *= -1
    while abs(z_b - z_a) > tol:
        if iters > max_iter:
            break
        if fc < fd:
            z_b, z_d, fd = z_d, z_c, fc
            z_c = z_a + resphi * (z_b - z_a)
            fc = -AMP(prop(field, z_c)) if phase_only else AMP(prop(field, z_c))
        else:
            z_a, z_c, fc = z_c, z_d, fd
            z_d = z_b - resphi * (z_b - z_a)
            fd = -AMP(prop(field, z_d)) if phase_only else AMP(prop(field, z_d))
        iters += 1
    z0 = (z_b + z_a) / 2
    final_score = -AMP(prop(field, z0)) if phase_only else AMP(prop(field, z0))
    return z0, final_score


class AutoFocusSearch:
    def __init__(self, chunk_size, num_chunks=1, num_workers=None, field_type='phase'):
        assert num_chunks > 0, "There must be at least 1 chunk."
        assert chunk_size > 0, "Chunk size must be at least 1."

        if type(field_type) is not str:
            raise ValueError("Make field amplitude or phase only.")
        if not (field_type.lower()=='phase' or field_type.lower()=='amplitude'):
            raise ValueError("Make field amplitude or phase only.")
        self.num_workers = num_workers or os.cpu_count()
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
        self._type = field_type.lower()
        self._dz = 0

    def __call__(self, field, zmin, zmax, wl, sz, threads=1, width=10e-3, **kwargs):
        zaxis, results, poi = self.broad_search(field, zmin, zmax, wl, sz, threads=threads)
        phase_only = True if self._type == 'phase' else False
        potential_extrema, scores = [], []
        for p in poi:
            if p is not None:
                z0 = zaxis[p]
                potential_extreme, score = golden_section_search(field, z0 - width, z0 + width, wl, sz,
                                                                       threads=threads, phase_only=phase_only, **kwargs)
                potential_extrema.append(potential_extreme)
                scores.append(score)
        return zaxis, results, potential_extrema[np.argmin(scores)]

    def _make_zaxis(self, zmin, zmax):
        n = self.chunk_size * self.num_chunks
        self._dz = (zmax - zmin) / n
        return np.linspace(zmin, zmax, n, endpoint=False)

    def _chunk_indices(self, zmin, zmax):
        zaxis = self._make_zaxis(zmin, zmax)
        for i in range(0, len(zaxis), self.chunk_size):
            yield i, i + self.chunk_size

    def broad_search(self, field, zmin, zmax, wl, sz, threads=1):
        zaxis, peaks = self._make_zaxis(zmin, zmax), []
        results = np.empty(self.num_chunks * self.chunk_size, dtype=float)
        chunks = list(self._chunk_indices(zmin, zmax))

        def task(args):
            start, end = args
            z = zaxis[start:end]
            if self.chunk_size == 1:
                z = np.array([z])
            phase_only = True if self._type == 'phase' else False
            return parallel_propagate(field, z, wl, sz, self._dz, start, threads, phase_only=phase_only)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for start, chunk_result, peak in executor.map(task, chunks):
                results[start:start + self.chunk_size] = chunk_result
                peaks.append(peak)
        return zaxis, results, peaks
