import numpy as np
from warnings import warn
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from py_off_axis_holo.field_propagation import AngularSpectrum

def AMP(field):
    if np.ndim(field) == 2:
        return np.sum(np.sqrt(np.abs(field)), axis=(0, 1))
    elif np.ndim(field) == 3:
        return np.sum(np.sqrt(np.abs(field)), axis=(1, 2))
    else:
        warn("Invalid field shape.")
        return None

def SPEC(Fh):
    if np.ndim(Fh) == 2:
        return np.sum(np.log(1 + np.sqrt(np.abs(Fh))), axis=(0, 1)) / (Fh.shape[0]*Fh.shape[1])
    elif np.ndim(Fh) == 3:
        return np.sum(np.log(1 + np.sqrt(np.abs(Fh))), axis=(1, 2)) / (Fh.shape[1]*Fh.shape[2])
    else:
        warn("Invalid field shape.")
        return None

class AutoFocusSearch:
    def __init__(self, chunk_size, num_chunks=1, num_workers=None):
        assert num_chunks > 0, "There must be at least 1 chunk."
        assert chunk_size > 0, "Chunk size must be at least 1."

        self.num_workers = num_workers or os.cpu_count()
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size

    def __call__(self, field, zmin, zmax, wl, sz, threads=1, width=10e-3, **kwargs):
        zaxis, results = self.broad_search(field, zmin, zmax, wl, sz, threads=threads)
        z0 = zaxis[np.argmin(results)]
        return self.golden_section_search(field, z0-width/2, z0+width/2, wl, sz, threads=threads, **kwargs)

    def _make_zaxis(self, zmin, zmax):
        n = self.chunk_size * self.num_chunks
        return np.linspace(zmin, zmax, n, endpoint=False)

    def _chunk_indices(self, zmin, zmax):
        zaxis = self._make_zaxis(zmin, zmax)
        for i in range(0, len(zaxis), self.chunk_size):
            yield i, i + self.chunk_size

    @staticmethod
    def _parallel_propagate(field, z, wl, sz, start, threads=1):
        m, n = field.shape
        prop = AngularSpectrum(m, n, wl, sz, nb=0, threads=threads)
        results = []
        for i in range(len(z)):
            fieldz = prop(field, z[i])
            results.append(AMP(fieldz))
        return start, np.array(results)

    def broad_search(self, field, zmin, zmax, wl, sz, threads=1):
        zaxis = self._make_zaxis(zmin, zmax)
        results = np.empty(self.num_chunks*self.chunk_size, dtype=float)
        chunks = list(self._chunk_indices(zmin, zmax))

        def task(args):
            start, end = args
            z = zaxis[start:end]
            if self.chunk_size == 1:
                z = np.array([z])
            return self._parallel_propagate(field, z, wl, sz, start, threads)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for start, chunk_result in executor.map(task, chunks):
                results[start:start + self.chunk_size] = chunk_result
        return zaxis, results

    def golden_section_search(self, field, z_a, z_b, wl, sz, threads=1, tol=1e-5, max_iter=100):
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        resphi = 2 - phi  # 1/phi^2

        m, n = field.shape
        prop = AngularSpectrum(m, n, wl, sz, nb=0, threads=threads)
        z_c = z_a + resphi * (z_b - z_a)
        z_d = z_b - resphi * (z_b - z_a)

        fc, fd = AMP(prop(field, z_c)), AMP(prop(field, z_d))
        for _ in range(max_iter):
            if abs(z_b - z_a) < tol:
                break

            if fc < fd:
                z_b = z_d
                z_d = z_c
                fd = fc

                z_c = z_a + resphi * (z_b - z_a)
                fc = AMP(prop(field, z_c))
            else:
                z_a = z_c
                z_c = z_d
                fc = fd

                z_d = z_b - resphi * (z_b - z_a)
                fd = AMP(prop(field, z_d))
        return (z_b + z_a) / 2