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

class AutoFocusSearch:
    def __init__(self, chunk_size, num_chunks=1, num_workers=None):
        assert num_chunks > 0, "There must be at least 1 chunk."
        assert chunk_size > 0, "Chunk size must be at least 1."

        self.num_workers = num_workers or os.cpu_count()
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size

    def _make_zaxis(self, zmin, zmax):
        n = self.chunk_size * self.num_chunks
        return np.linspace(zmin, zmax, n, endpoint=False)

    def _chunk_indices(self, zmin, zmax):
        zaxis = self._make_zaxis(zmin, zmax)
        for i in range(0, len(zaxis), self.chunk_size):
            yield i, i + self.chunk_size

    @staticmethod
    def _parallel_propagate(field, z, wl, sz, start, nb=0, threads=1):
        m, n = field.shape
        prop = AngularSpectrum(m, n, wl, sz, nb=nb, threads=threads)
        prop.stack_arrays(len(z))
        field = np.stack([field] * len(z))
        prop_field = prop(field, z)
        return start, AMP(prop_field)

    def parallel_search(self, field, zmin, zmax, wl, sz, nb=0, threads=1):
        zaxis = self._make_zaxis(zmin, zmax)
        results = np.empty(self.num_chunks*self.chunk_size, dtype=float)
        chunks = list(self._chunk_indices(zmin, zmax))

        def task(args):
            start, end = args
            z = zaxis[start:end]
            if self.chunk_size == 1:
                z = np.array([z])
            return self._parallel_propagate(field, z, wl, sz, start, nb, threads)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for start, chunk_result in executor.map(task, chunks):
                results[start:start + self.chunk_size] = chunk_result
        return zaxis, results

    def serial_search(self, field, zmin, zmax, wl, sz, nb=0, threads=1):
        zaxis = self._make_zaxis(zmin, zmax)
        results = np.empty(self.num_chunks * self.chunk_size, dtype=float)

        m, n = field.shape
        prop = AngularSpectrum(m, n, wl, sz, nb=nb, threads=threads)
        prop.stack_arrays(self.chunk_size)

        stacked_field = np.stack([field] * self.chunk_size)
        for start, stop in self._chunk_indices(zmin, zmax):
            fieldz = prop(stacked_field, zaxis[start:stop])
            print(AMP(fieldz))
            results[start:stop] = AMP(fieldz)
        return zaxis, results