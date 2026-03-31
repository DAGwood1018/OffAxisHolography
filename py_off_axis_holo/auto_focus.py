import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from py_off_axis_holo.field_propagation import AngularSpectrum

def AMP(field):
    if len(field.shape) < 2:
        return np.sum(np.abs(field))
    return np.sum(np.sum(np.sqrt(np.abs(field)), axis=-1), axis=-1)

class AutoFocusSearch:
    def __init__(self, num_workers=None, chunk_size=None):
        """
        num_workers : number of processes
        chunk_size  : size of chunks
        """

        self.num_workers = num_workers or os.cpu_count()
        self.chunk_size = chunk_size or None

    def __call__(self, field, zaxis, wl, sz, nb=0, threads=1):
        if self.chunk_size is None:
            self.chunk_size = self._auto_chunk_size(zaxis)
        return self.compute(field, zaxis, wl, sz, nb=nb, threads=threads)

    def _auto_chunk_size(self, zaxis):
        n = len(zaxis)
        return max(1, n // (self.num_workers * 2))

    def _chunk_indices(self, zaxis):
        for i in range(0, len(zaxis), self.chunk_size):
            yield i, i + self.chunk_size

    @staticmethod
    def _parallel_propagate(field, z, wl, sz, start, nb=0, threads=1):
        m, n = field.shape
        prop = AngularSpectrum(m, n, wl, sz, nb=nb, threads=threads)
        if not isinstance(z, (int, float)):
            if len(z) > 1:
                prop.stack_arrays(len(z))
                field = np.stack([field] * len(z))

        prop_field = prop(field, z)
        return start, AMP(prop_field)

    def compute(self, field, zaxis, wl, sz, nb=0, threads=1):
        results = np.empty_like(zaxis, dtype=float)
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for start, end in self._chunk_indices(zaxis):
                z = zaxis[start:end]
                futures.append(
                    executor.submit(self._parallel_propagate,
                        field, z, wl, sz, start,
                        nb=nb, threads=threads
                    )
                )

            for future in as_completed(futures):
                start, chunk_result = future.result()
                results[start:start + len(chunk_result)] = chunk_result
        return zaxis, results
