import numpy as np
from setuptools import setup
from Cython.Build import cythonize

if __name__ == '__main__':
    setup(
        name='OffAxisHolography',
        version='1.0.0',
        author='Davis Garwood',
        description='Code for Performing Off-axis Holography.',
        packages=['py_offaxis_holo', 'cy_offaxis_holo', 'off_axis_utils'],
        ext_modules=cythonize("./cy_offaxis_holo/array_padding.pyx", language_level="3", include_path=[np.get_include()]),
        install_requires=['numpy', 'scipy', 'pyfftw', 'numba', 'cython', 'opencv-python']
    )