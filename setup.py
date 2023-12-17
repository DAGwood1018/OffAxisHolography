import numpy as np
from setuptools import setup
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

libs = ['m', 'fftw3']
args = ['-std=c99', '-O3']
sources = ['cy_offaxis_holo/discrete_transforms.pyx', 'cy_offaxis_holo/fft_stuff.c']
include = ['include', np.get_include()]
linkerargs = ['-Wl,-rpath,lib']
libdirs = ['lib']

extensions = [
    Extension("cy_offaxis_holo",
              sources=sources,
              include_dirs=include,
              libraries=libs,
              library_dirs=libdirs,
              extra_compile_args=args,
              extra_link_args=linkerargs)
]

if __name__ == '__main__':
    setup(
        name='OffAxisHolography',
        version='1.0.0',
        author='Davis Garwood',
        description='Code for Performing Off-axis Holography.',
        packages=['py_offaxis_holo', 'cy_offaxis_holo', 'off_axis_utils'],
        ext_modules=cythonize(extensions),
        install_requires=['numpy', 'scipy', 'pyfftw', 'numba', 'cython', 'opencv-python']
    )