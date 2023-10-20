from setuptools import setup
from Cython.Build import cythonize

if __name__ == '__main__':
    setup(
        name='OffAxisHolography',
        version='1.0.0',
        author='Davis Garwood',
        description='Code for Performing Off-axis Holography.',
        packages=['off_axis_holography'],
        #ext_modules=cythonize("primes.pyx", language_level="3"),
        install_requires=['numpy', 'scipy', 'pyfftw', 'numba', 'cython', 'opencv-python']
    )