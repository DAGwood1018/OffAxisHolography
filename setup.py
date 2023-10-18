from setuptools import setup

if __name__ == '__main__':
    setup(
        name='OffAxisHolography',
        version='1.0.0',
        author='Davis Garwood',
        description='Code for Performing Off-axis Holography.',
        packages=['off_axis_holography'],
        install_requires=['numpy', 'scipy', 'pyfftw', 'numba', 'cython']
    )