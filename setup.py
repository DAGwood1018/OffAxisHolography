from setuptools import setup

if __name__ == '__main__':
    setup(
        name='OffAxisHolography',
        version='2.2.0',
        author='Davis Garwood',
        description='Code for Performing Off-axis Holography.',
        packages=['py_off_axis_holo'],
        install_requires=['numpy', 'scipy', 'numba', 'pyfftw', 'scikit-image', 'opencv']
    )