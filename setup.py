from skbuild import setup

if __name__ == '__main__':
    setup(
        name='OffAxisHolographyCpp',
        version='1.0.0',
        author='Davis Garwood',
        description='Code for Performing Off-axis Holography.',
        packages=['cpp_off_axis_holo'],
        install_requires=['numpy', 'scipy', 'pyfftw', 'opencv-python']
    )