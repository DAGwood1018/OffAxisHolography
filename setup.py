from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension
import os

# Get the absolute path of the directory containing the setup script
current_dir = os.path.abspath(os.path.dirname(__file__))

# Define the absolute path to the source files
source_files = [
    os.path.join(current_dir, "cpp_off_axis_holo", "src", "OffAxisFilter.cpp"),
    os.path.join(current_dir, "cpp_off_axis_holo", "src", "PhaseUnwrap.cpp"),
    os.path.join(current_dir, "cpp_off_axis_holo", "src", "OffAxisModule.cpp")
]

ext_modules = [
    Pybind11Extension(
       "off_axis_module",
        source_files,
        include_dirs=[
            "pybind11/include",  # Path to pybind11 headers
            "/usr/include/eigen3",  # Include Eigen
            "/usr/local/include"  # Include FFTW
        ],
        libraries=["fftw3", "fftw3_threads", "m", "opencv_core", "opencv_imgproc", "opencv_highgui"],
        library_dirs=["/usr/local/lib"],  # FFTW library directory
        define_macros=[('EIGEN_NO_DEBUG', '1')],  # Define Eigen macro
    ),
]

if __name__ == '__main__':
    setup(
        name='OffAxisHolography',
        version='1.0.0',
        author='Davis Garwood',
        description='Code for Performing Off-axis Holography.',
        packages=['py_off_axis_holo', 'cpp_off_axis_holo'],
        ext_modules=ext_modules,
        install_requires=['numpy', 'scipy', 'pyfftw', 'opencv-python']
    )