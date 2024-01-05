cmake_minimum_required(VERSION 3.12)
project(dft_lib)

# Find Python Dev
find_package(Python COMPONENTS Development)

# Set the path to the pybind11 source directory
set(PYBIND11_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pybind11)

# Specify the path to the Python include directory
include_directories(${Python_INCLUDE_DIRS})

# Specify the path to the pybind11 headers
include_directories(${PYBIND11_DIR}/include)

# Add the pybind11 source files
#add_library(dft_lib MODULE src/DFT.cpp src/BindDFT.cpp)
add_library(dft_lib MODULE src/DFT.cpp)

# Set filename
set_target_properties(dft_lib PROPERTIES PREFIX "" OUTPUT_NAME "dft_lib" SUFFIX ".so")

# Set the output directory for the compiled module
set_target_properties(dft_lib PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Set needed libraries
target_link_libraries(dft_lib PRIVATE -lfftw3 -lfftw3_threads -lm)