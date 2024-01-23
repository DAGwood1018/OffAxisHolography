cmake_minimum_required(VERSION 3.12)
project(holography_bindings)

# Find Python Dev
find_package(Python COMPONENTS Development)

find_package(OpenCV REQUIRED)

# Set the path to the pybind11 source directory
set(PYBIND11_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pybind11)

# Set the path to the pybind11 source directory
set(PYBIND11_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pybind11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

# Specify the path to the Python include directory
include_directories(${Python_INCLUDE_DIRS})

# Specify the path to the pybind11 headers
include_directories(${PYBIND11_DIR}/include)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# <OffAxFilter>:

# Add the pybind11 source files
add_library(off_axis_filter MODULE src/OffAxFilter.cpp)

# Set filename
set_target_properties(off_axis_filter PROPERTIES PREFIX "" OUTPUT_NAME "off_axis_filter" SUFFIX ".so")

# Set the output directory for the compiled module
set_target_properties(off_axis_filter PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Set needed libraries
target_link_libraries(off_axis_filter PRIVATE -lfftw3 -lfftw3_threads -lm ${OpenCV_LIBS})

# <PhaseUnwrap>:

# Add the pybind11 source files
add_library(phase_unwrap MODULE src/PhaseUnwrap.cpp)

# Set filename
set_target_properties(phase_unwrap PROPERTIES PREFIX "" OUTPUT_NAME "phase_unwrap" SUFFIX ".so")

# Set the output directory for the compiled module
set_target_properties(phase_unwrap PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Set needed libraries
target_link_libraries(phase_unwrap PRIVATE -lfftw3 -lfftw3_threads -lm ${OpenCV_LIBS})