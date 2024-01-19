cmake_minimum_required(VERSION 3.12)
project(holo_lib)

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

# Add the pybind11 source files
add_library(holo_lib MODULE src/OffAxFilter.cpp src/PhaseUnwrap.cpp src/bind.cpp)

# Set filename
set_target_properties(holo_lib PROPERTIES PREFIX "" OUTPUT_NAME "holo_lib" SUFFIX ".so")

# Set the output directory for the compiled module
set_target_properties(holo_lib PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Set needed libraries
target_link_libraries(holo_lib PRIVATE -lfftw3 -lfftw3_threads -lm ${OpenCV_LIBS})