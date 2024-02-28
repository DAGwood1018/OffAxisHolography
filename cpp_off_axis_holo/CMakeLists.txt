cmake_minimum_required(VERSION 3.12)
project(holography_bindings)

# Find Python Dev
find_package(Python COMPONENTS Development)

# Find Eigen
find_package(Eigen3 REQUIRED)
add_definitions(-DEIGEN_NO_DEBUG)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Set the path to the pybind11 source directory
set(PYBIND11_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pybind11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Specify the path to the Python include directory
include_directories(${Python_INCLUDE_DIRS})

# Specify the path to the pybind11 headers
include_directories(${PYBIND11_DIR}/include)

# Include Eigen headers
include_directories(${EIGEN3_INCLUDE_DIRS})

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# <OffAxisFilter>:

# Add the pybind11 source files
#add_library(off_axis_filter MODULE src/OffAxisFilter.cpp)

# Set filename
#set_target_properties(off_axis_filter PROPERTIES PREFIX "" OUTPUT_NAME "off_axis_filter" SUFFIX ".so")

# Set the output directory for the compiled module
#set_target_properties(off_axis_filter PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Set needed libraries
#target_link_libraries(off_axis_filter PRIVATE -lfftw3 -lfftw3_threads -lm ${OpenCV_LIBS})

# <PhaseUnwrap>:

# Add the pybind11 source files
#add_library(phase_unwrap MODULE src/PhaseUnwrap.cpp)

# Set filename
#set_target_properties(phase_unwrap PROPERTIES PREFIX "" OUTPUT_NAME "phase_unwrap" SUFFIX ".so")

# Set the output directory for the compiled module
#set_target_properties(phase_unwrap PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Set needed libraries
#target_link_libraries(phase_unwrap PRIVATE -lfftw3 -lfftw3_threads -lm ${OpenCV_LIBS} ${EIGEN3_LIBS})

# <OffAxisModule>:

# Add the pybind11 source files
add_library(off_axis_module MODULE src/OffAxisFilter.cpp src/PhaseUnwrap.cpp src/OffAxisModule.cpp)

# Set filename
set_target_properties(off_axis_module PROPERTIES PREFIX "" OUTPUT_NAME "off_axis_module" SUFFIX ".so")

# Set the output directory for the compiled module
set_target_properties(off_axis_module PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Set needed libraries
target_link_libraries(off_axis_module PRIVATE -lfftw3 -lfftw3_threads -lm ${OpenCV_LIBS} ${EIGEN3_LIBS})

# <Timing function>:

# Add the pybind11 source files
#add_library(timing_module MODULE src/time_execution.cpp src/OffAxisFilter.cpp src/PhaseUnwrap.cpp src/OffAxisModule.cpp)

# Set filename
#set_target_properties(timing_module PROPERTIES PREFIX "" OUTPUT_NAME "timing_module" SUFFIX ".so")

# Set the output directory for the compiled module
#set_target_properties(timing_module PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Set needed libraries
#target_link_libraries(timing_module PRIVATE -lfftw3 -lfftw3_threads -lm ${OpenCV_LIBS} ${EIGEN3_LIBS})

# Add executable
#add_executable(main src/main.cpp)

# Link Eigen library
#target_link_libraries(main ${EIGEN3_LIBS})