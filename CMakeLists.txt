cmake_minimum_required(VERSION 3.10)
project(off_axis_holo)

# Set Flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")


# <Setup Pybind11>:

include(FetchContent)

# Set the URL of the Pybind11 repository
set(PYBIND11_URL https://github.com/pybind/pybind11.git)

# Fetch the content from the Pybind11 repository
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY ${PYBIND11_URL}
)

# Fetch the content
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
endif()

# Specify the path to the pybind11 headers
include_directories(${pybind11_SOURCE_DIR}/include)


# <Find and Include Other Dependencies>:

# Find Python Dev
find_package(Python COMPONENTS Development)

# Find Eigen
find_package(Eigen3 REQUIRED)
add_definitions(-DEIGEN_NO_DEBUG)

# Find FFTW3
find_package(FFTW3 REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Specify the path to the Python include directory
include_directories(${Python_INCLUDE_DIRS})

# Include Eigen headers
include_directories(${EIGEN3_INCLUDE_DIRS})

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Include FFTW3 headers
include_directories(${FFTW3_INCLUDE_DIRS})

# <OffAxisModule>:

#find_package(PythonExtensions REQUIRED)

# Add the pybind11 source files
add_library(off_axis_module MODULE cpp_off_axis_holo/src/OffAxisFilter.cpp  cpp_off_axis_holo/src/PhaseUnwrap.cpp  cpp_off_axis_holo/src/ExtractPhase.cpp)

# Set filename
set_target_properties(off_axis_module PROPERTIES PREFIX "" OUTPUT_NAME "off_axis_module" SUFFIX ".so")

# Set the output directory for the compiled module
set_target_properties(off_axis_module PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/off_axis_holo/src)

# Set needed libraries
target_link_libraries(off_axis_module PRIVATE ${FFTW3_LIBS} ${OpenCV_LIBS} ${EIGEN3_LIBS})

#python_extension_module(off_axis_module)
#install(TARGETS off_axis_module LIBRARY DESTINATION  off_axis_holo)