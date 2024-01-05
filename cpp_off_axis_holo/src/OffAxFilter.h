#ifndef OFFAXFILTER_H
#define OFFAXFILTER_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex.h>
#include <fftw3.h>
#include <vector>
#include "DFT.h"

using namespace std;
namespace py = pybind11;

class OffAxFilter : public DFT {

public:
    OffAxFilter(py::array_t<int> dims, py::array_t<int> cropped_dims, py::array_t<int> mask, 
                py::array_t<double> window, unsigned threads, unsigned flags);
    ~OFFAxFilter();

protected:
    int* roi; // Shape of the transform after cropping.
    double *window; // Window to apply before transforming.
    int *mask; // Mask to apply in the Fourier plane.

    bool set_mask(py::array_t<int> mask);
    bool set_window(py::array_t<double> window);
};

#endif