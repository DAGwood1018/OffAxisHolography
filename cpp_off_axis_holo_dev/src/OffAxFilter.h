#ifndef OFFAXFILTER_H
#define OFFAXFILTER_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex.h>
#include <fftw3.h>
#include <vector>

using namespace std;
namespace py = pybind11;

class OffAxFilter {

public:
    OffAxFilter(py::array_t<int> n, py::array_t<int> roi, py::array_t<bool> mask,
                py::array_t<complex<double>> ref, py::array_t<double> window, unsigned threads, unsigned flags);
    ~OffAxFilter();

    bool filter(vector<complex<double>> *fringes);
    bool filter(py::array_t<complex<double>> fringes);
    void forwards();
    void backwards();
    int size();

    void to_string(bool inverse);

protected:

    // Properties for planning Fourier Transforms
    unsigned nthreads; // Number of threads to use for multithreading 
    unsigned flags; // FFTW planner flags
    fftw_complex *a; // Data transformed by forward and written to by backward
    fftw_complex *b; // Data transformed by backward and written to by forward
    fftw_complex *c; // Array for storing cropped FFT.
    fftw_plan plan_forward; // FFTW plan for fft
	fftw_plan plan_backward; // FFTW plan for ifft

    unsigned N; // Number of elements.
    int* shape; // Shape of input.
    int* roi; // ROI to crop input to.
    double *window; // Window to apply before transforming.
    complex<double> *ref; // Reference beam for tilt correction.
    bool *mask; // Mask to apply in the Fourier plane.

    void alloc();
    void crop(); 
    bool write(vector<complex<double>> *vec);
    bool write(py::array_t<complex<double>> arr);
};

#endif