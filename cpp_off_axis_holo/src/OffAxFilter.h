#ifndef OFFAXFILTER_H
#define OFFAXFILTER_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <complex.h>
#include <fftw3.h>
#include <vector>

using namespace std;
namespace py = pybind11;

class OffAxFilter {

public:
    OffAxFilter(py::array_t<int, py::array::c_style | py::array::forcecast> n, py::array_t<int, py::array::c_style | py::array::forcecast> roi, 
                py::array_t<bool, py::array::c_style | py::array::forcecast> mask, py::array_t<complex<double>, py::array::c_style | py::array::forcecast> ref, 
                unsigned nthreads, unsigned flags);
    ~OffAxFilter();

    py::array_t<double> __call__(py::array_t<complex<double>, py::array::c_style | py::array::forcecast> fringes);
    bool filter(py::array_t<complex<double>, py::array::c_style | py::array::forcecast> fringes);
    bool filter(vector<complex<double>> *fringes);
    vector<double> extract_phase();
    void forwards();
    void backwards();
    int size();
    int threads_in_use();
    void show_input();

    py::array_t<bool> get_mask();
    py::array_t<complex<double>> get_ref();
    void crop(); 
    bool write(py::array_t<complex<double>, py::array::c_style | py::array::forcecast> arr);
    void peak_a();
    void peak_b();
    void peak_c();
    void peak_d();

protected:

    // Properties for planning Fourier Transforms
    unsigned nthreads; // Number of threads to use for multithreading 
    unsigned flags; // FFTW planner flags
    fftw_complex *a; // Data to be transformed by fft forward
    fftw_complex *b; // Output of fft forward
    fftw_complex *c; // Cropped output of fft forward to be transformed by fft backward
    fftw_complex *d; // Output of fft backward.
    fftw_plan fft_forward; // FFTW plan for fft
	fftw_plan fft_backward; // FFTW plan for ifft


    int N; // Number of elements.
    int* shape; // Shape of input.
    int* roi; // ROI to crop input to.
    complex<double> *ref; // Reference beam for tilt correction.
    bool *mask; // Mask to apply in the Fourier plane.

    void alloc();
    //void crop(); 
    bool write(vector<complex<double>> *vec);
    //bool write(py::array_t<complex<double>, py::array::c_style | py::array::forcecast> arr);
};

#endif