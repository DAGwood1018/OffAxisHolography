#ifndef DFT_H
#define DFT_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex.h>
#include <fftw3.h>
#include <vector>

using namespace std;
namespace py = pybind11;

class DFT {

public:
    DFT(unsigned rank, int *n, unsigned threads, unsigned flags, bool ortho);
    DFT(py::array_t<int> n, unsigned threads, unsigned flags, bool ortho);
    ~DFT();
    
    bool forwards(vector<complex<double>> *a);
    bool backwards(vector<complex<double>> *b);
    bool forwards(py::array_t<complex<double>> a);
    bool backwards(py::array_t<complex<double>> b);
    virtual void forwards();
    virtual void backwards();
    int size();
    int threads_in_use();

protected:
    fftw_complex *a; // Data transformed by forward and written to by backward
    fftw_complex *b; // Data transformed by backward and written to by forward

    unsigned rank; // Dimensionality of transform
    unsigned N; // Total number of elements of the transform
    int* shape; // Shape of the transform
    bool ortho; // Whether to normalize the transforms to be orthogonal
    unsigned nthreads; // Number of threads to use for multithreading 
    unsigned flags; // FFTW planner flags
    fftw_plan plan_forward; // FFTW plan for fft
	fftw_plan plan_backward; // FFTW plan for ifft

    virtual void alloc();
    virtual bool write(vector<complex<double>> *vec, bool inverse);
    virtual bool write(py::array_t<complex<double>> arr, bool inverse);
};

#endif