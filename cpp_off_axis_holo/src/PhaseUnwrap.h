#ifndef UNWRAP_H
#define UNWRAP_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex.h>
#include <fftw3.h>
#include <vector>

//using namespace Eigen;

// Helper functions for phase unwrapping
double wrap_to_pi(double angle);
int diff(int* k1, int* k2, size_t N);
void offset_phases(double* phase_wrap, double* phase_unwrap, int* k, size_t N);

using namespace std;
namespace py = pybind11;

class PhaseUnwrap {

public:
    PhaseUnwrap(py::array_t<int, py::array::c_style | py::array::forcecast> n, unsigned nthreads, unsigned flags);
    ~PhaseUnwrap();

    vector<double> unwrap(vector<double> *phase_wrap);
    py::array_t<double> unwrap(py::array_t<double, py::array::c_style | py::array::forcecast> phase_wrap);
    void execute();
    void forwards();
    void backwards();
    int size();
    int threads_in_use();

    void to_string();

protected:

    // Properties for planning Fourier Transforms
    unsigned nthreads; // Number of threads to use for multithreading 
    unsigned flags; // FFTW planner flags
    double *a; // Array for storing and unwrapping phase information.
    double *b; // Array that stores the descrete cosine transform of the phase.
    fftw_plan dct_forward; // FFTW plan for dct
    fftw_plan dct_backward; // FFTW plan for idct

    int N; // Number of elements.
    int* shape; // Shape of input.
    double* phase_wrap;
    double* phase_unwrap;
    double* residual_wrap;
    double* residual_unwrap;
    double* phi;
    int* k1;
    int* k2;

    void alloc();
    void solve(double* in, double* out);
    bool write(vector<double> *vec);
    bool write(py::array_t<double, py::array::c_style | py::array::forcecast> arr);
};

#endif