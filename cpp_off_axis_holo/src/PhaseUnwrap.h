#ifndef UNWRAP_H
#define UNWRAP_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <complex.h>
#include <fftw3.h>
#include <vector>

//using namespace Eigen;

// Helper functions for phase unwrapping
Eigen::MatrixXd wrap_to_pi(const Eigen::MatrixXd& angles);

using namespace std;
namespace py = pybind11;

class PhaseUnwrap {

public:
    PhaseUnwrap(int M, int N, unsigned nthreads, unsigned flags);
    PhaseUnwrap(py::array_t<int, py::array::c_style | py::array::forcecast> n, unsigned nthreads, unsigned flags);
    ~PhaseUnwrap();

    py::array_t<double> __call__(py::array_t<double, py::array::c_style | py::array::forcecast> phase_wrap);
    Eigen::MatrixXd operator()(Eigen::MatrixXd& phase_wrap);
    void unwrap();
    int size();
    int threads_in_use();
    void show_input();
    void show_output();

protected:

    // Properties for planning Fourier Transforms
    unsigned nthreads; // Number of threads to use for multithreading 
    unsigned flags; // FFTW planner flags
    double *a; // Array for storing and unwrapping phase information.
    double *b; // Array that stores the descrete cosine transform of the phase.
    fftw_plan dct_forward; // FFTW plan for dct
    fftw_plan dct_backward; // FFTW plan for idct

    int M; // Number rows
    int N; // Number of columns.
    Eigen::MatrixXd* phase_wrap;
    Eigen::MatrixXd* phase_unwrap;

    void alloc();
    void show(Eigen::MatrixXd* phase);
    void forwards(double *a);
    void backwards(double *a);
    void solve(Eigen::MatrixXd& in, Eigen::MatrixXd& out);
    bool write(Eigen::MatrixXd& mat);
    bool write(py::array_t<double, py::array::c_style | py::array::forcecast> arr);
};

#endif