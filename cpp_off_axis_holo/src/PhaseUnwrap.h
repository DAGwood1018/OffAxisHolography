#ifndef UNWRAP_H
#define UNWRAP_H

#include <Eigen/Dense>
#include <complex.h>
#include <fftw3.h>
#include <vector>

//using namespace Eigen;

// Helper functions for phase unwrapping
Eigen::MatrixXd wrap_to_pi(const Eigen::MatrixXd& angles);

using namespace std;

class PhaseUnwrap {

public:
    PhaseUnwrap(int M, int N, unsigned nthreads, unsigned flags);
    ~PhaseUnwrap();

    Eigen::MatrixXd operator()(Eigen::MatrixXd& phase_wrap, bool zero);
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
};

#endif