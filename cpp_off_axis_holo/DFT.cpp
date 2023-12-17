#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <complex>
#include <fftw3.h>
//#include <pybind11/pybind11.h>
#include "DFT.h"
using namespace std;
//namespace py = pybind11;


/**
* @brief Allocate memory and plan DFT.
* @param rank Integer giving rank of transform.
* @param n Pointer to array specifying the shape of the transform.
* @param nthreads Integer number of threads to use.
* @param flags FFTW planner flags to use. Combine using bitwise OR. 
**/
DFT::DFT(unsigned rank, int *n, unsigned nthreads, unsigned flags) {
    // Find size of transform
    int N = 1;
    for (unsigned i = 0; i < rank; i++) {
        N *= n[i];
    }
    
    // Initiate multithreading
    if (fftw_init_threads()==0) {
        printf("Error in initiating multi-threading.");
    } else {
        fftw_plan_with_nthreads(nthreads);
    }

    this->rank = rank;
    this->N = N;
    this->shape = n;
    this->flags = flags;
    this->nthreads = fftw_planner_nthreads();
    this->a = (fftw_complex *) fftw_alloc_complex(N);
    this->b = (fftw_complex *) fftw_alloc_complex(N);
    this->plan_forward = fftw_plan_dft(rank, n, this->a, this->b, FFTW_FORWARD, flags);
    this->plan_backward = fftw_plan_dft(rank, n, this->b, this->a, FFTW_BACKWARD, flags);
}

/**
 * @brief Free memory for a DFT.
**/
DFT::~DFT() {
    fftw_free(this->a);
    fftw_free(this->b);
    fftw_destroy_plan(this->plan_forward);
    fftw_destroy_plan(this->plan_backward);
}

int DFT::size() {
    return this->N;
}

/**
 * @brief Executes plan for FFT transform.
**/
void DFT::forwards() {
    fftw_execute(this->plan_forward);
}

/**
 * @brief Executes plan for IFFT transform.
**/
void DFT::backwards() {
    fftw_execute(this->plan_backward);
}

/**
 * @brief Executes plan for FFT transform.
**/
bool DFT::forwards(vector<complex<double>> *a) {
    bool res = this->write_array(a, false);
    if (res) this->forwards();
    return res;
}

/**
 * @brief Executes plan for IFFT transform.
**/
bool DFT::backwards(vector<complex<double>> *b) {
    bool res = this->write_array(b, true);
    if (res) this->backwards();
    return res;
}


/**
 * @brief Writes array to memory.
 * @param arr Pointer to input data.
 * @param inverse Write to the inverse transform array b if true else write to array a.
**/
bool DFT::write_array(vector<complex<double>> *arr, bool inverse) {
    if (arr->size() == this->N) {
        for (long unsigned i=0; i<this->N; i++) {
            if (inverse) {
                this->b[i][0] = real((*arr)[i]);
                this->b[i][1] = imag((*arr)[i]);
            } else {
                this->a[i][0] = real((*arr)[i]);
                this->a[i][1] = imag((*arr)[i]);
            }
        }
        return true;
    } else {
        return false;
    }
}

/*
PYBIND11_MODULE(dft_module, m) {
    py::class_<DFT>(m, "DFT")
        .def(py::init<unsigned, int*, unsigned, unsigned>())
        .def("forwards", &DFT::forwards)
        .def("backwards", &DFT::backwards)
        .def("forwards", &DFT::forwards)
        .def("backwards", &DFT::backwards)
        .def("size", &DFT::size);
}
*/