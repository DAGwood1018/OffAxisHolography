#include "PhaseUnwrap.h"
#include<stdexcept>
#include<cmath>
#include<numeric>
#include<iostream>

using namespace std;

double wrap_to_pi(double angle) {
    return fmod(angle + M_PI, 2 * M_PI) - M_PI;
}

void offset_phases(double* phase_wrap, double* phase_unwrap, double* phi, int* k, size_t N) {
    /*
    phi1 += np.mean(phase_wrap) - np.mean(phi1)  # adjust piston
    K1 = np.round((phi1 - phase_wrap) / (2 * np.pi))  # calculate integer K
    phase_unwrap = phase_wrap + 2 * K1 * np.pi
    */

    double mean_phase_wrap = accumulate(phase_wrap, phase_wrap+N, 0) / N;
    double mean_phi = accumulate(phi, phi+N, 0) / N;
    
    for (int i = 0; i < N; i++) {
        phi[i] += (mean_phase_wrap - mean_phi);
    }

    for (int i = 0; i < N; i++) {
        k[i] = round(phi[i] - phase_wrap[i]) / (2 * M_PI); // calculate integer K
        phase_unwrap[i] = phase_wrap[i] + 2 * k[i] * M_PI;
    } 
}

int diff(int* k1, int* k2, size_t N) {
    int diff = 0;
    for (int i = 0; i < N; i++) {
        diff += abs(k2[i]-k1[i]);
    }
    return diff;
}


/**
* @brief Initialize phase unwrapping object.
**/
PhaseUnwrap::PhaseUnwrap(py::array_t<int, py::array::c_style | py::array::forcecast> n, unsigned nthreads, unsigned flags) {

    py::buffer_info sz_buf = n.request();
    if (sz_buf.size != 2) {
        throw py::value_error("Expect 2D shape.");
    }

    this->shape = static_cast<int *>(sz_buf.ptr);
    this->N = this->shape[0] * this->shape[1]; 
    this->flags = flags;
    this->nthreads = nthreads;
    this->alloc();
}


/**
 * @brief Free memory from phase unwrapping.
**/
PhaseUnwrap::~PhaseUnwrap() {
    fftw_free(this->a);
    fftw_free(this->b);
    fftw_destroy_plan(this->dct_forward);
    fftw_destroy_plan(this->dct_backward);
    delete[] this->phase_unwrap;
    delete[] this->phase_wrap;
    delete[] this->residual_wrap;
    delete[] this->residual_unwrap;
    delete[] this->phi;
    delete[] this->k1;
    delete[] this->k2;
}


/**
 * @brief Allocate resources for Phase Unwrapping.
**/
void PhaseUnwrap::alloc() {

    // Initiate multithreading
    if (fftw_init_threads()==0) {
        printf("Error in initiating multi-threading.");
    } else {
        fftw_plan_with_nthreads(this->nthreads);
    }

    this->nthreads = fftw_planner_nthreads();
    this->a = (double *) fftw_alloc_real(this->N);
    this->b = (double *) fftw_alloc_real(this->N);
    this->dct_forward = fftw_plan_r2r_2d(this->shape[0], this->shape[1], this->a, this->b, FFTW_REDFT10, FFTW_REDFT10, this->flags);
    this->dct_backward = fftw_plan_r2r_2d(this->shape[0], this->shape[1], this->b, this->a, FFTW_REDFT01, FFTW_REDFT01, this->flags);

    this->phase_wrap = new double[this->N];
    this->phase_unwrap = new double[this->N];
    this->residual_wrap = new double[this->N];
    this->residual_unwrap = new double[this->N];
    this->phi = new double[this->N];
    this->k1 = new int[this->N];
    this->k2 = new int[this->N];
}


/**
 * @brief Return the total number of elements.
**/
int PhaseUnwrap::size() {
    return this->N;
}

/**
 * @brief Returns the number of threads in use.
**/
int PhaseUnwrap::threads_in_use() {
    return this->nthreads;
}


/**
 * @brief Unwrap a given wrapped phase image.
 * @param fringes Pointer to input vector.
**/
vector<double> PhaseUnwrap::unwrap(vector<double> *phase_wrap) {
    bool res = this->write(phase_wrap);
    vector<double> phase_unwrap(this->N);
    if (res) {
        this->execute();
        for (auto i=0; i<this->N; i++) {
            phase_unwrap.push_back(this->phase_unwrap[i]);
        }
    }
    return phase_unwrap;
}


/**
 * @brief Unwrap a given wrapped phase image.
 * @param phase Numpy ndarray.
**/
py::array_t<double> PhaseUnwrap::unwrap(py::array_t<double, py::array::c_style | py::array::forcecast> phase_wrap) {
    bool res = this->write(phase_wrap);
    size_t shape[2] = {static_cast<size_t>(this->shape[0]), static_cast<size_t>(this->shape[1])};
    py::array_t<double> phase_unwrap(shape);
    if (res) {
        this->execute();
        auto view =  phase_unwrap.mutable_unchecked<2>();
        for (auto i = 0; i<phase_unwrap.shape(0); i++) {
            for (auto j = 0; j<phase_unwrap.shape(1); j++) {
                view(i, j) = this->phase_unwrap[i*phase_unwrap.shape(0) + j];
            }
        }
    }
    return phase_unwrap;
}


void PhaseUnwrap::execute() {

    // Extract phase from complex fields.
    this->solve(this->phase_wrap, this->phi);
    offset_phases(this->phase_wrap, this->phase_unwrap, this->phi, this->k1, this->N);
    for (auto i = 0; i < this->N; i++) {
        this->residual_wrap[i] = wrap_to_pi(this->phase_unwrap[i] - this->phi[i]);
    }
    this->solve(this->residual_wrap, this->residual_unwrap);
    for (auto i = 0; i < this->N; i++) {
        this->phi[i] += this->residual_unwrap[i];
    }
    offset_phases(this->phase_wrap, this->phase_unwrap, this->phi, this->k2, this->N);
    for (auto i = 0; i < this->N; i++) {
        residual_wrap[i] = wrap_to_pi(this->phase_unwrap[i] - this->phi[i]);
    }

    while (diff(this->k1, this->k2, this->N) > 0) {
        this->solve(this->residual_wrap, this->residual_unwrap);
        for (auto i = 0; i < this->N; i++) {
            this->k1[i] = this->k2[i];
            this->phi[i] += this->residual_unwrap[i];
        }
        offset_phases(this->phase_wrap, this->phase_unwrap, this->phi, this->k2, this->N);
        for (auto i = 0; i < this->N; i++) {
            this->residual_wrap[i] = wrap_to_pi(this->phase_unwrap[i] - this->phi[i]);
        }
    }
}


void PhaseUnwrap::solve(double* in, double* out) {
    // Calculate edx
    for (auto i = 0; i < this->shape[0]; ++i) {
        this->a[i * (this->shape[1] + 2)] = 0; // First column
        for (auto j = 1; j <= this->shape[1]; ++j) {
            this->a[i * (this->shape[1] + 2) + j] = wrap_to_pi(in[i * this->shape[1] + (j - 1)]) -
                                       wrap_to_pi(in[i * this->shape[1] + (j - 2)]);
        }
        this->a[i * (this->shape[1] + 2) + this->shape[1] + 1] = 0; // Last column
    }

    // Calculate edy
    for (auto j = 0; j < this->shape[1] + 2; ++j) {
        this->a[j] = 0; // First row
        for (auto i = 1; i <= this->shape[1]; ++i) {
            this->a[i * (this->shape[1] + 2) + j] = wrap_to_pi(in[(i - 1) * this->shape[1] + j]) -
                                       wrap_to_pi(in[(i - 2) * this->shape[1] + j]);
        }
        this->a[this->shape[0] * (this->shape[1] + 2) + j] = 0; // Last row
    }

    // Calculate Laplacian
    for (auto i = 1; i <= this->shape[0]; ++i) {
        for (auto j = 1; j <= this->shape[1]; ++j) {
            this->a[i * (this->shape[1] + 2) + j] = this->a[i * (this->shape[1] + 2) + (j + 1)] - this->a[i * (this->shape[1] + 2) + j] +
                                       this->a[(i + 1) * (this->shape[1] + 2) + j] - this->a[i * (this->shape[1] + 2) + j];
        }
    }

    // Execute forward DCT transform
    this->forwards();

    double denom;
    for (auto i = 0; i < this->shape[0]; ++i) {
        for (auto j = 0; j < this->shape[1]; ++j) {
            if (i==0 && j==0) {
                this->b[0] = 0;
            } else {
                denom = 2 / (cos(M_PI * i / this->shape[0]) + cos(M_PI * j / this->shape[1]) - 2);
                this->b[i * this->shape[0] + j] /= denom;
            }
        }
    } 

    // Execute backward DCT transform
    this->backwards();
    
    // Write to output array
    for (auto i = 0; i < this->N; i++) {
        out[i] = this->a[i];
    }
}


/**
 * @brief Executes plan for DCT transform.
**/
void PhaseUnwrap::forwards() {
    fftw_execute(this->dct_forward);
}


/**
 * @brief Executes plan for IDCT transform.
**/
void PhaseUnwrap::backwards() {
    fftw_execute(this->dct_backward);
}


/**
 * @brief Writes vector to memory.
 * @param vec Pointer to input vector.
**/
bool PhaseUnwrap::write(vector<double> *vec) {
    if (vec->size() == this->N) {
        for (auto i = 0; i<this->N; i++) {
            this->phase_wrap[i] = (*vec)[i];
        }
        return true;
    } else {
        return false;
    }
}


/**
 * @brief Writes numpy array to memory.
 * @param arr Numpy ndarray.
**/
bool PhaseUnwrap::write(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    // Access the underlying buffer of the NumPy array
    py::buffer_info buf = arr.request();
    
    // Pointer to the data
    double *ptr = static_cast<double *>(buf.ptr);
    
    // Length of the array
    unsigned size = buf.size;

    if (size == this->N) {
        for (auto i = 0; i<this->N; i++) {
            this->phase_wrap[i] = ptr[i];
        }
        return true;
    } else {
        return false;
    }
}


void PhaseUnwrap::to_string() {
    for (auto i = 0; i<5; i++) {
        for (auto j = 0; j<5; j++) {
            printf("%3d %+9.5f\n", i, this->phase_unwrap[i*this->shape[0] + j]);
        }
    }
}