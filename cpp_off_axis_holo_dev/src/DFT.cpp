#include "DFT.h"

using namespace std;
namespace py = pybind11;

/**
* @brief Allocate memory and plan DFT.
* @param rank Integer giving rank of transform.
* @param n Pointer to array specifying the shape of the transform.
* @param nthreads Integer number of threads to use.
* @param flags FFTW planner flags to use. Combine using bitwise OR. 
* @param ortho Whehter the transformation should be orthogonal.
**/
DFT::DFT(unsigned rank, int *n, unsigned nthreads, unsigned flags, bool ortho) {
    // Find size of transform
    int N = 1;
    for (unsigned i = 0; i < rank; i++) {
        N *= n[i];
    }

    this->rank = rank;
    this->N = N;
    this->shape = n;
    this->ortho = ortho;
    this->flags = flags;
    this->nthreads = nthreads;
    this->alloc();
} 

/**
* @brief Allocate memory and plan DFT.
* @param n Numpy array giving shape.
* @param nthreads Integer number of threads to use.
* @param flags FFTW planner flags to use. Combine using bitwise OR. 
* @param ortho Whehter the transformation should be orthogonal.
**/
DFT::DFT(py::array_t<int> n, unsigned nthreads, unsigned flags, bool ortho) {
    // Access numpy array
    py::buffer_info buf = n.request();
    unsigned rank = buf.size;
    int *shape = static_cast<int *>(buf.ptr);
    
    // Find size of transform
    int N = 1;
    for (unsigned i = 0; i < rank; i++) {
        N *= shape[i];
    }

    this->rank = rank;
    this->N = N;
    this->shape = shape;
    this->ortho = ortho;
    this->flags = flags;
    this->nthreads = nthreads;
    this->alloc();
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

/**
 * @brief Allocate resources to FFT class properties.
**/
void DFT::alloc() {
    // Initiate multithreading
    if (fftw_init_threads()==0) {
        printf("Error in initiating multi-threading.");
    } else {
        fftw_plan_with_nthreads(this->nthreads);
    }

    this->nthreads = fftw_planner_nthreads();
    this->a = (fftw_complex *) fftw_alloc_complex(this->N);
    this->b = (fftw_complex *) fftw_alloc_complex(this->N);
    this->plan_forward = fftw_plan_dft(this->rank, this->shape, this->a, this->b, FFTW_FORWARD, this->flags);
    this->plan_backward = fftw_plan_dft(this->rank, this->shape, this->b, this->a, FFTW_BACKWARD, this->flags);
}

/**
 * @brief Return the total number of elements.
**/
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
    if (this->ortho) {
       for (unsigned i=0; i<this->N; i++) {
            this->a[i][0] /= this->N;
            this->a[i][1] /= this->N;
        }
    }
}

/**
 * @brief Executes plan for FFT transform.
 * @param a Pointer to input vector.
**/
bool DFT::forwards(vector<complex<double>> *a) {
    bool res = this->write(a, false);
    if (res) this->forwards();
    return res;
}

/**
 * @brief Executes plan for IFFT transform.
 * @param b Pointer to input vector.
**/
bool DFT::backwards(vector<complex<double>> *b) {
    bool res = this->write(b, true);
    if (res) this->backwards();
    return res;
}

/**
 * @brief Executes plan for FFT transform.
 * @param a Numpy ndarray.
**/
bool DFT::forwards(py::array_t<complex<double>> a) {
    bool res = this->write(a, false);
    if (res) this->forwards();
    return res;
}

/**
 * @brief Executes plan for IFFT transform.
 * @param b Numpy ndarray.
**/
bool DFT::backwards(py::array_t<complex<double>> b) {
    bool res = this->write(b, true);
    if (res) this->backwards();
    return res;
}

/**
 * @brief Writes vector to memory.
 * @param vec Pointer to input vector.
 * @param inverse Write to the inverse transform array b if true else write to array a.
**/
bool DFT::write(vector<complex<double>> *vec, bool inverse) {
    if (vec->size() == this->N) {
        for (unsigned i=0; i<this->N; i++) {
            if (inverse) {
                this->b[i][0] = real((*vec)[i]);
                this->b[i][1] = imag((*vec)[i]);
            } else {
                this->a[i][0] = real((*vec)[i]);
                this->a[i][1] = imag((*vec)[i]);
            }
        }
        return true;
    } else {
        return false;
    }
}

/**
 * @brief Writes numpy array to memory.
 * @param arr Numpy ndarray.
 * @param inverse Write to the inverse transform array b if true else write to array a.
**/
bool DFT::write(py::array_t<complex<double>> arr, bool inverse) {
    // Access the underlying buffer of the NumPy array
    py::buffer_info buf = arr.request();
    
    // Pointer to the data
    complex<double> *ptr = static_cast<complex<double> *>(buf.ptr);
    
    // Length of the array
    unsigned size = buf.size;

    if (size == this->N) {
        for (unsigned i=0; i<this->N; i++) {
            if (inverse) {
                this->b[i][0] = ptr[i].real();
                this->b[i][1] = ptr[i].imag();
            } else {
                this->a[i][0] = ptr[i].real();
                this->a[i][1] = ptr[i].imag();
            }
        }
        return true;
    } else {
        return false;
    }
}
