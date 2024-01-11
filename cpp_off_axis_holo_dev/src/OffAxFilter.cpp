#include "OffAxFilter.h"
#include<stdexcept>
#include<iostream>
using namespace std;

/**
* @brief Initialize off-axis filter object.
**/
OffAxFilter::OffAxFilter(py::array_t<int> n, py::array_t<int> roi, py::array_t<bool> mask, 
                py::array_t<complex<double>> ref, py::array_t<double> window, unsigned threads, unsigned flags) {
    
    // Access numpy arrays
    py::buffer_info sz_buf = n.request();
    if (sz_buf.size != 2) {
        throw py::value_error("Expect 2D shape.");
    }

    py::buffer_info roi_buf = roi.request();
    if (roi_buf.size != 4) {
        throw py::value_error("Expect (x, y, dx, dy) for ROI.");
    }

    this->shape = static_cast<int *>(sz_buf.ptr);
    this->roi = static_cast<int *>(roi_buf.ptr);

    py::buffer_info mask_buf = mask.request();
    this->mask = static_cast<bool *>(mask_buf.ptr);

    py::buffer_info ref_buf = ref.request();
    this->ref = static_cast<complex<double> *>(ref_buf.ptr);

    py::buffer_info win_buf = ref.request();
    this->window = static_cast<double *>(win_buf.ptr);

    this->N = this->shape[0] * this->shape[1]; 
    this->flags = flags;
    this->nthreads = nthreads;
    this->alloc();
}


/**
 * @brief Free memory for off-axis filter.
**/
OffAxFilter::~OffAxFilter() {
    fftw_free(this->a);
    fftw_free(this->b);
    fftw_free(this->c);
    fftw_destroy_plan(this->plan_forward);
    fftw_destroy_plan(this->plan_backward);
}


/**
 * @brief Allocate resources for Fourier transforms.
**/
void OffAxFilter::alloc() {

    // This multithreading is breaking. Should I go back to DFT implementation that worked for some reason?
    // Initiate multithreading
    //if (fftw_init_threads()==0) {
    //    printf("Error in initiating multi-threading.");
    //} else {
    //    fftw_plan_with_nthreads(this->nthreads);
    //}

    int roi_shape[2] = {this->roi[2], this->roi[3]};

    this->nthreads = fftw_planner_nthreads();
    this->a = (fftw_complex *) fftw_alloc_complex(this->N);
    this->b = (fftw_complex *) fftw_alloc_complex(this->N);
    this->c = (fftw_complex *) fftw_alloc_complex(roi[2]*roi[3]);
    this->plan_forward = fftw_plan_dft(2, this->shape, this->a, this->b, FFTW_FORWARD, this->flags);
    this->plan_backward = fftw_plan_dft(2, roi_shape, this->c, this->c, FFTW_BACKWARD, this->flags);
}


/**
 * @brief Return the total number of elements.
**/
int OffAxFilter::size() {
    return this->N;
}


/**
 * @brief Filter a provided interference pattern.
 * @param fringes Pointer to input vector.
**/
bool OffAxFilter::filter(vector<complex<double>> *fringes) {
    bool res = this->write(fringes);
    if (res) {
        this->forwards();
        this->backwards();
    }
    return res;
}


/**
 * @brief Filter a provided interference pattern.
 * @param fringes Numpy ndarray.
**/
bool OffAxFilter::filter(py::array_t<complex<double>> fringes) {
    bool res = this->write(fringes);
    if (res) {
        this->forwards();
        this->backwards();
    }
    return res;
}


/**
 * @brief Executes plan for FFT transform.
**/
void OffAxFilter::forwards() {
    fftw_execute(this->plan_forward);
    this->crop();
}


/**
 * @brief Executes plan for IFFT transform.
**/
void OffAxFilter::backwards() {
    fftw_execute(this->plan_backward);
    /*
    for (unsigned i=0; i<this->size(); i++) {
        this->a[i][0] /= this->size();
        this->a[i][1] /= this->size();
    } */
}


/**
 * @brief Crop transformed array to the ROI.
**/
void OffAxFilter::crop() {
    int k, l=0;
    for (int i=this->roi[0]; i<this->roi[0]+this->roi[2]; i++) {
        for (int j=this->roi[1]; j<roi[1]+this->roi[3]; j++) {
            k = j*this->shape[1] + i;
            this->c[l][0] = this->b[k][0];
            this->c[l][1] = this->b[k][1];
            l += 1;
        }
    }
}


/**
 * @brief Writes vector to memory.
 * @param vec Pointer to input vector.
**/
bool OffAxFilter::write(vector<complex<double>> *vec) {
    if (vec->size() == this->N) {
        for (unsigned i=0; i<this->N; i++) {
            (*vec)[i] *= this->ref[i] * this->window[i];
            this->a[i][0] = real((*vec)[i]);
            this->a[i][1] = imag((*vec)[i]);
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
bool OffAxFilter::write(py::array_t<complex<double>> arr) {
    // Access the underlying buffer of the NumPy array
    py::buffer_info buf = arr.request();
    
    // Pointer to the data
    complex<double> *ptr = static_cast<complex<double> *>(buf.ptr);
    
    // Length of the array
    unsigned size = buf.size;

    if (size == this->N) {
        for (unsigned i=0; i<this->N; i++) {
            ptr[i] *= this->ref[i] * this->window[i];
            this->a[i][0] = ptr[i].real();
            this->a[i][1] = ptr[i].imag();
        }
        return true;
    } else {
        return false;
    }
}


void OffAxFilter::to_string(bool inverse) {
    if (inverse) {
        for (int i = 0; i < this->N; i++) {
            printf("%3d %+9.5f %+9.5f i\n", i, this->b[i][0], this->b[i][1]);
        }
    } else {
        for (int i = 0; i < this->N; i++) {
            printf("%3d %+9.5f %+9.5f i\n", i, this->a[i][0], this->a[i][1]);
        }
    }
}