#include "OffAxFilter.h"
#include<stdexcept>
#include<cmath>
#include<iostream>

using namespace std;
namespace py = pybind11;

/**
* @brief Initialize off-axis filter object.
**/
OffAxFilter::OffAxFilter(py::array_t<int, py::array::c_style | py::array::forcecast> n, py::array_t<int, py::array::c_style | py::array::forcecast> roi, 
                        py::array_t<bool, py::array::c_style | py::array::forcecast> mask, py::array_t<complex<double>, py::array::c_style | py::array::forcecast> ref, 
                        py::array_t<double, py::array::c_style | py::array::forcecast> window, unsigned nthreads, unsigned flags) {
    
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
    fftw_free(this->d);
    fftw_destroy_plan(this->fft_forward);
    fftw_destroy_plan(this->fft_backward);
}


/**
 * @brief Allocate resources for Fourier transforms.
**/
void OffAxFilter::alloc() {

    // Initiate multithreading
    if (fftw_init_threads()==0) {
        printf("Error in initiating multi-threading.");
    } else {
        fftw_plan_with_nthreads(this->nthreads);
    }

    this->nthreads = fftw_planner_nthreads();
    this->a = (fftw_complex *) fftw_alloc_complex(this->N);
    this->b = (fftw_complex *) fftw_alloc_complex(this->N);
    this->c = (fftw_complex *) fftw_alloc_complex(roi[2]*roi[3]);
    this->d = (fftw_complex *) fftw_alloc_complex(roi[2]*roi[3]);
    this->fft_forward = fftw_plan_dft(2, this->shape, this->a, this->b, FFTW_FORWARD, this->flags);
    this->fft_backward = fftw_plan_dft_2d(this->roi[2], this->roi[3], this->c, this->d, FFTW_BACKWARD, this->flags);
}


/**
 * @brief Return the total number of elements.
**/
int OffAxFilter::size() {
    return this->N;
}

/**
 * @brief Returns the number of threads in use.
**/
int OffAxFilter::threads_in_use() {
    return this->nthreads;
}


/**
 * @brief Python call function that filters interfence pattern and returns numpy array.
 * @param fringes Numpy ndarray.
**/
py::array_t<complex<double>> OffAxFilter::__call__(py::array_t<complex<double>, py::array::c_style | py::array::forcecast> fringes) {
    complex<double> c;
    bool res = this->filter(fringes);
    size_t shape[2] = {static_cast<size_t>(this->roi[2]), static_cast<size_t>(this->roi[3])};
    py::array_t<complex<double>> field(shape);
    if (res) {
        auto view =  field.mutable_unchecked<2>();
        for (auto i=0; i<field.shape(0); i++) {
            for (auto j=0; j<field.shape(1); j++) {
                c.real(this->d[i*field.shape(0) + j][0]);
                c.imag(this->d[i*field.shape(0) + j][1]);
                view(i, j) = c;
            }
        }
    }
    return field;
}


/**
 * @brief Filter a provided interference pattern.
 * @param fringes Pointer to input vector.
**/
bool OffAxFilter::filter(vector<complex<double>> *fringes) {
    bool res = this->write(fringes);
    if (res) {
        this->forwards();
        this->crop();
        this->backwards();
    }
    return res;
}


/**
 * @brief Filter a provided interference pattern.
 * @param fringes Numpy ndarray.
**/
bool OffAxFilter::filter(py::array_t<complex<double>, py::array::c_style | py::array::forcecast> fringes) {
    bool res = this->write(fringes);
    if (res) {
        this->forwards();
        this->crop();
        this->backwards();
    }
    return res;
}


/**
 * @brief Extracts the phase information from the filtered field.
**/
double* OffAxFilter::extract_phase() {
    complex<double> c;
    double* phase_wrap = new double[this->roi[2]*this->roi[3]];
    for (auto i = 0; i < this->roi[2]*this->roi[3]; i++) {
        c.real(this->d[i][0]);
        c.imag(this->d[i][1]);
        phase_wrap[i] = arg(c);
    }
    return phase_wrap;
}


/**
 * @brief Executes plan for FFT transform.
**/
void OffAxFilter::forwards() {
    fftw_execute(this->fft_forward);
}


/**
 * @brief Executes plan for IFFT transform.
**/
void OffAxFilter::backwards() {
    fftw_execute(this->fft_backward);
}


/**
 * @brief Crop transformed array to the ROI.
**/
void OffAxFilter::crop() {
    int k, l=0;
    for (auto i=this->roi[0]; i<this->roi[0]+this->roi[2]; i++) {
        for (auto j=this->roi[1]; j<roi[1]+this->roi[3]; j++) {
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
        for (auto i=0; i<this->N; i++) {
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
bool OffAxFilter::write(py::array_t<complex<double>, py::array::c_style | py::array::forcecast> arr) {
    // Access the underlying buffer of the NumPy array
    py::buffer_info buf = arr.request();
    
    // Pointer to the data
    complex<double> *ptr = static_cast<complex<double> *>(buf.ptr);
    
    // Length of the array
    unsigned size = buf.size;

    if (size == this->N) {
        for (auto i=0; i<this->N; i++) {
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
        for (auto i = 0; i < this->roi[2]*this->roi[3]; i++) {
            printf("%3d %+9.5f %+9.5f i\n", i, this->c[i][0], this->c[i][1]);
        }
    } else {
        for (auto i = 0; i < this->N; i++) {
            printf("%3d %+9.5f %+9.5f i\n", i, this->a[i][0], this->a[i][1]);
        }
    }
}