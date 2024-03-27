#include "OffAxisFilter.h"
#include <opencv2/opencv.hpp>

using namespace std;
namespace py = pybind11;


/**
* @brief Initialize off-axis filter object.
**/
OffAxisFilter::OffAxisFilter(py::array_t<int, py::array::c_style | py::array::forcecast> n, py::array_t<int, py::array::c_style | py::array::forcecast> roi, 
                        py::array_t<double, py::array::c_style | py::array::forcecast> mask, py::array_t<complex<double>, py::array::c_style | py::array::forcecast> ref,
                        unsigned nthreads, unsigned flags) {

    // Access numpy array
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
    this->mask = static_cast<double *>(mask_buf.ptr);

    py::buffer_info ref_buf = ref.request();
    this->ref = static_cast<complex<double> *>(ref_buf.ptr);

    this->rank = 2;
    this->N = shape[0] * shape[1];
    this->flags = flags;
    this->nthreads = nthreads;
    this->alloc();
}

/**
 * @brief Free memory for off-axis filter.
**/
OffAxisFilter::~OffAxisFilter() {
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
void OffAxisFilter::alloc() {
    // Initiate multithreading
    if (fftw_init_threads()==0) {
        printf("Error in initiating multi-threading.");
    } else {
        fftw_plan_with_nthreads(this->nthreads);
    }

    int roi_shape[2] = {this->roi[2], this->roi[3]};
    this->nthreads = fftw_planner_nthreads();
    this->a = (fftw_complex *) fftw_alloc_complex(this->N);
    this->b = (fftw_complex *) fftw_alloc_complex(this->N);
    this->c = (fftw_complex *) fftw_alloc_complex(this->roi[2]*this->roi[3]);
    this->d = (fftw_complex *) fftw_alloc_complex(this->roi[2]*this->roi[3]);
    this->fft_forward = fftw_plan_dft(this->rank, this->shape, this->a, this->b, FFTW_FORWARD, this->flags);
    this->fft_backward = fftw_plan_dft(this->rank, roi_shape, this->c, this->d, FFTW_BACKWARD, this->flags);
}


/**
 * @brief Call function that filters interfence pattern and returns numpy array.
 * @param fringes Numpy ndarray.
**/
Eigen::MatrixXd OffAxisFilter::operator()(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> fringes) {
    bool res = this->write(fringes);
    Eigen::MatrixXd filtered = Eigen::MatrixXd::Zero(this->roi[2], this->roi[3]);

    complex<double> c;
    if (res) {
        this->filter();
        for (auto i = 0; i < this->roi[2]; i++) {
            for (auto j = 0; j < this->roi[3]; j++) {
                c.real(this->d[i*this->roi[2]+j][0]);
                c.imag(this->d[i*this->roi[2]+j][1]);
                filtered(i, j) = arg(c);
            }
        }
    } 

    return filtered;
}


/**
 * @brief Return the total number of elements.
**/
int OffAxisFilter::size() {
    return this->N;
}


/**
 * @brief Returns the number of threads in use.
**/
int OffAxisFilter::threads_in_use() {
    return this->nthreads;
}


/**
 * @brief Filter a provided interference pattern.
**/
void OffAxisFilter::filter() {
    this->forwards();
    this->crop();
    this->backwards();
}


/**
 * @brief Executes plan for FFT transform.
**/
void OffAxisFilter::forwards() {
    fftw_execute(this->fft_forward);
    for (auto i=0; i<this->N; i++) {
        this->b[i][0] *= this->mask[i];
        this->b[i][1] *= this->mask[i];
    }
}


/**
 * @brief Executes plan for IFFT transform.
**/
void OffAxisFilter::backwards() {
    fftw_execute(this->fft_backward);
    for (auto i=0; i<this->roi[2]*this->roi[3]; i++) {
        this->d[i][0] /= this->N;
        this->d[i][1] /= this->N;
    }
}


/**
 * @brief Crop transformed array to the ROI.
**/
void OffAxisFilter::crop() {
    int k, l=0;
    int x0 = this->roi[0];
    int xend = this->roi[0] + this->roi[2];
    int y0 = this->roi[1];
    int yend = this->roi[1] + this->roi[3];

    for (auto i=y0; i<yend; i++) {
        for (auto j=x0; j<xend; j++) {
            k = j*this->shape[1] + i;
            this->c[l][0] = this->b[k][0];
            this->c[l][1] = this->b[k][1];
            l += 1;
        }
    }
}


/**
 * @brief Writes numpy array to memory.
 * @param arr Numpy ndarray.
**/
bool OffAxisFilter::write(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> arr) {
    // Access the underlying buffer of the NumPy array
    py::buffer_info buf = arr.request();
    
    // Pointer to the data
    uint8_t *ptr = static_cast<uint8_t *>(buf.ptr);
    
    // Length of the array
    unsigned size = buf.size;

    if (size == this->N) {
        for (auto i=0; i<this->N; i++) {
            this->a[i][0] = this->ref[i].real() * static_cast<double>(ptr[i]);
            this->a[i][1] = this->ref[i].imag() * static_cast<double>(ptr[i]);
        }
        return true;
    } else {
        cerr << "Given interference pattern had an incompatible size." << endl;
        return false;
    }
}


void OffAxisFilter::show(fftw_complex *arr, bool cropped) {
    int N;
    int x;
    int y;
    if (cropped) {
        N = this->roi[2]*this->roi[3];
        x = this->roi[2];
        y = this->roi[3];
    } else {
        N = this->N;
        x = this->shape[0];
        y = this->shape[1];
    }

    complex<double> c;
    double* A = new double[N];
    c.real(arr[0][0]);
    c.imag(arr[0][1]);
    A[0] = pow(abs(c), 2);

    double min_val = A[0];
    double max_val = A[0];
    for (auto i = 1; i < N; i++) {
        c.real(arr[i][0]);
        c.imag(arr[i][1]);
        A[i] = pow(abs(c), 2);
        
        if (A[i] > max_val) max_val = A[i];
        if (A[i] < min_val) min_val = A[i];
    }

    uint8_t* img_data = new uint8_t[N];
    for (auto i = 0; i < N; ++i) {
        img_data[i] = static_cast<uint8_t>(255.0 * (A[i] - min_val) / (max_val - min_val));
    }

    // Create an OpenCV Mat object
    cv::Mat img_mat(x, y, CV_8UC1, img_data);

    // Display the image using OpenCV
    cv::imshow("Image", img_mat);
    cv::waitKey(0);

    delete[] A;
    delete[] img_data;
}

void OffAxisFilter::show_input() {
    this->show(this->a, false);
}

void OffAxisFilter::show_output() {
    this->show(this->d, true);
}