#include "PhaseUnwrap.h"
#include <opencv2/opencv.hpp>
#include<stdexcept>
#include<cmath>
#include<numeric>
#include<iostream>

using namespace std;

Eigen::MatrixXd wrap_to_pi(const Eigen ::MatrixXd& angles) {
    Eigen::MatrixXd a(angles.rows(), angles.cols());
    for (auto i = 0; i < angles.rows(); i++) {
        for (auto j = 0; j < angles.cols(); j++) {
            a(i, j) = fmod(angles(i, j) + M_PI, 2*M_PI);
            if (a(i, j) < 0) a(i, j) += 2 * M_PI;
            a(i, j) -= M_PI;
        }
    }
    return a;
}


/**
* @brief Initialize phase unwrapping object.
**/
PhaseUnwrap::PhaseUnwrap(int N, int M, unsigned nthreads, unsigned flags) {
    int shape[2] = {N, M};
    this->shape = shape;
    this->N = this->shape[0] * this->shape[1]; 
    this->flags = flags;
    this->nthreads = nthreads;
    this->alloc();
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
    delete this->phase_wrap;
    delete this->phase_unwrap;
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

    this->phase_wrap = new Eigen::MatrixXd(this->shape[0], this->shape[1]);
    this->phase_unwrap = new Eigen::MatrixXd(this->shape[0], this->shape[1]);
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
 * @param phase Numpy ndarray.
**/
py::array_t<double> PhaseUnwrap::__call__(py::array_t<double, py::array::c_style | py::array::forcecast> phase_wrap) {
    bool res = this->unwrap(phase_wrap);
    //this->test(phase_wrap);
    size_t shape[2] = {static_cast<size_t>(this->shape[0]), static_cast<size_t>(this->shape[1])};
    py::array_t<double> phase_unwrap(shape);
    py::buffer_info buffer = phase_unwrap.request();
    double* ptr = static_cast<double *>(buffer.ptr);
    
    if (res) {
        memcpy(ptr, this->phase_unwrap->data(), sizeof(double)*this->N);
    } else {
        for (auto i = 0; i < this->N; i++) {
            ptr[i] = 0;
        } 
    }

    return phase_unwrap;
}


bool PhaseUnwrap::unwrap(py::array_t<double, py::array::c_style | py::array::forcecast> phase_wrap) {
    bool res = this->write(phase_wrap);
    
    // Unwrap given phase
    if (res) {

        // Initialize needed matrices
        Eigen::MatrixXd wrapped_phase(this->shape[0], this->shape[1]);
        Eigen::MatrixXd unwrapped_phase(this->shape[0], this->shape[1]);
        Eigen::MatrixXd phi(this->shape[0], this->shape[1]);
        Eigen::MatrixXd wrapped_residual(this->shape[0], this->shape[1]);
        Eigen::MatrixXd unwrapped_residual(this->shape[0], this->shape[1]);
        Eigen::MatrixXd K1(this->shape[0], this->shape[1]);
        Eigen::MatrixXd K2(this->shape[0], this->shape[1]);

        // Copy input to working matrix
        memcpy(wrapped_phase.data(), this->phase_wrap->data(), sizeof(double)*this->N);

        this->solve(wrapped_phase, phi);
        phi.array() += wrapped_phase.mean() - phi.mean();  // adjust piston

        K1 = ((phi - wrapped_phase) / (2 * M_PI)).array().round();  // calculate integer K
        unwrapped_phase = wrapped_phase + 2 * K1 * M_PI;
        wrapped_residual = wrap_to_pi(unwrapped_phase - phi);

        this->solve(wrapped_residual, unwrapped_residual);
        phi += unwrapped_residual;
        phi.array() += wrapped_phase.mean() - phi.mean();  // adjust piston

        K2 = ((phi - wrapped_phase) / (2 * M_PI)).array().round();  // calculate integer K
        unwrapped_phase = wrapped_phase + 2 * K2 * M_PI;
        wrapped_residual = wrap_to_pi(unwrapped_phase - phi);

        while (((K2 - K1).cwiseAbs()).sum() > 0) {
            K1 = K2;
            this->solve(wrapped_residual, unwrapped_residual);
            phi += unwrapped_residual;
            phi.array() += wrapped_phase.mean() - phi.mean();  // adjust piston
            K2 = ((phi - wrapped_phase) / (2 * M_PI)).array().round();  // calculate integer K
            unwrapped_phase = wrapped_phase + 2 * K2 * M_PI;
            wrapped_residual = wrap_to_pi(unwrapped_phase - phi);
        }

        // Copy result to class property
        memcpy(this->phase_unwrap->data(), unwrapped_phase.data(), sizeof(double)*this->N);
    }
    return res;
}


void PhaseUnwrap::solve(Eigen::MatrixXd& in, Eigen::MatrixXd& out) {

    Eigen::MatrixXd edx(in.rows(), in.cols()+1);
    Eigen::MatrixXd edy(in.rows()+1, in.cols());
    Eigen::MatrixXd lap(in.rows(), in.cols());

     // Calculate edx
    edx << Eigen::MatrixXd::Zero(in.rows(), 1), wrap_to_pi(in.rightCols(in.cols() - 1) - in.leftCols(in.cols() - 1)), Eigen::MatrixXd::Zero(in.rows(), 1);

    // Calculate edy
    edy << Eigen::MatrixXd::Zero(1, in.cols()), wrap_to_pi(in.bottomRows(in.rows() - 1) - in.topRows(in.rows() - 1)), Eigen::MatrixXd::Zero(1, in.cols());

    // Calculate laplacian
    lap = (edx.rightCols(edx.cols() - 1) - edx.leftCols(edx.cols() - 1)) + (edy.bottomRows(edy.rows() - 1) - edy.topRows(edy.rows() - 1));

    // Execute forward DCT transform
    this->forwards(lap.data());

    // TODO double check normalization
    double denom;
    for (auto i = 0; i < this->shape[0]; ++i) {
        for (auto j = 0; j < this->shape[1]; ++j) {
            if (i==0 && j==0) {
                this->b[0] = 0;
            } else {
                denom = 2 * (cos(M_PI * i / this->shape[0]) + cos(M_PI * j / this->shape[1]) - 2);
                this->b[i * this->shape[0] + j] /=  4 * denom * N; // Achieves the proper normalization
            }
        }
    } 

    // Execute backward DCT transform
    this->backwards(out.data()); 
}

/**
 * @brief Executes plan for DCT transform.
**/
void PhaseUnwrap::forwards(double* a) {
    memcpy(this->a, a, sizeof(double)*this->N);
    fftw_execute(this->dct_forward);
}


/**
 * @brief Executes plan for IDCT transform.
**/
void PhaseUnwrap::backwards(double* a) {
    fftw_execute(this->dct_backward);
    memcpy(a, this->a, sizeof(double)*this->N);
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
       memcpy(this->phase_wrap->data(), ptr, sizeof(double)*this->N);
       return true;
    } else {
        return false;
    }
}


void PhaseUnwrap::show(Eigen::MatrixXd* mat) {
    double min_val = mat->minCoeff();
    double max_val = mat->maxCoeff();

    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> img_data = (255.0*(mat->array() - min_val)/(max_val-min_val)).cast<uint8_t>();

    // Create an OpenCV Mat object
    cv::Mat img_mat(this->shape[0], this->shape[1], CV_8UC1, img_data.data());

    // Display the image using OpenCV
    cv::imshow("Image", img_mat);
    cv::waitKey(0);
}


void PhaseUnwrap::show_wrapped() {
    this->show(this->phase_wrap);
}


void PhaseUnwrap::show_unwrapped() {
    this->show(this->phase_unwrap);
}


// Create Pybind11 Module
PYBIND11_MODULE(phase_unwrap, m) {
        py::class_<PhaseUnwrap>(m, "PhaseUnwrap")
                .def(py::init<py::array_t<int, py::array::c_style | py::array::forcecast>, unsigned, unsigned>())
                .def("__call__", static_cast<py::array_t<double> (PhaseUnwrap::*)(py::array_t<double, py::array::c_style | py::array::forcecast>)>(&PhaseUnwrap::__call__))
                .def("unwrap", static_cast<bool (PhaseUnwrap::*)(py::array_t<double, py::array::c_style | py::array::forcecast>)>(&PhaseUnwrap::unwrap))
                .def("size", &PhaseUnwrap::size)
                .def("threads_in_use", &PhaseUnwrap::threads_in_use)
                .def("show_wrapped", &PhaseUnwrap::show_wrapped)
                .def("show_unwrapped", &PhaseUnwrap::show_unwrapped)
                ;
}