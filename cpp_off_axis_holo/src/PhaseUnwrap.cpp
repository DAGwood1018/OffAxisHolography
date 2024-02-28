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
PhaseUnwrap::PhaseUnwrap(int M, int N, unsigned nthreads, unsigned flags) {
    this->M = M;
    this->N = N; 
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
        cout << "Error in initiating multi-threading." << endl;
    } else {
        fftw_plan_with_nthreads(this->nthreads);
    }

    this->nthreads = fftw_planner_nthreads();
    this->a = (double *) fftw_alloc_real(this->size());
    this->b = (double *) fftw_alloc_real(this->size());
    this->dct_forward = fftw_plan_r2r_2d(this->M, this->N, this->a, this->b, FFTW_REDFT10, FFTW_REDFT10, this->flags);
    this->dct_backward = fftw_plan_r2r_2d(this->M, this->N, this->b, this->a, FFTW_REDFT01, FFTW_REDFT01, this->flags);

    this->phase_wrap = new Eigen::MatrixXd(this->M, this->N);
    this->phase_unwrap = new Eigen::MatrixXd(this->M, this->N);
}


/**
 * @brief Return the total number of elements.
**/
int PhaseUnwrap::size() {
    return this->M * this->N;
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
Eigen::MatrixXd PhaseUnwrap::operator()(Eigen::MatrixXd& phase_wrap) {
    bool res = this->write(phase_wrap);
    Eigen::MatrixXd phase_unwrap = Eigen::MatrixXd::Zero(this->M, this->N);
    if (res) {
        this->unwrap();
        memcpy(phase_unwrap.data(), this->phase_unwrap->data(), sizeof(double)*this->size());
    }

    return phase_unwrap;
}


void PhaseUnwrap::unwrap() {
    // Initialize needed matrices
    Eigen::MatrixXd wrapped_phase(this->M, this->N); 
    Eigen::MatrixXd unwrapped_phase(this->M, this->N);
    Eigen::MatrixXd phi(this->M, this->N);
    Eigen::MatrixXd wrapped_residual(this->M, this->N);
    Eigen::MatrixXd unwrapped_residual(this->M, this->N);
    Eigen::MatrixXd K1(this->M, this->N);
    Eigen::MatrixXd K2(this->M, this->N);

    // Copy input to working matrix
    memcpy(wrapped_phase.data(), this->phase_wrap->data(), sizeof(double)*this->size());

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

    int i=0;
    while (((K2 - K1).cwiseAbs()).sum() > 0) {
        K1 = K2;
        this->solve(wrapped_residual, unwrapped_residual);
        phi += unwrapped_residual;
        phi.array() += wrapped_phase.mean() - phi.mean();  // adjust piston
        K2 = ((phi - wrapped_phase) / (2 * M_PI)).array().round();  // calculate integer K
        unwrapped_phase = wrapped_phase + 2 * K2 * M_PI;
        wrapped_residual = wrap_to_pi(unwrapped_phase - phi);
        i++;
    }

    // Copy result to class property
    memcpy(this->phase_unwrap->data(), unwrapped_phase.data(), sizeof(double)*this->size());
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
    for (auto i = 0; i < this->M; ++i) {
        for (auto j = 0; j < this->N; ++j) {
            if (i==0 && j==0) {
                this->b[0] = 0;
            } else {
                denom = 2 * (cos(M_PI * i / this->M) + cos(M_PI * j / this->N) - 2);
                this->b[i * this->M + j] /=  4 * denom * this->size(); // Achieves the proper normalization
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
    memcpy(this->a, a, sizeof(double)*this->size());
    fftw_execute(this->dct_forward);
}


/**
 * @brief Executes plan for IDCT transform.
**/
void PhaseUnwrap::backwards(double* a) {
    fftw_execute(this->dct_backward);
    memcpy(a, this->a, sizeof(double)*this->size());
}


bool PhaseUnwrap::write(Eigen::MatrixXd& mat) {
    if (mat.size() == this->size()) {
       memcpy(this->phase_wrap->data(), mat.data(), sizeof(double)*this->size());
       return true;
    } else {
        cerr << "Wrapped phase input had an incompatible size." << endl;
        return false;
    }
}


void PhaseUnwrap::show(Eigen::MatrixXd* mat) {
    double min_val = mat->minCoeff();
    double max_val = mat->maxCoeff();

    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> img_data = (255.0*(mat->array() - min_val)/(max_val-min_val)).cast<uint8_t>();

    // Create an OpenCV Mat object
    cv::Mat img_mat(this->M, this->N, CV_8UC1, img_data.data());

    // Display the image using OpenCV
    cv::imshow("Image", img_mat);
    cv::waitKey(0);
}


void PhaseUnwrap::show_input() {
    this->show(this->phase_wrap);
}


void PhaseUnwrap::show_output() {
    this->show(this->phase_unwrap);
}