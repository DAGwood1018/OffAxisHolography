#include "ExtractPhase.h"
#include<iostream>

using namespace std;
namespace py = pybind11;

/**
* @brief Initialize off-axis filter object.
**/
ExtractPhase::ExtractPhase(py::array_t<int, py::array::c_style | py::array::forcecast> n, py::array_t<int, py::array::c_style | py::array::forcecast> roi, 
                        py::array_t<double, py::array::c_style | py::array::forcecast> mask, py::array_t<complex<double>, py::array::c_style | py::array::forcecast> ref,
                        unsigned nthreads, unsigned flags) {

    py::buffer_info roi_buf = roi.request();
    if (roi_buf.size != 4) {
        throw py::value_error("Expect (x, y, dx, dy) for ROI.");
    }
    int* ROI = static_cast<int *>(roi_buf.ptr);
    
    this->filter = new OffAxisFilter(n, roi, mask, ref, nthreads, flags);
    this->unwrap = new PhaseUnwrap(ROI[2], ROI[3], nthreads, flags);
}

/**
 * @brief Free memory for off-axis filter and phase unwrapper.
**/
ExtractPhase::~ExtractPhase() {
    delete this->filter;
    delete this->unwrap;
}


/**
 * @brief Python call function that filters interfence pattern, unwraps the phase, and returns a numpy array.
 * @param fringes Numpy ndarray.
**/
Eigen::MatrixXd ExtractPhase::__call__(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> fringes) {
    Eigen::MatrixXd phase_wrap = this->filter->operator()(fringes);
    return this->unwrap->operator()(phase_wrap, true);
}


/**
 * @brief Return the total element count of an input.
**/
int ExtractPhase::input_size() {
    return this->filter->size();
}


/**
 * @brief Return the total element count of an output.
**/
int ExtractPhase::output_size() {
    return this->unwrap->size();
}


/**
 * @brief Returns the total number of threads in use.
**/
int ExtractPhase::threads_in_use() {
    return this->filter->threads_in_use() + this->unwrap->threads_in_use();
}


void ExtractPhase::show_filter_input() {
    this->filter->show_input();
}

void ExtractPhase::show_filter_output() {
    this->filter->show_output();
}

void ExtractPhase::show_phase_input() {
    this->unwrap->show_input();
}

void ExtractPhase::show_phase_output() {
    this->unwrap->show_output();
}


PYBIND11_MODULE(off_axis_module, m) {
        py::class_<ExtractPhase>(m, "ExtractPhase")
                .def(py::init<py::array_t<int, py::array::c_style | py::array::forcecast>, py::array_t<int, py::array::c_style | py::array::forcecast>, 
                        py::array_t<double, py::array::c_style | py::array::forcecast>, py::array_t<complex<double>, py::array::c_style | py::array::forcecast>,
                        unsigned, unsigned>())
                .def("__call__", static_cast<Eigen::MatrixXd (ExtractPhase::*)(py::array_t<uint8_t, py::array::c_style | py::array::forcecast>)>(&ExtractPhase::__call__))
                .def("input_size", &ExtractPhase::input_size)
                .def("output_size", &ExtractPhase::output_size)
                .def("threads_in_use", &ExtractPhase::threads_in_use)
                .def("show_filter_input", &ExtractPhase::show_filter_input)
                .def("show_filter_output", &ExtractPhase::show_filter_output)
                .def("show_phase_input", &ExtractPhase::show_phase_input)
                .def("show_phase_output", &ExtractPhase::show_phase_output)
                ;
}