#include "OffAxisModule.h"
#include<iostream>

using namespace std;
namespace py = pybind11;

/**
* @brief Initialize off-axis filter object.
**/
OffAxisModule::OffAxisModule(py::array_t<int, py::array::c_style | py::array::forcecast> n, py::array_t<int, py::array::c_style | py::array::forcecast> roi, 
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
OffAxisModule::~OffAxisModule() {
    delete this->filter;
    delete this->unwrap;
}


/**
 * @brief Python call function that filters interfence pattern, unwraps the phase, and returns a numpy array.
 * @param fringes Numpy ndarray.
**/
Eigen::MatrixXd OffAxisModule::__call__(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> fringes) {
    Eigen::MatrixXd phase_wrap = this->filter->operator()(fringes);
    return this->unwrap->execute(phase_wrap);
}


/**
 * @brief Return the total element count of an input.
**/
int OffAxisModule::input_size() {
    return this->filter->size();
}


/**
 * @brief Return the total element count of an output.
**/
int OffAxisModule::output_size() {
    return this->unwrap->size();
}


/**
 * @brief Returns the total number of threads in use.
**/
int OffAxisModule::threads_in_use() {
    return this->filter->threads_in_use() + this->unwrap->threads_in_use();
}


void OffAxisModule::show_filter_input() {
    this->filter->show_input();
}

void OffAxisModule::show_filter_output() {
    this->filter->show_output();
}

void OffAxisModule::show_phase_input() {
    this->unwrap->show_input();
}

void OffAxisModule::show_phase_output() {
    this->unwrap->show_output();
}


PYBIND11_MODULE(off_axis_module, m) {
        py::class_<OffAxisModule>(m, "OffAxisModule")
                .def(py::init<py::array_t<int, py::array::c_style | py::array::forcecast>, py::array_t<int, py::array::c_style | py::array::forcecast>, 
                        py::array_t<double, py::array::c_style | py::array::forcecast>, py::array_t<complex<double>, py::array::c_style | py::array::forcecast>,
                        unsigned, unsigned>())
                .def("__call__", static_cast<Eigen::MatrixXd (OffAxisModule::*)(py::array_t<uint8_t, py::array::c_style | py::array::forcecast>)>(&OffAxisModule::__call__))
                .def("input_size", &OffAxisModule::input_size)
                .def("output_size", &OffAxisModule::output_size)
                .def("threads_in_use", &OffAxisModule::threads_in_use)
                .def("show_filter_input", &OffAxisModule::show_filter_input)
                .def("show_filter_output", &OffAxisModule::show_filter_output)
                .def("show_phase_input", &OffAxisModule::show_phase_input)
                .def("show_phase_output", &OffAxisModule::show_phase_output)
                ;
}