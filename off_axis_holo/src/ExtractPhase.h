#ifndef EXTRACT_H
#define EXTRACT_H

#include "OffAxisFilter.h"
#include "PhaseUnwrap.h"

using namespace std;
namespace py = pybind11;


class ExtractPhase {

public:
    ExtractPhase(py::array_t<int, py::array::c_style | py::array::forcecast> n, py::array_t<int, py::array::c_style | py::array::forcecast> roi, 
                py::array_t<double, py::array::c_style | py::array::forcecast> mask, py::array_t<complex<double>, py::array::c_style | py::array::forcecast> ref,
                unsigned threads, unsigned flags);
    ~ExtractPhase();
    
    Eigen::MatrixXd __call__(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> fringes);

    int input_size();
    int output_size(); 
    int threads_in_use();
    void show_filter_input();
    void show_filter_output();
    void show_phase_input();
    void show_phase_output();

protected:
    OffAxisFilter *filter;
    PhaseUnwrap *unwrap;
};

#endif