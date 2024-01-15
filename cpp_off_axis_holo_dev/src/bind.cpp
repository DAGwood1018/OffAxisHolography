#include "DFT.h"
#include "OffAxFilter.h"

// Create Pybind11 Module
PYBIND11_MODULE(holo_lib, m) {
     py::class_<DFT>(m, "DFT")
        .def(py::init<py::array_t<int>, unsigned, unsigned, bool>())
        .def("forwards", static_cast<bool (DFT::*)(py::array_t<complex<double>>)>(&DFT::forwards))
        .def("backwards", static_cast<bool (DFT::*)(py::array_t<complex<double>>)>(&DFT::backwards))
        .def("size", &DFT::size)
        .def("threads_in_use", &DFT::threads_in_use);

    py::class_<OffAxFilter>(m, "OffAxFilter")
        .def(py::init<py::array_t<int, py::array::c_style | py::array::forcecast>, py::array_t<int, py::array::c_style | py::array::forcecast>, py::array_t<bool, py::array::c_style | py::array::forcecast>, 
                    py::array_t<complex<double>, py::array::c_style | py::array::forcecast>, py::array_t<double, py::array::c_style | py::array::forcecast>, unsigned, unsigned>())
        .def("__call__", static_cast<py::array_t<complex<double>> (OffAxFilter::*)(py::array_t<complex<double>, py::array::c_style | py::array::forcecast>)>(&OffAxFilter::__call__))
        .def("filter", static_cast<bool (OffAxFilter::*)(py::array_t<complex<double>, py::array::c_style | py::array::forcecast>)>(&OffAxFilter::filter))
        .def("size", &OffAxFilter::size)
        .def("threads_in_use", &OffAxFilter::threads_in_use)
        .def("to_string", &OffAxFilter::to_string)
        ;
}