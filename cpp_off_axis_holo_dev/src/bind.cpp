#include "DFT.h"
#include "OffAxFilter.h"

// Create Pybind11 Module
PYBIND11_MODULE(holo_lib, m) {
     py::class_<DFT>(m, "DFT")
        .def(py::init<py::array_t<int>, unsigned, unsigned, bool>())
        .def("forwards", static_cast<bool (DFT::*)(py::array_t<complex<double>>)>(&DFT::forwards))
        .def("backwards", static_cast<bool (DFT::*)(py::array_t<complex<double>>)>(&DFT::backwards))
        .def("size", &DFT::size);

    py::class_<OffAxFilter>(m, "OffAxFilter")
        .def(py::init<py::array_t<int>, py::array_t<int>, py::array_t<bool>, 
                    py::array_t<complex<double>>, py::array_t<double>, unsigned, unsigned>())
        .def("filter", static_cast<bool (OffAxFilter::*)(py::array_t<complex<double>>)>(&OffAxFilter::filter))
        .def("size", &OffAxFilter::size)
        .def("to_string", &OffAxFilter::to_string)
        ;
}