#include "OffAxFilter.h"
#include "PhaseUnwrap.h"

// Create Pybind11 Module
PYBIND11_MODULE(holo_lib, m) {
        py::class_<OffAxFilter>(m, "OffAxFilter")
                .def(py::init<py::array_t<int, py::array::c_style | py::array::forcecast>, py::array_t<int, py::array::c_style | py::array::forcecast>, 
                        py::array_t<double, py::array::c_style | py::array::forcecast>, py::array_t<complex<double>, py::array::c_style | py::array::forcecast>,
                        unsigned, unsigned>())
                .def("__call__", static_cast<py::array_t<double> (OffAxFilter::*)(py::array_t<uint8_t, py::array::c_style | py::array::forcecast>)>(&OffAxFilter::__call__))
                .def("filter", static_cast<bool (OffAxFilter::*)(py::array_t<uint8_t, py::array::c_style | py::array::forcecast>)>(&OffAxFilter::filter))
                .def("size", &OffAxFilter::size)
                .def("threads_in_use", &OffAxFilter::threads_in_use)
                .def("show_input", &OffAxFilter::show_input)
                .def("show_output", &OffAxFilter::show_output)
                ;

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