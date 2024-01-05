#include "OffAxFilter.h"

/**
* @brief Allocate memory and plan DFT.
* @param n Numpy array giving shape.
* @param nthreads Integer number of threads to use.
* @param flags FFTW planner flags to use. Combine using bitwise OR. 
* @param ortho Whehter the transformation should be orthogonal.
**/
OffAxFilter::OffAxFilter(py::array_t<int> dims, py::array_t<int> roi, py::array_t<int> mask, 
                py::array_t<double> window, unsigned threads, unsigned flags) : DFT(dims, threads, flags, false) {
    
    // Access numpy array
    py::buffer_info buf = roi.request();
    int *roi = static_cast<int *>(buf.ptr);
    
    this->set_mask(mask);
    this->set_window(window);
}

/**
 * @brief Free memory for off-axis filter.
**/
OffAxFilter::~OffAxFilter() :  {
    delete[] this->window;
    delete[] this->mask;
}

/**
 * @brief Processes numpy array giving the mask.
 * @param  Numpy ndarray.
**/
bool OffAxFilter:set_mask:(py::array_t<int> mask) {
    // Allocate pointer to window array
    this->mask = new int[this->N]

    // Access the underlying buffer of the NumPy array
    py::buffer_info buf = window.request();
    
    // Pointer to the data
    int *ptr = static_cast<int *>(buf.ptr);
    
    // Length of the array
    unsigned size = buf.size;

    if (size == this->N) {
        for (unsigned i=0; i<this->N; i++) {
           this->mask[i] = ptr[i]
        }
        return true;
    } else {
        return false;
    }
}

/**
 * @brief Processes numpy array giving the window function.
 * @param window Numpy ndarray.
**/
bool OffAxFilter:set_window:(py::array_t<double> window) {
    // Allocate pointer to window array
    this->window = new double[this->N]

    // Access the underlying buffer of the NumPy array
    py::buffer_info buf = window.request();
    
    // Pointer to the data
    double *ptr = static_cast<double *>(buf.ptr);
    
    // Length of the array
    unsigned size = buf.size;

    if (size == this->N) {
        for (unsigned i=0; i<this->N; i++) {
           this->window[i] = ptr[i]
        }
        return true;
    } else {
        return false;
    }
}

PYBIND11_MODULE(offax_lib, m) {
    py::class_<OffAxFilter, DFT>(m, "OffAxFilter")
        .def(py::init<py::array_t<int>, py::array_t<int>, py::array_t<int>, py::array_t<double>, unsigned, unsigned>())
}
