#ifndef DFT_H
#define DFT_H
#include<complex.h>
#include<fftw3.h>
#include<vector>
using namespace std;

class DFT {

public:
    fftw_complex *a; // Data transformed by forward and written to by backward
    fftw_complex *b; // Data transformed by backward and written to by forward

    DFT(unsigned rank, int *n, unsigned threads, unsigned flags);
    ~DFT();
    bool forwards(vector<complex<double>> *a);
    bool backwards(vector<complex<double>> *b);
    void forwards();
    void backwards();
    int size();

protected:
    unsigned rank; // Dimensionality of transform
    long unsigned N; // Total number of elements of the transform
    int* shape; // Shape of the transform
    unsigned nthreads; // Number of threads to use for multithreading 
    unsigned flags; // FFTW planner flags
    fftw_plan plan_forward; // FFTW plan for fft
	fftw_plan plan_backward; // FFTW plan for ifft

    bool write_array(vector<complex<double>> *arr, bool inverse);
};

#endif