#ifndef DFT_H
#define DFT_H
#include<complex.h>
#include<fftw3.h>
using namespace std;

typedef struct DFT {
    int rank;
    int size;
    int* shape;
    int nthreads;
    unsigned flags;
    fftw_plan plan_forward;
	fftw_plan plan_backward;
    fftw_complex *in;
    fftw_complex *out;
} DFT;

void alloc_dft(DFT *dft_plan, int *n, int nthreads, unsigned flags);

void delete_dft(DFT *dft_plan);

void forwards(DFT *dft_plan);

void backwards(DFT *dft_plan);

void write_in(DFT *dft_plan, fftw_complex *a);

void write_out(DFT *dft_plan, fftw_complex *b);

#endif