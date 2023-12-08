#ifndef PHASE_UNWRAP_H
#define PHASE_UNWRAP_H
#include<complex.h>
#include<fftw3.h>
using namespace std;

typedef struct phase_unwrap {
    int N;
    int M;
    int nthreads;
    unsigned flags;
    fftw_plan plan_forward;
	fftw_plan plan_backward;
    double *in;
    double *out;
} phase_unwrap;

void alloc_unwrap(phase_unwrap *unwrap, int N, int M, int nthreads, unsigned flags);

void delete_unwrap(phase_unwrap *unwrap);

void forwards(phase_unwrap *unwrap);

void backwards(phase_unwrap *unwrap);

void write_in(phase_unwrap *redft_plan, double *a);

void write_out(phase_unwrap *un, double *b);

#endif