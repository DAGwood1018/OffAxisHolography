#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <complex>
#include <fftw3.h>
#include "phase_unwrapping.h"
using namespace std;

void alloc_unwrap(phase_unwrap *unwrap, int N, int M, int nthreads, unsigned flags) {    
    if (fftw_init_threads()==0) {
        printf("Error in initiating multi-threading.");
    } else {
        fftw_plan_with_nthreads(nthreads);
    }

    unwrap->N = N;
    unwrap->M = M;
    unwrap->flags = flags;
    unwrap->nthreads = fftw_planner_nthreads();
    unwrap->in = (double *) fftw_alloc_real(N);
    unwrap->out = (double *) fftw_alloc_real(N);
    unwrap->plan_forward = fftw_plan_r2r_2d(N, M, unwrap->in, unwrap->out, FFTW_REDFT10, FFTW_REDFT10, flags);
    unwrap->plan_backward = fftw_plan_r2r_2d(N, M, unwrap->out, unwrap->in, FFTW_REDFT01, FFTW_REDFT10, flags);
} 