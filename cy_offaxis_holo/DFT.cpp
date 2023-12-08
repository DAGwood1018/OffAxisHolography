#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <complex>
#include <fftw3.h>
#include "DFT.h"
using namespace std;

void alloc_dft(DFT *dft_plan, int rank, int *n, int nthreads, unsigned flags) {
    int N = 1;
    for (int i = 0; i < rank; i++) {
        N *= n[i];
    }
    
    if (fftw_init_threads()==0) {
        printf("Error in initiating multi-threading.");
    } else {
        fftw_plan_with_nthreads(nthreads);
    }

    dft_plan->rank = rank;
    dft_plan->size = N;
    dft_plan->shape = n;
    dft_plan->flags = flags;
    dft_plan->nthreads = fftw_planner_nthreads();
    dft_plan->in = (fftw_complex *) fftw_alloc_complex(N);
    dft_plan->out = (fftw_complex *) fftw_alloc_complex(N);
    dft_plan->plan_forward = fftw_plan_dft(rank, n, dft_plan->in, dft_plan->out, FFTW_FORWARD, flags);
    dft_plan->plan_backward = fftw_plan_dft(rank, n, dft_plan->out, dft_plan->in, FFTW_BACKWARD, flags);
}

void delete_dft(DFT *dft_plan) {
    fftw_free(dft_plan->in);
    fftw_free(dft_plan->out);
    fftw_destroy_plan(dft_plan->plan_forward);
    fftw_destroy_plan(dft_plan->plan_backward);
}

void forwards(DFT *dft_plan) {
    fftw_execute(dft_plan->plan_forward);
}

void backwards(DFT *dft_plan) {
    fftw_execute(dft_plan->plan_backward);
}

void write_in(DFT *dft_plan, fftw_complex *a) {
    for (int i=0; i<dft_plan->size; i++) {
        dft_plan->in[i][0] = a[i][0];
        dft_plan->in[i][1] = a[i][1];
    }
}

void write_out(DFT *dft_plan, fftw_complex *b) {
    for (int i=0; i<dft_plan->size; i++) {
        dft_plan->out[i][0] = b[i][0];
        dft_plan->out[i][1] = b[i][1];
    }
}

int main() {
    int N = 5;
    int rank = 2;
    fftw_complex in[N*N];
    int shape[2] = {N, N};

    /* prepare a cosine wave */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            in[i*N+j][0] = cos(6*M_PI*i/N + 6*M_PI*j/N);
            in[i*N+j][1] = 0;
        }
    }

    for (int i = 0; i < N * N; i++) {
        printf("freq: %3d %+9.5f %+9.5f I\n", i, in[i][0], in[i][1]);
    }

    DFT *test = new DFT;
    alloc_dft(test, rank, shape, 4, FFTW_MEASURE);

    write_in(test, in);

    for (int i = 0; i < N*N; i++) {
        printf("freq: %3d %+9.5f %+9.5f I\n", i, test->in[i][0], test->in[i][1]);
    }

    forwards(test);

    cout << endl;

    for (int i = 0; i < N*N; i++) {
        printf("freq: %3d %+9.5f %+9.5f I\n", i, test->out[i][0], test->out[i][1]);
    }
    
    backwards(test);

    cout << endl;

    for (int i = 0; i < N*N; i++) {
        printf("freq: %3d %+9.5f %+9.5f I\n", i, test->in[i][0]/test->size, test->in[i][1]/test->size);
    }
    delete_dft(test);
    fftw_cleanup_threads();
    
    return 0;
}