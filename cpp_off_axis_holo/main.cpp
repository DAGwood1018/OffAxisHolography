#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <complex>
#include <fftw3.h>
#include "DFT.h"
using namespace std;

int main() {
    int N = 5;
    unsigned rank = 2;
    vector<complex<double>> *in = new vector<complex<double>>(N*N);
    int shape[2] = {N, N};

    /* prepare a cosine wave */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            (*in)[i*N+j] = complex<double>(cos(6*M_PI*i/N + 6*M_PI*j/N), 0);
        }
    }

    for (int i = 0; i < N * N; i++) {
        printf("freq: %3d %+9.5f %+9.5f I\n", i, real((*in)[i]), imag((*in)[i]));
    }

    DFT *test = new DFT(rank, shape, 4, FFTW_MEASURE);

    bool res = test->forwards(in);
    cout << res << endl;

    cout << endl;

    for (int i = 0; i < N*N; i++) {
        printf("freq: %3d %+9.5f %+9.5f I\n", i, test->b[i][0], test->b[i][1]);
    }
    
    test->backwards();

    cout << endl;

    for (int i = 0; i < N*N; i++) {
        printf("freq: %3d %+9.5f %+9.5f I\n", i, test->a[i][0]/test->size(), test->a[i][1]/test->size());
    }
    delete test;
    fftw_cleanup_threads();
    
    return 0;
}