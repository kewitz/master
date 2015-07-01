/*
 The MIT License (MIT)

 Copyright (c) 2014 Leonardo Kewitz

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "./cuda_snippets.h"
#include "./escheme.h"


void integ_element(element *E, float *rsum) {
    float J1, J2, J3, J4, dJ, mat = E->mat, f = E->f;
    // Calcula argumentos necessários
    J1 = E->x[1] - E->x[0];
    J2 = E->y[1] - E->y[0];
    J3 = E->x[2] - E->x[0];
    J4 = E->y[2] - E->y[0];
    dJ = 2*(J1*J4 - J3*J2)/mat;

    // Calcula a matriz de contribuições do elemento.
    E->matriz[0] = dJ != 0.0f ? (pow(J2-J4, 2.0f) + pow(J3-J1, 2.0f))/dJ : 0.0f;
    E->matriz[1] = dJ != 0.0f ? (pow(J4, 2.0f) + pow(J3, 2.0f))/dJ : 0.0f;
    E->matriz[2] = dJ != 0.0f ? (pow(J2, 2.0f) + pow(J1, 2.0f))/dJ : 0.0f;
    E->matriz[3] = dJ != 0.0f ? ((J2-J4)*J4 - (J3-J1)*J3)/dJ : 0.0f;
    E->matriz[4] = dJ != 0.0f ? ((J3-J1)*J1 - (J2-J4)*J2)/dJ : 0.0f;
    E->matriz[5] = dJ != 0.0f ? (J4*-1*J2 - J3*J1)/dJ : 0.0f;

    if (f > 0.0f) {
        f = f*dJ*0.5f*0.333f;
        rsum[E->nodes[0]] += f;
        rsum[E->nodes[1]] += f;
        rsum[E->nodes[2]] += f;
    }
}

void zera_vetor(float *V, int n) {
    for (int i = 0; i < n; i++)
        V[i] = 0.0f;
}

extern "C" int runCPU(int ng, int nn, int kmax, float errmin, group *groups,
    float *V, float *S, bool verbose, float *bench) {

    float *rsum = cast(float*, malloc(nn*sizeof(float)));
    float *dsum = cast(float*, malloc(nn*sizeof(float)));

    zera_vetor(rsum, nn);
    zera_vetor(dsum, nn);

    for (i = 0; i < ng; i++) {
        G = &groups[i];
        for (j = 0; j < G->ne; j++) {
            E = &G->elements[j];
            integ_element(E, rsum);

            unsigned int *N = E->nodes;

            dsum[n1] += E->matriz[0];
            dsum[n2] += E->matriz[1];
            dsum[n3] += E->matriz[2];
        }
    }
}
