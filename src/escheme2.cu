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

extern "C" unsigned int alloc(const int nn) {
    cudaDeviceReset();
    cudaDeviceProp prop = getInfo();
    unsigned int gm = prop.totalGlobalMem*.8 - sizeof(float)*nn*6
                      - sizeof(float)*4;
    return cast(unsigned int, gm / (sizeof(node) + 6*sizeof(element)));
}

void integ_element(element *E, float *S) {
    float mat = E->mat, f = E->f;
    // Calcula gradN
    float q1 = E->y[1]-E->y[2], q2 = E->y[2]-E->y[0], q3 = E->y[0]-E->y[1];
    float r1 = E->x[2]-E->x[1], r2 = E->x[0]-E->x[2], r3 = E->x[1]-E->x[0];
    // Calcula det(gradN)
    float det = E->x[1]*E->y[2] + E->x[0]*E->y[1] + E->x[2]*E->y[0]
              - E->x[0]*E->y[2] - E->x[2]*E->y[1] - E->x[1]*E->y[0];
    float cof = (mat/det)/2.0;
    // Calcula a matriz de contribuições do elemento.
    E->matriz[0] = (q1*q1 + r1*r1)*cof;
    E->matriz[1] = (q2*q2 + r2*r2)*cof;
    E->matriz[2] = (q3*q3 + r3*r3)*cof;
    E->matriz[3] = (q1*q2 + r1*r2)*cof;
    E->matriz[4] = (q1*q3 + r1*r3)*cof;
    E->matriz[5] = (q2*q3 + r2*r3)*cof;

    f = f*(det/6.0);
    S[E->nodes[0]] += f;
    S[E->nodes[1]] += f;
    S[E->nodes[2]] += f;
}

void zera_vetor(float *vec, int n) {
    for (int i = 0; i < n; i++)
        vec[i] = 0.0f;
}

void transpose_multiply(element *E, float *vec, float *res) {
    unsigned int *N = E->nodes;

    res[N[0]] += E->matriz[0]*vec[N[0]] + E->matriz[3]*vec[N[1]]
                 + E->matriz[4]*vec[N[2]];
    res[N[1]] += E->matriz[1]*vec[N[1]] + E->matriz[3]*vec[N[0]]
                 + E->matriz[5]*vec[N[2]];
    res[N[2]] += E->matriz[2]*vec[N[2]] + E->matriz[4]*vec[N[0]]
                 + E->matriz[5]*vec[N[1]];
}

float inner_product(int size, float *vec) {
    float product = 0.0f;
    for (int i = 0; i < size; i++)
        product += pow(vec[i], 2);
    return product;
}

float vector_norm(int size, float *vec) {
    float product = 0.0f;
    for (int i = 0; i < size; i++)
        product += abs(pow(vec[i], 2));
    return product;
}

extern "C" int runCPU(int ng, int nn, int kmax, float errmin, group *groups,
    float *V, float *S, bool verbose, float *bench) {

    int i, j;
    element *E;
    group *G;
    node N;

    size_t vecSize = nn*sizeof(float);
    float *rsum = cast(float*, malloc(vecSize));
    float *dsum = cast(float*, malloc(vecSize));
    float *R = cast(float*, malloc(vecSize));
    float *P = cast(float*, malloc(vecSize));
    float *U = cast(float*, malloc(vecSize));

    zera_vetor(rsum, nn);
    zera_vetor(dsum, nn);
    zera_vetor(R, nn);
    zera_vetor(P, nn);

    float vnorm = vector_norm(nn, V);
    for (i = 0; i < ng; i++) {
        G = &groups[i];
        for (j = 0; j < G->ne; j++) {
            E = &G->elements[j];
            integ_element(E, rsum);
        }
        for (j = 0; j < G->ne; j++) {
            E = &G->elements[j];
            unsigned int *N = E->nodes;

            dsum[N[0]] += E->matriz[0];
            dsum[N[1]] += E->matriz[1];
            dsum[N[2]] += E->matriz[2];

            rsum[N[0]] += -E->matriz[3]*V[N[1]] - E->matriz[4]*V[N[2]];
            rsum[N[1]] += -E->matriz[3]*V[N[0]] - E->matriz[5]*V[N[2]];
            rsum[N[2]] += -E->matriz[4]*V[N[0]] - E->matriz[5]*V[N[1]];
        }
    }

    for (i = 0; i < ng; i++) {
        G = &groups[i];
        for (j = 0; j < G->nn; j++) {
            N = G->nodes[j];
            R[N.i] += S[N.i] + rsum[N.i] - dsum[N.i]*V[N.i];
            S[N.i] = 0.0;
        }

        // zera_vetor(S, nn);
        for (j = 0; j < G->ne; j++) {
            E = &G->elements[j];
            integ_element(E, S);
        }
        for (j = 0; j < G->ne; j++) {
            E = &G->elements[j];
            transpose_multiply(E, R, P);
        }
    }
    float sum1 = inner_product(nn, P);
    for (i = 0; i < nn; i++)
        P[i] /= sum1;

    int k = 1;
    float alpha, beta, err = 1;
    while (k < kmax && err > errmin) {
        zera_vetor(U, nn);
        zera_vetor(S, nn);
        for (i = 0; i < ng; i++) {
            G = &groups[i];
            for (j = 0; j < G->ne; j++) {
                E = &G->elements[j];
                integ_element(E, S);
                transpose_multiply(E, P, U);
            }
        }
        sum1 = inner_product(nn, U);
        alpha = 1.0f/sum1;

        for (i = 0; i < nn; i++) {
            V[i] += alpha*P[i];
            R[i] -= alpha*U[i];
        }

        zera_vetor(U, nn);
        zera_vetor(S, nn);
        for (i = 0; i < ng; i++) {
            G = &groups[i];
            for (j = 0; j < G->ne; j++) {
                E = &G->elements[j];
                integ_element(E, S);
                transpose_multiply(E, R, U);
            }
        }

        sum1 = inner_product(nn, U);
        beta = 1.0f/sum1;

        for (i = 0; i < nn; i++) {
            P[i] += beta*U[i];
        }

        err = sqrt(vector_norm(nn, R)/vnorm);
        k++;
    }

    free(rsum);
    free(dsum);
    free(R);
    free(P);
    free(U);

    return k;
}

extern "C" int testeCG(int n, int kmax, float errmin, float* A, float* x,
    float* b) {

    int i, j;
    float product, alpha, beta;

    float *R = cast(float*, malloc(sizeof(float)*n));
    float *P = cast(float*, malloc(sizeof(float)*n));
    float *U = cast(float*, malloc(sizeof(float)*n));

    // R[0] = b - Ax[0]
    for (j = 0; j < n; j++) {
        R[j] = 0.0f;
        for (i = 0; i < n; i++)
            R[j] += b[i] - A[j*n + i]*x[i];
    }

    // P[0] = A*R/<A*R,A*R>
    for (j = 0; j < n; j++) {
        P[j] = 0.0f;
        for (i = 0; i < n; i++)
            P[j] += A[i*n + j]*x[i];
    }
    product = 0.0f;
    for (i = 0; i < n; i++)
        product += P[i]*P[i];
    for (i = 0; i < n; i++)
        P[i] /= product;

    int k = 1;
    float err = 1, e;
    while (k < kmax && err > errmin) {
        for (j = 0; j < n; j++) {
            U[j] = 0.0f;
            for (i = 0; i < n; i++)
                U[j] += A[j*n + i]*P[i];
        }
        product = 0.0f;
        for (i = 0; i < n; i++)
            product += pow(U[i], 2);
        alpha = 1.0f/product;

        for (i = 0; i < n; i++) {
            x[i] += alpha*P[i];
            R[i] -= alpha*U[i];
        }

        for (j = 0; j < n; j++) {
            U[j] = 0.0f;
            for (i = 0; i < n; i++)
                U[j] += A[i*n + j]*R[i];
        }
        product = 0.0f;
        for (i = 0; i < n; i++)
            product += pow(U[i], 2);
        beta = 1.0f/product;

        for (i = 0; i < n; i++) {
            P[i] += beta*U[i];
        }

        err = 0.0;
        for (i = 0; i < n; i++) {
            e = abs(R[i]);
            if (e > err)
                err = e;
        }
        k++;
    }

    return k;
}
