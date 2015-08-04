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
#include <assert.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "./cuda_snippets.h"
#include "./escheme.h"

#define BSIZE 512
#define putf(a, b) smemcpy(a, b, sizeof(float), cudaMemcpyHostToDevice);
#define getf(a, b) smemcpy(a, b, sizeof(float), cudaMemcpyDeviceToHost);
#define CUDA true

extern "C" unsigned int alloc(const int nn) {
    cudaDeviceProp prop = getInfo();
    unsigned int gm = prop.totalGlobalMem*.8 - sizeof(float)*nn*6
                      - sizeof(float)*4;
    cudaDeviceReset();
    return cast(unsigned int, gm / (sizeof(node) + 6*sizeof(element)));
}

#if CUDA
// Kernel de responsável por calcular as matrizes de contribuição de um
// elemento.
__global__ void kernel_element(int ne, element *elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ne) return;

    element E = elements[i];

    // Calcula gradN
    float q1 = E.y[1]-E.y[2], q2 = E.y[2]-E.y[0], q3 = E.y[0]-E.y[1];
    float r1 = E.x[2]-E.x[1], r2 = E.x[0]-E.x[2], r3 = E.x[1]-E.x[0];
    // Calcula det(gradN)
    float det = E.x[1]*E.y[2] + E.x[0]*E.y[1] + E.x[2]*E.y[0]
              - E.x[0]*E.y[2] - E.x[2]*E.y[1] - E.x[1]*E.y[0];
    float cof = (E.mat/det)/2;
    // Calcula a matriz de contribuições do elemento.
    elements[i].matriz[0] = (q1*q1 + r1*r1)*cof;
    elements[i].matriz[1] = (q2*q2 + r2*r2)*cof;
    elements[i].matriz[2] = (q3*q3 + r3*r3)*cof;
    elements[i].matriz[3] = (q1*q2 + r1*r2)*cof;
    elements[i].matriz[4] = (q1*q3 + r1*r3)*cof;
    elements[i].matriz[5] = (q2*q3 + r2*r3)*cof;
}

// Kernel de pré-processamento responsável por calcular diag_sum e right_sum.
__global__ void kernel_preprocess(int ne, element *elements, float *V,
    float *dsum, float *rsum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ne) return;

    int n1, n2, n3;
    element E = elements[i];
    n1 = E.nodes[0]; n2 = E.nodes[1]; n3 = E.nodes[2];

    atomicAdd(&dsum[n1], E.matriz[0]);
    atomicAdd(&dsum[n2], E.matriz[1]);
    atomicAdd(&dsum[n3], E.matriz[2]);

    atomicAdd(&rsum[n1], - E.matriz[3]*V[n2] - E.matriz[4]*V[n3]);
    atomicAdd(&rsum[n2], - E.matriz[3]*V[n1] - E.matriz[5]*V[n3]);
    atomicAdd(&rsum[n3], - E.matriz[4]*V[n1] - E.matriz[5]*V[n2]);
}

// Kernel de pré-condicionamento.
__global__ void kernel_precond(int nn, node *nodes, float *dsum, float *rsum,
    float *R, float *P, float *V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nn) return;

    node N = nodes[i];

    float ri = N.calc ? rsum[N.i] - dsum[N.i]*V[N.i] : 0.0;
    R[N.i] = ri;
    P[N.i] = ri;
}

// U = SS*P
__global__ void kernel_iter_element(int ne, element *elements, float *U,
    float *P) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ne) return;

    element E = elements[i];
    int n1 = E.nodes[0], n2 = E.nodes[1], n3 = E.nodes[2];

    atomicAdd(&U[n1], E.matriz[0]*P[n1] + E.matriz[3]*P[n2] +
              E.matriz[4]*P[n3]);
    atomicAdd(&U[n2], E.matriz[3]*P[n1] + E.matriz[1]*P[n2] +
              E.matriz[5]*P[n3]);
    atomicAdd(&U[n3], E.matriz[4]*P[n1] + E.matriz[5]*P[n2] +
              E.matriz[2]*P[n3]);
}

// Corrige os valores de U para nós submetidos à condição de contorno.
__global__ void kernel_iter_element_fix(int nn, node *nodes, float *U,
    float *P) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= nn) return;

    node N = nodes[i];
    if (!N.calc) {
        U[N.i] = P[N.i];
    }
}

// vec[i] = 0.0f
__global__ void kernel_util_zero(int size, float *vec) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    vec[i] = 0.0;
}

// sum += vecA[i]*vecB[i]
__global__ void kernel_util_vecsummult(int size, float *vecA, float *vecB,
    float *sum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ti = threadIdx.x;
    __shared__ float _sum[BSIZE];


    _sum[ti] = (i < size) ? vecA[i]*vecB[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
       if (ti < s)
           _sum[ti] += _sum[ti + s];

       __syncthreads();
    }
    if (ti == 0) {
        atomicAdd(sum, _sum[0]);
    }
}

// vecA[i] += scalar * vecB[i]
__global__ void kernel_util_addtovec(int size, const float scalar, float *vecA,
    float *vecB) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    vecA[i] += scalar*vecB[i];
}

// vecA[i] = vecB[i] + scalar*vecC[i]
__global__ void kernel_util_addtovec2(int size, const float scalar, float *vecA,
    float *vecB, float *vecC) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    vecA[i] = vecB[i] + scalar*vecC[i];
}

// Função externa que processa o problema na GPU.
extern "C" int runGPU(int ng, int nn, int kmax, float errmin, group *groups,
    float *V, bool verbose, float *bench) {
    clock_t t = clock();
    int i, k = 1;
    unsigned int maxn = alloc(nn);

    // Array Sizes
    size_t s_Elements = sizeof(element)*maxn*6,
           s_Nodes = sizeof(node)*maxn,
           s_V = sizeof(float)*nn;

    // Scalars.
    float sum1 = 0.0f, sum2 = 0.0f, sum3 = 1.0f, sum4 = 0.0f, alpha = 0.0f,
          beta = 0.0f, *_sum1, *_sum2, *_sum3, *_sum4;

    // Device Arrays.
    float *_dsum, *_rsum, *_V, *_U, *_P, *_R;
    group *G;
    node *_nodes;
    element *_elements;
    smalloc(&_dsum, s_V); smalloc(&_rsum, s_V);
    smalloc(&_elements, s_Elements); smalloc(&_nodes, s_Nodes);
    smalloc(&_sum1, sizeof(float)); smalloc(&_sum2, sizeof(float));
    smalloc(&_sum3, sizeof(float)); smalloc(&_sum4, sizeof(float));
    smalloc(&_V, s_V); smalloc(&_U, s_V); smalloc(&_P, s_V); smalloc(&_R, s_V);

    smemcpy(_V, V, s_V, cudaMemcpyHostToDevice);

    kernel_util_zero<<<(1+nn/BSIZE), BSIZE>>>(nn, _dsum);
    kernel_util_zero<<<(1+nn/BSIZE), BSIZE>>>(nn, _rsum);
    cudaDeviceSynchronize();
    for (i = 0; i < ng; i++) {
        G = &groups[i];
        smemcpy(_elements, G->elements, sizeof(element)*G->ne,
            cudaMemcpyHostToDevice);
        kernel_element<<<(1+G->ne/BSIZE), BSIZE>>>(G->ne, _elements);
        cudaDeviceSynchronize();
        kernel_preprocess<<<(1+G->ne/BSIZE), BSIZE>>>(G->ne, _elements, _V,
            _dsum, _rsum);
        cudaDeviceSynchronize();
    }
    for (i = 0; i < ng; i++) {
        G = &groups[i];
        smemcpy(_nodes, G->nodes, sizeof(node)*G->nn, cudaMemcpyHostToDevice);
        kernel_precond<<<(1+G->nn/BSIZE), BSIZE>>>(G->nn, _nodes, _dsum, _rsum,
            _R, _P, _V);
        cudaDeviceSynchronize();
    }

    while (k < kmax && fabs(sqrt(sum3)) > errmin) {
        // U[] = 0
        kernel_util_zero<<<(1+nn/BSIZE), BSIZE>>>(nn, _U);
        cudaDeviceSynchronize();
        for (i = 0; i < ng; i++) {
            G = &groups[i];
            if (ng > 1)
                smemcpy(_elements, G->elements, sizeof(element)*G->ne,
                    cudaMemcpyHostToDevice);
            kernel_element<<<(1+G->ne/BSIZE), BSIZE>>>(G->ne, _elements);
            cudaDeviceSynchronize();
            kernel_iter_element<<<(1+G->ne/BSIZE), BSIZE>>>(G->ne, _elements,
                _U, _P);
            cudaDeviceSynchronize();
        }
        for (i = 0; i < ng; i++) {
            G = &groups[i];
            if (ng > 1)
                smemcpy(_nodes, G->nodes, sizeof(node)*G->nn,
                    cudaMemcpyHostToDevice);
            kernel_iter_element_fix<<<(1+G->nn/BSIZE), BSIZE>>>(G->nn, _nodes,
                _U, _P);
            cudaDeviceSynchronize();
        }

        sum1 = 0.0f; sum2 = 0.0f;
        putf(_sum1, &sum1); putf(_sum2, &sum2);
        kernel_util_vecsummult<<<(1+nn/BSIZE), BSIZE>>>(nn, _P, _R, _sum1);
        kernel_util_vecsummult<<<(1+nn/BSIZE), BSIZE>>>(nn, _P, _U, _sum2);
        cudaDeviceSynchronize();
        getf(&sum1, _sum1); getf(&sum2, _sum2);

        alpha = sum2 != 0.0 ? sum1/sum2 : 0.0;
        kernel_util_addtovec<<<(1+nn/BSIZE), BSIZE>>>(nn, alpha, _V, _P);
        kernel_util_addtovec<<<(1+nn/BSIZE), BSIZE>>>(nn, -alpha, _R, _U);
        cudaDeviceSynchronize();

        sum3 = 0.0f; sum4 = 0.0f;
        putf(_sum3, &sum3); putf(_sum4, &sum4);
        kernel_util_vecsummult<<<(1+nn/BSIZE), BSIZE>>>(nn, _R, _R, _sum3);
        kernel_util_vecsummult<<<(1+nn/BSIZE), BSIZE>>>(nn, _R, _U, _sum4);
        cudaDeviceSynchronize();
        getf(&sum3, _sum3); getf(&sum4, _sum4);

        beta = sum2 != 0.0 ? -sum4/sum2 : 0.0;
        kernel_util_addtovec2<<<(1+nn/BSIZE), BSIZE>>>(nn, beta, _P, _R, _P);
        cudaDeviceSynchronize();

        k++;
    }

    smemcpy(V, _V, s_V, cudaMemcpyDeviceToHost);

    cudaFree(_V); cudaFree(_U); cudaFree(_P); cudaFree(_R);
    cudaFree(_sum1); cudaFree(_sum2); cudaFree(_sum3); cudaFree(_sum4);
    cudaFree(_elements); cudaFree(_nodes);
    cudaFree(_dsum); cudaFree(_rsum);

    t = clock() - t;
    bench[0] = cast(float, t)/CLOCKS_PER_SEC;
    return k;
}
#endif

void integ_element(element *E) {
    float mat = E->mat;
    // Calcula gradN
    float q1 = E->y[1]-E->y[2], q2 = E->y[2]-E->y[0], q3 = E->y[0]-E->y[1];
    float r1 = E->x[2]-E->x[1], r2 = E->x[0]-E->x[2], r3 = E->x[1]-E->x[0];
    // Calcula det(gradN)
    float det = E->x[1]*E->y[2] + E->x[0]*E->y[1] + E->x[2]*E->y[0]
              - E->x[0]*E->y[2] - E->x[2]*E->y[1] - E->x[1]*E->y[0];
    float cof = (mat/det)/2.0;
    assert(!isnan(det));
    assert(!isnan(cof));
    // Calcula a matriz de contribuições do elemento.
    E->matriz[0] = (powf(q1, 2.0) + powf(r1, 2.0))*cof;
    E->matriz[1] = (powf(q2, 2.0) + powf(r2, 2.0))*cof;
    E->matriz[2] = (powf(q3, 2.0) + powf(r3, 2.0))*cof;
    E->matriz[3] = (q1*q2 + r1*r2)*cof;
    E->matriz[4] = (q1*q3 + r1*r3)*cof;
    E->matriz[5] = (q2*q3 + r2*r3)*cof;
}
// Função externa que processa o problema no CPU.
extern "C" int runCPU(int ng, int nn, int kmax, float errmin, group *groups,
    float *V, bool verbose, float *bench) {
    clock_t t = clock();
    unsigned int i, j, k;
    element *E;
    group *G;
    node N;

    // Pre-processamento. Calcula dsum e rsum.
    int n1, n2, n3;
    float *rsum = cast(float*, malloc(nn*sizeof(float)));
    float *dsum = cast(float*, malloc(nn*sizeof(float)));
    // Inicialização dos vetores.
    for (i = 0; i < nn; i++) {
        rsum[i] = 0.0f;
        dsum[i] = 0.0f;
    }
    for (i = 0; i < ng; i++) {
        G = &groups[i];
        for (j = 0; j < G->ne; j++) {
            E = &G->elements[j];
            integ_element(E);

            n1 = E->nodes[0]; n2 = E->nodes[1]; n3 = E->nodes[2];

            dsum[n1] += E->matriz[0];
            dsum[n2] += E->matriz[1];
            dsum[n3] += E->matriz[2];

            rsum[n1] += -E->matriz[3]*V[n2] -E->matriz[4]*V[n3];
            rsum[n2] += -E->matriz[3]*V[n1] -E->matriz[5]*V[n3];
            rsum[n3] += -E->matriz[4]*V[n1] -E->matriz[5]*V[n2];
        }
    }

    // CG
    float ri, alpha, beta, sum1, sum2, sum3 = 1.0f, sum4;
    float *r = cast(float*, malloc(nn*sizeof(float)));
    float *p = cast(float*, malloc(nn*sizeof(float)));
    float *u = cast(float*, malloc(nn*sizeof(float)));

    // Pré-condicionamento.
    for (i = 0; i < ng; i++) {
        G = &groups[i];
        for (j = 0; j < G->nn; j++) {
            N = G->nodes[j];
            ri = N.calc ? rsum[N.i] - dsum[N.i]*V[N.i] : 0.0f;
            r[N.i] = ri;
            p[N.i] = ri;
        }
    }

    k = 1;
    while (k < kmax && fabs(sqrt(sum3)) > errmin) {
        for (i = 0; i < nn; i++) {
            u[i] = 0.0;
        }

        for (i = 0; i < ng; i++) {
            G = &groups[i];
            for (j = 0; j < G->ne; j++) {
                E = &G->elements[j];
                integ_element(E);

                n1 = E->nodes[0]; n2 = E->nodes[1]; n3 = E->nodes[2];
                u[n1] += E->matriz[0]*p[n1] + E->matriz[3]*p[n2]
                         + E->matriz[4]*p[n3];
                u[n2] += E->matriz[3]*p[n1] + E->matriz[1]*p[n2]
                         + E->matriz[5]*p[n3];
                u[n3] += E->matriz[4]*p[n1] + E->matriz[5]*p[n2]
                         + E->matriz[2]*p[n3];
            }
            for (j = 0; j < G->nn; j++) {
                N = G->nodes[j];
                if (!N.calc)
                    u[N.i] = p[N.i];
            }
        }

        sum1 = 0.0; sum2 = 0.0;
        for (i = 0; i < nn; i++) {
            sum1 += p[i]*r[i];
            sum2 += p[i]*u[i];
        }

        alpha = sum2 != 0.0 ? sum1/sum2 : 0.0;
        for (i = 0; i < nn; i++) {
            V[i] += alpha*p[i];
            r[i] -= alpha*u[i];
        }

        sum3 = 0.0; sum4 = 0.0;
        for (i = 0; i < nn; i++) {
            sum3 += r[i]*r[i];
            sum4 += r[i]*u[i];
        }

        beta = sum2 != 0.0 ? -sum4/sum2 : 0.0f;
        for (i = 0; i < nn; i++) {
            p[i] = r[i] + beta*p[i];
        }

        k++;
    }

    free(dsum);
    free(rsum);
    free(r);
    free(p);
    free(u);

    t = clock() - t;
    bench[0] = cast(float, t)/CLOCKS_PER_SEC;
    return k;
}
