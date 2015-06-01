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

#define BSIZE 512
#define smalloc(a, b) CudaSafeCall(cudaMalloc(a, b))
#define smemcpy(a, b, c, d) CudaSafeCall(cudaMemcpy(a, b, c, d))
#define putf(a, b) smemcpy(a, b, sizeof(float), cudaMemcpyHostToDevice);
#define getf(a, b) smemcpy(a, b, sizeof(float), cudaMemcpyDeviceToHost);
#define cast(t, v) static_cast<t>(v)

// Kernel de pré-processamento responsável por calcular as matrizes de contribu-
// ição de todos os elementos.
//    ne: número de elementos.
//    elements: array de elementos da malha.
//    elements: array de nós da malha.
__global__ void kernel_integration(int ne, elementri *elements, node *nodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ne) return;

    elementri E = elements[i];
    node N1 = nodes[E.nodes[0]], N2 = nodes[E.nodes[1]], N3 = nodes[E.nodes[2]];

    // Calcula argumentos necessários
    float J1, J2, J3, J4, dJ;
    J1 = N2.x - N1.x;
    J2 = N2.y - N1.y;
    J3 = N3.x - N1.x;
    J4 = N3.y - N1.y;
    dJ = 2*(J1*J4 - J3*J2);

    // Calcula a matriz de contribuições do elemento.
    elements[i].matriz[0] = dJ != 0.0 ?
        (powf(J2-J4, 2.0) + powf(J3-J1, 2.0))/dJ : 0.0;
    elements[i].matriz[1] = dJ != 0.0 ?
        (powf(J4, 2.0) + powf(J3, 2.0))/dJ : 0.0;
    elements[i].matriz[2] = dJ != 0.0 ?
        (powf(J2, 2.0) + powf(J1, 2.0))/dJ : 0.0;
    elements[i].matriz[3] = dJ != 0.0 ?
        ((J2-J4)*J4 - (J3-J1)*J3)/dJ : 0.0;
    elements[i].matriz[4] = dJ != 0.0 ?
        ((J3-J1)*J1 - (J2-J4)*J2)/dJ : 0.0;
    elements[i].matriz[5] = dJ != 0.0 ?
        (J4*-1*J2 - J3*J1)/dJ : 0.0;
}

// Kernel de pré-processamento responsável por calcular diag_sum e right_sum.
//    ne: número de elementos.
//    elements: array de elementos da malha.
//    V: vetor de tensões dos nós.
//    dsum: vetor diag_sum.
//    rsum: vetor right_sum.
__global__ void kernel_preprocess(int ne, elementri * elements, float * V,
                                  float * dsum, float * rsum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ne) return;

    int n1, n2, n3;
    elementri E = elements[i];
    n1 = E.nodes[0]; n2 = E.nodes[1]; n3 = E.nodes[2];

    atomicAdd(&dsum[n1], E.matriz[0]);
    atomicAdd(&dsum[n2], E.matriz[1]);
    atomicAdd(&dsum[n3], E.matriz[2]);

    atomicAdd(&rsum[n1], -E.matriz[3]*V[n2] -E.matriz[4]*V[n3]);
    atomicAdd(&rsum[n2], -E.matriz[3]*V[n1] -E.matriz[5]*V[n3]);
    atomicAdd(&rsum[n3], -E.matriz[4]*V[n1] -E.matriz[5]*V[n2]);
}

// Kernel de pré-condicionamento.
__global__ void kernel_precond(int nn, node * nodes, float * dsum, float * rsum,
                               float * R, float * P, float * V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nn) return;

    float ri = nodes[i].calc ? rsum[i] - dsum[i]*V[i] : 0.0;
    R[i] = ri;
    P[i] = ri;
}

__global__ void kernel_iter_element(int ne, elementri * elements, float * u,
                                    float * p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ne) return;

    elementri E = elements[i];
    int n1 = E.nodes[0], n2 = E.nodes[1], n3 = E.nodes[2];

    atomicAdd(&u[n1], E.matriz[0]*p[n1] + E.matriz[3]*p[n2] +
              E.matriz[4]*p[n3]);
    atomicAdd(&u[n2], E.matriz[3]*p[n1] + E.matriz[1]*p[n2] +
              E.matriz[5]*p[n3]);
    atomicAdd(&u[n3], E.matriz[4]*p[n1] + E.matriz[5]*p[n2] +
              E.matriz[2]*p[n3]);
}

__global__ void kernel_iter_element_fix(int nn, node *nodes, float *u,
    float *p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nn) return;

    if (!nodes[i].calc)
        u[i] = p[i];
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
//    ne: número de elementos.
//    nn: número de nós.
//    kmax: número máximo de iterações.
//    errmin: erro mínimo para considerar a convergência do resultado.
//    elements: array de elementos da malha.
//    nodes: array de nós da malha.
//    V: vetor de tensões dos nós.
//    verbose: se 'true' imprime informações do algorítmo.
//    bench: array de tempos de processamento para benchmarking.
extern "C" int runGPU(int ne, int nn, int kmax, float errmin,
                      elementri *elements, node *nodes, float *V, bool verbose,
                      float *bench) {
    clock_t t = clock();
    int k = 1;
    const dim3 threads(BSIZE);
    const dim3 elementblocks(1 + ne/BSIZE);
    const dim3 nodeblocks(1 + nn/BSIZE);

    // Array Sizes
    size_t s_Elements = sizeof(elementri)*ne,
           s_Nodes = sizeof(node)*nn,
           s_V = sizeof(float)*nn;

    // Scalars.
    float sum1 = 0.0f, sum2 = 0.0f, sum3 = 1.0f, sum4 = 0.0f, alpha = 0.0f,
          beta = 0.0f, *_sum1, *_sum2, *_sum3, *_sum4;

    // Device Arrays.
    float *_dsum, *_rsum, *_V, *_U, *_P, *_R;
    node *_nodes;
    elementri *_elements;
    smalloc(&_dsum, s_V); smalloc(&_rsum, s_V);
    smalloc(&_elements, s_Elements); smalloc(&_nodes, s_Nodes);
    smalloc(&_sum1, sizeof(float)); smalloc(&_sum2, sizeof(float));
    smalloc(&_sum3, sizeof(float)); smalloc(&_sum4, sizeof(float));
    smalloc(&_V, s_V); smalloc(&_U, s_V); smalloc(&_P, s_V); smalloc(&_R, s_V);

    smemcpy(_elements, elements, s_Elements, cudaMemcpyHostToDevice);
    smemcpy(_V, V, s_V, cudaMemcpyHostToDevice);
    smemcpy(_nodes, nodes, s_Nodes, cudaMemcpyHostToDevice);

    kernel_integration<<<elementblocks, threads>>>(ne, _elements, _nodes);
    kernel_util_zero<<<nodeblocks, threads>>>(nn, _dsum);
    kernel_util_zero<<<nodeblocks, threads>>>(nn, _rsum);
    cudaDeviceSynchronize();
    kernel_preprocess<<<elementblocks, threads>>>(ne, _elements, _V, _dsum,
        _rsum);
    cudaDeviceSynchronize();
    kernel_precond<<<nodeblocks, threads>>>(nn, _nodes, _dsum, _rsum, _R, _P,
        _V);

    while (k < kmax && fabs(sqrt(sum3)) > errmin) {
        // U[] = 0
        kernel_util_zero<<<nodeblocks, threads>>>(nn, _U);
        cudaDeviceSynchronize();
        kernel_iter_element<<<elementblocks, threads>>>(ne, _elements, _U, _P);
        cudaDeviceSynchronize();
        kernel_iter_element_fix<<<nodeblocks, threads>>>(nn, _nodes, _U, _P);
        cudaDeviceSynchronize();

        sum1 = 0.0f; sum2 = 0.0f;
        putf(_sum1, &sum1); putf(_sum2, &sum2);
        kernel_util_vecsummult<<<nodeblocks, threads>>>(nn, _P, _R, _sum1);
        kernel_util_vecsummult<<<nodeblocks, threads>>>(nn, _P, _U, _sum2);
        cudaDeviceSynchronize();
        getf(&sum1, _sum1); getf(&sum2, _sum2);

        alpha = sum2 != 0.0 ? sum1/sum2 : 0.0;
        kernel_util_addtovec<<<nodeblocks, threads>>>(nn, alpha, _V, _P);
        kernel_util_addtovec<<<nodeblocks, threads>>>(nn, -alpha, _R, _U);
        cudaDeviceSynchronize();

        sum3 = 0.0f; sum4 = 0.0f;
        putf(_sum3, &sum3); putf(_sum4, &sum4);
        kernel_util_vecsummult<<<nodeblocks, threads>>>(nn, _R, _R, _sum3);
        kernel_util_vecsummult<<<nodeblocks, threads>>>(nn, _R, _U, _sum4);
        cudaDeviceSynchronize();
        getf(&sum3, _sum3); getf(&sum4, _sum4);

        beta = sum2 != 0.0 ? -sum4/sum2 : 0.0;
        kernel_util_addtovec2<<<nodeblocks, threads>>>(nn, beta, _P, _R, _P);
        cudaDeviceSynchronize();

        k++;
    }
    smemcpy(V, _V, s_V, cudaMemcpyDeviceToHost);

    cudaFree(_sum1); cudaFree(_sum2); cudaFree(_sum3); cudaFree(_sum4);
    cudaFree(_V); cudaFree(_U); cudaFree(_P); cudaFree(_R);
    cudaFree(_elements); cudaFree(_nodes);
    cudaFree(_dsum); cudaFree(_rsum);

    t = clock() - t;
    bench[0] = cast(float, t)/CLOCKS_PER_SEC;
    return k;
}

// Função externa que processa o problema no CPU.
//    ne: número de elementos.
//    nn: número de nós.
//    kmax: número máximo de iterações.
//    errmin: erro mínimo para considerar a convergência do resultado.
//    elements: array de elementos da malha.
//    nodes: array de nós da malha.
//    V: vetor de tensões dos nós.
//    verbose: se 'true' imprime informações do algorítmo.
//    bench: array de tempos de processamento para benchmarking.
extern "C" int runCPU(int ne, int nn, int kmax, float errmin,
                      elementri *elements, node *nodes, float *V, bool verbose,
                      float *bench) {
    clock_t t = clock();
    int i, k;
    // Pre-processamento. Calcula as matrizes de contribuição dos elementos.
    float J1, J2, J3, J4, dJ;
    elementri E;
    node N1, N2, N3;
    for (i = 0; i < ne; i++) {
        E = elements[i];
        N1 = nodes[E.nodes[0]]; N2 = nodes[E.nodes[1]]; N3 = nodes[E.nodes[2]];

        // Calcula argumentos necessários
        J1 = N2.x - N1.x;
        J2 = N2.y - N1.y;
        J3 = N3.x - N1.x;
        J4 = N3.y - N1.y;
        dJ = 2*(J1*J4 - J3*J2);

        // Calcula a matriz de contribuições do elemento.
        elements[i].matriz[0] = dJ != 0.0 ?
            (pow(J2-J4, 2.0) + pow(J3-J1, 2.0))/dJ : 0.0;
        elements[i].matriz[1] = dJ != 0.0 ?
            (pow(J4, 2.0) + pow(J3, 2.0))/dJ : 0.0;
        elements[i].matriz[2] = dJ != 0.0 ?
            (pow(J2, 2.0) + pow(J1, 2.0))/dJ : 0.0;
        elements[i].matriz[3] = dJ != 0.0 ?
            ((J2-J4)*J4 - (J3-J1)*J3)/dJ : 0.0;
        elements[i].matriz[4] = dJ != 0.0 ?
            ((J3-J1)*J1 - (J2-J4)*J2)/dJ : 0.0;
        elements[i].matriz[5] = dJ != 0.0 ?
            (J4*-1*J2 - J3*J1)/dJ : 0.0;
    }

    // Pre-processamento. Calcula dsum e rsum.
    int n1, n2, n3;
    float *rsum = cast(float*, malloc(nn*sizeof(float)));
    float *dsum = cast(float*, malloc(nn*sizeof(float)));
    // Inicialização dos vetores.
    for (i = 0; i < nn; i++) {
        rsum[i] = 0.0;
        dsum[i] = 0.0;
    }
    for (i = 0; i < ne; i++) {
        E = elements[i];
        n1 = E.nodes[0]; n2 = E.nodes[1]; n3 = E.nodes[2];

        dsum[n1] += E.matriz[0];
        dsum[n2] += E.matriz[1];
        dsum[n3] += E.matriz[2];

        rsum[n1] += - E.matriz[3]*V[n2] - E.matriz[4]*V[n3];
        rsum[n2] += - E.matriz[3]*V[n1] - E.matriz[5]*V[n3];
        rsum[n3] += - E.matriz[4]*V[n1] - E.matriz[5]*V[n2];
    }

    // CG
    float ri, alpha, beta, sum1, sum2, sum3 = 1, sum4;
    float *r = cast(float*, malloc(nn*sizeof(float)));
    float *p = cast(float*, malloc(nn*sizeof(float)));
    float *u = cast(float*, malloc(nn*sizeof(float)));

    // Pré-condicionamento.
    for (i = 0; i < nn; i++) {
        ri = nodes[i].calc ? rsum[i] - dsum[i]*V[i] : 0.0;
        p[i] = ri;
        r[i] = ri;
    }

    k = 1;
    while (k < kmax && fabs(sqrt(sum3)) > errmin) {
        for (i = 0; i < nn; i++)
            u[i] = 0.0;

        for (i = 0; i < ne; i++) {
            E = elements[i];
            n1 = E.nodes[0]; n2 = E.nodes[1]; n3 = E.nodes[2];

            u[n1] += E.matriz[0]*p[n1] + E.matriz[3]*p[n2] + E.matriz[4]*p[n3];
            u[n2] += E.matriz[3]*p[n1] + E.matriz[1]*p[n2] + E.matriz[5]*p[n3];
            u[n3] += E.matriz[4]*p[n1] + E.matriz[5]*p[n2] + E.matriz[2]*p[n3];
        }

        for (i = 0; i < nn; i++)
            if (!nodes[i].calc)
                u[i] = p[i];

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

        beta = sum2 != 0.0 ? -sum4/sum2 : 0.0;
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

// Sadiku's Numerical Techniques in Electromagnetics. pg.712
extern "C" int testeCG(int n, int kmax, float err, float* A, float* x,
                       float* b) {
    int i, j, k = 1;
    float alpha, beta, sum1, sum2, sum3 = 1, sum4;
    float *r = cast(float*, malloc(n*sizeof(float)));
    float *p = cast(float*, malloc(n*sizeof(float)));
    float *u = cast(float*, malloc(n*sizeof(float)));

    for (i = 0; i < n; i++) {
        p[i] = b[i];
        r[i] = b[i];
    }

    while (k < kmax && fabs(sqrt(sum3)) > err) {
        for (j = 0; j < n; j++) {
            u[j] = 0.0;
            for (i = 0; i < n; i++)
                u[j] += A[i*n + j]*p[i];
        }

        sum1 = 0.0; sum2 = 0.0;
        for (i = 0; i < n; i++) {
            sum1 += p[i]*r[i];
            sum2 += p[i]*u[i];
        }

        alpha = sum2 != 0.0 ? sum1/sum2 : 0.0;

        for (i = 0; i < n; i++) {
            x[i] += alpha*p[i];
            r[i] -= alpha*u[i];
        }

        sum3 = 0.0; sum4 = 0.0;
        for (i = 0; i < n; i++) {
            sum3 += r[i]*r[i];
            sum4 += r[i]*u[i];
        }

        beta = sum2 != 0.0 ? -sum4/sum2 : 0.0;

        for (i = 0; i < n; i++) {
            p[i] = r[i] + beta*p[i];
        }

        k++;
    }

    free(r);
    free(p);
    free(u);

    return k;
}

extern "C" float teste_sum_reduction(int size, float *a, float *b) {
    const dim3 threads(BSIZE);
    const dim3 blocks(1 + size/BSIZE);

    float *_a, *_b, *_sum, sum = 0;
    smalloc(&_a, sizeof(float)*size);
    smalloc(&_b, sizeof(float)*size);
    smalloc(&_sum, sizeof(float));

    smemcpy(_a, a, sizeof(float)*size, cudaMemcpyHostToDevice);
    smemcpy(_b, b, sizeof(float)*size, cudaMemcpyHostToDevice);
    smemcpy(_sum, &sum, sizeof(float), cudaMemcpyHostToDevice);

    kernel_util_vecsummult<<<blocks, threads>>>(size, _a, _b, _sum);
    cudaDeviceSynchronize();
    smemcpy(&sum, _sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    return sum;
}
