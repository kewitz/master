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
// Kernel de pré-processamento responsável por calcular as matrizes de contribu-
// ição de todos os elementos.
//    ne: número de elementos.
//    elements: array de elementos da malha.
//    elements: array de nós da malha.
__global__ void kernel_integration(int ne, element *elements, float *S) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ne) return;

    element *E = &elements[i];

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

// Kernel de pré-processamento responsável por calcular diag_sum e right_sum.
//    ne: número de elementos.
//    elements: array de elementos da malha.
//    V: vetor de tensões dos nós.
//    dsum: vetor diag_sum.
//    rsum: vetor right_sum.
__global__ void kernel_preprocess(int ne, element *elements, float *V, float *S,
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
    float *R, float *P, float *V, float *S) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nn) return;

    node N = nodes[i];

    float ri = N.calc ? S[N.i] + rsum[N.i] - dsum[N.i]*V[N.i] : 0.0;
    R[N.i] = ri;
    P[N.i] = ri;
}

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

__global__ void kernel_iter_element_fix(int nn, node *nodes, float *U,
    float *P, float *S) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= nn) return;

    node N = nodes[i];
    if (!N.calc) {
        U[N.i] = P[N.i];
    }
    // else {
        // U[N.i] += S[N.i];
    // }
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
extern "C" int runGPU(int ng, int nn, int kmax, float errmin, group *groups,
    float *V, float *S, bool verbose, float *bench) {
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
    float *_dsum, *_rsum, *_V, *_S, *_U, *_P, *_R;
    group *G;
    node *_nodes;
    element *_elements;
    smalloc(&_dsum, s_V); smalloc(&_rsum, s_V);
    smalloc(&_elements, s_Elements); smalloc(&_nodes, s_Nodes);
    smalloc(&_sum1, sizeof(float)); smalloc(&_sum2, sizeof(float));
    smalloc(&_sum3, sizeof(float)); smalloc(&_sum4, sizeof(float));
    smalloc(&_S, s_V); smalloc(&_V, s_V); smalloc(&_U, s_V); smalloc(&_P, s_V);
    smalloc(&_R, s_V);

    smemcpy(_V, V, s_V, cudaMemcpyHostToDevice);

    kernel_util_zero<<<(1+nn/BSIZE), BSIZE>>>(nn, _dsum);
    kernel_util_zero<<<(1+nn/BSIZE), BSIZE>>>(nn, _rsum);
    kernel_util_zero<<<(1+nn/BSIZE), BSIZE>>>(nn, _S);
    cudaDeviceSynchronize();
    for (i = 0; i < ng; i++) {
        G = &groups[i];
        smemcpy(_elements, G->elements, sizeof(element)*G->ne,
            cudaMemcpyHostToDevice);
        kernel_integration<<<(1+G->ne/BSIZE), BSIZE>>>(G->ne, _elements, _S);
        cudaDeviceSynchronize();
        kernel_preprocess<<<(1+G->ne/BSIZE), BSIZE>>>(G->ne, _elements, _V, _S,
            _dsum, _rsum);
        cudaDeviceSynchronize();
    }
    for (i = 0; i < ng; i++) {
        G = &groups[i];
        smemcpy(_nodes, G->nodes, sizeof(node)*G->nn, cudaMemcpyHostToDevice);
        kernel_precond<<<(1+G->nn/BSIZE), BSIZE>>>(G->nn, _nodes, _dsum, _rsum,
            _R, _P, _V, _S);
        cudaDeviceSynchronize();
    }

    while (k < kmax && fabs(sqrt(sum3)) > errmin) {
        // U[] = 0
        kernel_util_zero<<<(1+nn/BSIZE), BSIZE>>>(nn, _U);
        kernel_util_zero<<<(1+nn/BSIZE), BSIZE>>>(nn, _S);
        cudaDeviceSynchronize();
        for (i = 0; i < ng; i++) {
            G = &groups[i];
            smemcpy(_elements, G->elements, sizeof(element)*G->ne,
                cudaMemcpyHostToDevice);
            kernel_integration<<<(1+G->ne/BSIZE), BSIZE>>>(G->ne, _elements,
                _rsum);
            cudaDeviceSynchronize();
            kernel_iter_element<<<(1+G->ne/BSIZE), BSIZE>>>(G->ne, _elements,
                _U, _P);
            cudaDeviceSynchronize();
        }
        for (i = 0; i < ng; i++) {
            G = &groups[i];
            smemcpy(_nodes, G->nodes, sizeof(node)*G->nn,
                cudaMemcpyHostToDevice);
            kernel_iter_element_fix<<<(1+G->nn/BSIZE), BSIZE>>>(G->nn, _nodes,
                _U, _P, _S);
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

    cudaFree(_V); cudaFree(_S); cudaFree(_U); cudaFree(_P); cudaFree(_R);
    cudaFree(_sum1); cudaFree(_sum2); cudaFree(_sum3); cudaFree(_sum4);
    cudaFree(_elements); cudaFree(_nodes);
    cudaFree(_dsum); cudaFree(_rsum);

    t = clock() - t;
    bench[0] = cast(float, t)/CLOCKS_PER_SEC;
    return k;
}
#endif

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
extern "C" int runCPU(int ng, int nn, int kmax, float errmin, group *groups,
    float *V, float *S, bool verbose, float *bench) {
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
        S[i] = 0.0f;
    }
    for (i = 0; i < ng; i++) {
        G = &groups[i];
        for (j = 0; j < G->ne; j++) {
            E = &G->elements[j];
            integ_element(E, S);

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
            ri = N.calc ? S[N.i] + rsum[N.i] - dsum[N.i]*V[N.i] : 0.0f;
            r[N.i] = ri;
            p[N.i] = ri;
        }
    }

    k = 1;
    while (k < kmax && fabs(sqrt(sum3)) > errmin) {
        for (i = 0; i < nn; i++) {
            u[i] = 0.0;
            S[i] = 0.0;
        }

        for (i = 0; i < ng; i++) {
            G = &groups[i];
            for (j = 0; j < G->ne; j++) {
                E = &G->elements[j];
                integ_element(E, S);
            }
            for (j = 0; j < G->ne; j++) {
                E = &G->elements[j];

                n1 = E->nodes[0]; n2 = E->nodes[1]; n3 = E->nodes[2];

                u[n1] += E->matriz[0]*p[n1] + E->matriz[3]*p[n2]
                         + E->matriz[4]*p[n3];
                u[n2] += E->matriz[3]*p[n1] + E->matriz[1]*p[n2]
                         + E->matriz[5]*p[n3];
                u[n3] += E->matriz[4]*p[n1] + E->matriz[5]*p[n2]
                         + E->matriz[2]*p[n3];
            }
        }

        for (i = 0; i < ng; i++) {
            G = &groups[i];
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

        alpha = sum1/sum2;
        assert(!isnan(alpha));
        for (i = 0; i < nn; i++) {
            V[i] += alpha*p[i];
            r[i] -= alpha*u[i];
        }

        sum3 = 0.0; sum4 = 0.0;
        for (i = 0; i < nn; i++) {
            sum3 += r[i]*r[i];
            sum4 += r[i]*u[i];
        }

        beta = -sum4/sum2;
        assert(!isnan(alpha));
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
    float alpha, beta, sum1, sum2, sum3 = 1.0f, sum4;
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

        sum1 = 0.0f; sum2 = 0.0f;
        for (i = 0; i < n; i++) {
            sum1 += p[i]*r[i];
            sum2 += p[i]*u[i];
        }

        alpha = sum2 != 0.0f ? sum1/sum2 : 0.0f;
        alpha = isnan(alpha) ? 0.0f : alpha;

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

#if CUDA
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
#endif
