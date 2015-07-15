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
#include "./nscheme.h"

// Constantes
#define DEBUG true
#define CUDA true
#define BSIZE 64
// Macros
#define cma(a, b, c, d, e) CudaSafeCall(cudaMemcpyAsync(a, b, c, d, e))


// Calcula espaço teórico máximo de nós e elementos que cabem na memória da GPU.
extern "C" unsigned int alloc(const int nn) {
    cudaDeviceProp prop = getInfo();
    unsigned int gm = prop.totalGlobalMem*.9 - sizeof(float)*nn*2;
    cudaDeviceReset();
    return cast(unsigned int, gm / (sizeof(node) + 6*sizeof(element)));
}

#if CUDA
// vec[i] = 0.0f
__global__ void kernel_util_zero(int nn, node *nodes, float *vec) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nn) return;
    vec[nodes[i].i] = 0.0f;
}

// Kernel de pre-processamento responsável por calcular as matrizes de contribu-
// ição de todos os elementos.
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

// Kernel responsável por uma iteração.
__global__ void kernel_node(int nn, float errmin, float R, element *elements,
    node *nodes, unsigned int *color, float *V, int *conv) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nn) return;

    node N = nodes[color[i]];

    int e, c;
    float diag_sum = 0.0f, right_sum = 0.0f, Vo = V[N.i], Vi, diff;
    element E;

    for (e = 0; e < N.ne; e++) {
        E = elements[N.elements[e]];
        if (N.i == E.nodes[0]) {
            diag_sum  += E.matriz[0];
            right_sum -= E.matriz[3]*V[E.nodes[1]];
            right_sum -= E.matriz[4]*V[E.nodes[2]];
        } else if (N.i == E.nodes[1]) {
            diag_sum += E.matriz[1];
            right_sum -= E.matriz[3]*V[E.nodes[0]];
            right_sum -= E.matriz[5]*V[E.nodes[2]];
        } else {
            diag_sum += E.matriz[2];
            right_sum -= E.matriz[4]*V[E.nodes[0]];
            right_sum -= E.matriz[5]*V[E.nodes[1]];
        }
    }

    Vi = right_sum/diag_sum;
    diff = Vi - Vo;
    Vi += R*diff;
    c = (abs(diff/Vi) > errmin);
    atomicOr(conv, c);
    V[N.i] = Vi;
}

// Função externa que processa o problema, responsável por alocar a memória no
// device e invocar todas os kernels necessários.
extern "C" int runGPU(int ng, int nn, int nc, int kmax, float R, float errmin,
    group *groups, float *V, bool verbose, float *bench) {
    // Inicia cronômetro do benchmark.
    unsigned int maxn = alloc(nn);
    cudaDeviceReset();
    clock_t t = clock();
    // Aloca variáveis.
    int k = 1, g, conv, *d_conv;
    float *d_V;
    group G;
    element *d_elements;
    node *d_nodes;
    unsigned int *d_colors;

    // Malloc e Memcpy de variáveis globais.
    smalloc(&d_V, sizeof(float)*nn);
    smalloc(&d_conv, sizeof(int));
    smalloc(&d_nodes, sizeof(node)*maxn);
    smalloc(&d_elements, sizeof(element)*maxn*6);
    smalloc(&d_colors, sizeof(unsigned int)*nn);
    smemcpy(d_V, V, sizeof(float)*nn, cudaMemcpyHostToDevice);

    // Iterações
    conv = 1;
    while (conv == 1 && k < kmax) {
        if (k%3)
            conv = 0;
        smemcpy(d_conv, &conv, sizeof(int), cudaMemcpyHostToDevice);
        for (g = 0; g < ng; g++) {
            cudaDeviceSynchronize();
            G = groups[g];
            if (ng > 1 || k == 1) {
                smemcpy(d_nodes, G.nodes, sizeof(node)*G.nn,
                    cudaMemcpyHostToDevice);
                smemcpy(d_elements, G.elements, sizeof(element)*G.ne,
                    cudaMemcpyHostToDevice);
                smemcpy(d_colors, G.cnodes, sizeof(unsigned int)*G.nn,
                    cudaMemcpyHostToDevice);
            }
            unsigned int *color = d_colors;
            kernel_element<<<(1 + G.ne/BSIZE), BSIZE>>>(G.ne, d_elements);
            cudaDeviceSynchronize();
            for (int c = 0; c < G.nc; c++) {
                kernel_node<<<(1 + G.colors[c]/BSIZE), BSIZE>>>(G.colors[c],
                    errmin, R, d_elements, d_nodes, color, d_V, d_conv);
                color += G.colors[c];
            }
        }
        cudaDeviceSynchronize();
        if (k%3)
            smemcpy(&conv, d_conv, sizeof(int), cudaMemcpyDeviceToHost);
        k++;
    }

    smemcpy(V, d_V, sizeof(float)*nn, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_elements); cudaFree(d_nodes);
    cudaFree(d_V); cudaFree(d_conv);

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
    // Calcula a matriz de contribuições do elemento.
    E->matriz[0] = (q1*q1 + r1*r1)*cof;
    E->matriz[1] = (q2*q2 + r2*r2)*cof;
    E->matriz[2] = (q3*q3 + r3*r3)*cof;
    E->matriz[3] = (q1*q2 + r1*r2)*cof;
    E->matriz[4] = (q1*q3 + r1*r3)*cof;
    E->matriz[5] = (q2*q3 + r2*r3)*cof;
}

void calc_node(node N, float errmin, float R, float *V, element *elements,
    bool *run) {
    int e;
    float diag_sum = 0.0f, right_sum = 0.0f, Vo = V[N.i], Vi, diff;
    element E;

    for (e = 0; e < N.ne; e++) {
        E = elements[N.elements[e]];
        if (N.i == E.nodes[0]) {
            diag_sum  += E.matriz[0];
            right_sum -= E.matriz[3]*V[E.nodes[1]];
            right_sum -= E.matriz[4]*V[E.nodes[2]];
        }
        if (N.i == E.nodes[1]) {
            diag_sum += E.matriz[1];
            right_sum -= E.matriz[3]*V[E.nodes[0]];
            right_sum -= E.matriz[5]*V[E.nodes[2]];
        }
        if (N.i == E.nodes[2]) {
            diag_sum += E.matriz[2];
            right_sum -= E.matriz[4]*V[E.nodes[0]];
            right_sum -= E.matriz[5]*V[E.nodes[1]];
        }
    }

    Vi = right_sum/diag_sum;
    diff = Vi - Vo;
    Vi += R*diff;
    *run |= (fabs(diff/Vi) > errmin);
    V[N.i] = Vi;
}

extern "C" int runCPU(int ng, int nn, int nc, int kmax, float R, float errmin,
    group *groups, float *V, bool verbose, float *bench) {
    // Aloca variáveis.
    int i, j, k = 1;

    // Loop principal das iterações.
    bool run = true;
    while (run && k < kmax) {
        run = false;
        // Loop de grupo emulado.
        for (i = 0; i < ng; i++) {
            group G = groups[i];
            // Integra os elementos do Grupo.
            for (j = 0; j < G.ne; j++)
                integ_element(&G.elements[j]);
            unsigned int offset = 0;
            for (int l = 0; l < G.nc; l++) {
                // Calcula os potenciais nos nós do Grupo.
                for (j = 0; j < G.colors[l]; j++)
                    calc_node(G.nodes[G.cnodes[j+offset]], errmin, R, V,
                        G.elements, &run);
                offset += G.colors[l];
            }
        }
        k++;
    }

    return k;
}

extern "C" void test_group(int ng, group *groups) {
    unsigned int i;

    for (i = 0; i < ng; i++) {
        group G = groups[i];
        printf("Group %i has %i nodes and %i elements.\n", i, G.nn, G.ne);
        printf("Nodes: %p\t Elements: %p\n\n", G.nodes, G.elements);
    }
}
