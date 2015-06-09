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
#include "./nscheme.h"

#define DEBUG false
#define BSIZE 128

#define cma(a, b, c, d, e) CudaSafeCall(cudaMemcpyAsync(a, b, c, d, e))


// Kernel responsável por uma iteração.
__global__ void kernel_node(const int nn, const float R, const float errmin,
    elementri *elements, node *nodes, float *V, int *conv) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nn) return;

    node Node = nodes[i];

    int e, c;
    float diag_sum = 0.0, right_sum = 0.0, Vo = V[Node.i], Vi, diff;
    elementri Element;

    for (e = 0; e < Node.ne; e++) {
        Element = elements[Node.elements[e]];
        if (Node.i == Element.nodes[0]) {
            diag_sum  += Element.matriz[0];
            right_sum -= Element.matriz[3]*V[Element.nodes[1]];
            right_sum -= Element.matriz[4]*V[Element.nodes[2]];
        }
        if (Node.i == Element.nodes[1]) {
            diag_sum += Element.matriz[1];
            right_sum -= Element.matriz[3]*V[Element.nodes[0]];
            right_sum -= Element.matriz[5]*V[Element.nodes[2]];
        }
        if (Node.i == Element.nodes[2]) {
            diag_sum += Element.matriz[2];
            right_sum -= Element.matriz[4]*V[Element.nodes[0]];
            right_sum -= Element.matriz[5]*V[Element.nodes[1]];
        }
    }

    Vi = diag_sum == 0 ? 0.0f : fdividef(right_sum, diag_sum);
    diff = Vi - Vo;
    Vi += R*diff;
    c = fabs(diff) > errmin*fabs(Vi);
    atomicOr(conv, c);
    V[Node.i] = Vi;
}

// Kernel de pre-processamento responsável por calcular as matrizes de contribu-
// ição de todos os elementos.
__global__ void kernel_element(int ne, elementri *elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ne) return;

    elementri *E = &elements[i];

    // Calcula argumentos necessários
    float J1, J2, J3, J4, dJ;
    J1 = E->x[1] - E->x[0];
    J2 = E->y[1] - E->y[0];
    J3 = E->x[2] - E->x[0];
    J4 = E->y[2] - E->y[0];
    dJ = 2*(J1*J4 - J3*J2);

    // Calcula a matriz de contribuições do elemento.
    E->matriz[0] = (pow(J2-J4, 2) + pow(J3-J1, 2))/dJ;   // C11
    E->matriz[1] = (pow(J4, 2) + pow(J3, 2))/dJ;         // C22
    E->matriz[2] = (pow(J2, 2) + pow(J1, 2))/dJ;         // C33
    E->matriz[3] = ((J2-J4)*J4 - (J3-J1)*J3)/dJ;       // C12 C21
    E->matriz[4] = ((J2-J4)*-1*J2 + (J3-J1)*J1)/dJ;    // C13 C31
    E->matriz[5] = (J4*-1*J2 - J3*J1)/dJ;              // C23 C32
}

// Função externa que processa o problema, responsável por alocar a memória no
// device e invocar todas os kernels necessários.
extern "C" int runGPU(const int ng, const int nn, const int kmax, const float R,
    const float errmin, group *groups, float *V, bool verbose, float *bench) {
    // Inicia cronômetro do benchmark.
    clock_t t = clock();
    // Aloca variáveis.
    int k = 1, g, conv, *d_conv;
    float *d_V;
    group G;
    elementri *d_elements;
    node *d_nodes;

    // Malloc e Memcpy de variáveis globais.
    smalloc(&d_V, sizeof(float)*nn);
    smalloc(&d_conv, sizeof(int));
    smemcpy(d_V, V, sizeof(float)*nn, cudaMemcpyHostToDevice);

    // Iterações
    conv = 1;
    while (conv == 1 && k < kmax) {
        conv = 0;
        smemcpy(d_conv, &conv, sizeof(int), cudaMemcpyHostToDevice);
        for (g = 0; g < ng; g++) {
            cudaDeviceSynchronize();
            G = groups[g];
            // Malloc
            smalloc(&d_elements, sizeof(elementri)*G.ne);
            smalloc(&d_nodes, sizeof(node)*G.nn);
            // Memcpy e processamento dos elementos.
            smemcpy(d_elements, G.elements, sizeof(elementri)*G.ne,
                cudaMemcpyHostToDevice);
            kernel_element<<<(1 + G.ne/BSIZE), BSIZE>>>(G.ne, d_elements);
            // Memcpy dos nós enquanto se processa os elementos.
            smemcpy(d_nodes, G.nodes, sizeof(node)*G.nn,
                cudaMemcpyHostToDevice);
            kernel_node<<<(1 + G.nn/BSIZE), BSIZE>>>(G.nn, R, errmin,
                d_elements, d_nodes, d_V, d_conv);
            cudaDeviceSynchronize();
            cudaFree(d_elements);
            cudaFree(d_nodes);
        }
        cudaDeviceSynchronize();
        smemcpy(&conv, d_conv, sizeof(int), cudaMemcpyDeviceToHost);
        k++;
    }

    smemcpy(V, d_V, sizeof(float)*nn, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_V); cudaFree(d_conv);

    t = clock() - t;
    bench[0] = cast(float, t)/CLOCKS_PER_SEC;
    return k;
}

extern "C" int streamGPU(const int ng, const int nn, const int kmax, const float R,
    const float errmin, group *groups, float *V, bool verbose, float *bench) {
    // Inicia cronômetro do benchmark.
    clock_t t = clock();
    // Aloca variáveis.
    int k = 1, g, conv, *d_conv;
    float *d_V;
    group G;
    elementri *d_elements;
    node *d_nodes;

    // Cria streams.
    cudaStream_t s[2];
    for (int i = 0; i < 2; ++i)
        cudaStreamCreate(&s[i]);

    // Malloc e Memcpy de variáveis globais.
    smalloc(&d_V, sizeof(float)*nn);
    smalloc(&d_conv, sizeof(int));
    cma(d_V, V, sizeof(float)*nn, cudaMemcpyHostToDevice, s[0]);

    // Iterações
    conv = 1;
    while (conv == 1 && k < kmax) {
        conv = 0;
        cma(d_conv, &conv, sizeof(int), cudaMemcpyHostToDevice, s[0]);
        for (g = 0; g < ng; g++) {
            G = groups[g];
            // Malloc
            smalloc(&d_elements, sizeof(elementri)*G.ne);
            smalloc(&d_nodes, sizeof(node)*G.nn);
            // Memcpy e processamento dos elementos.
            cma(d_elements, G.elements, sizeof(elementri)*G.ne,
                cudaMemcpyHostToDevice, s[0]);
            // Memcpy dos nós enquanto se processa os elementos.
            cma(d_nodes, G.nodes, sizeof(node)*G.nn,
                cudaMemcpyHostToDevice, s[1]);
            kernel_element<<<(1 + G.ne/BSIZE), BSIZE, 0, s[0]>>>(G.ne,
                d_elements);
            kernel_node<<<(1 + G.nn/BSIZE), BSIZE, 0, s[0]>>>(G.nn, R, errmin,
                d_elements, d_nodes, d_V, d_conv);
            cudaDeviceSynchronize();
            cudaFree(d_elements);
            cudaFree(d_nodes);
        }
        cudaDeviceSynchronize();
        smemcpy(&conv, d_conv, sizeof(int), cudaMemcpyDeviceToHost);
        k++;
    }

    smemcpy(V, d_V, sizeof(float)*nn, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (int i = 0; i < 2; ++i)
        cudaStreamDestroy(s[i]);

    cudaFree(d_V); cudaFree(d_conv);

    t = clock() - t;
    bench[0] = cast(float, t)/CLOCKS_PER_SEC;
    return k;
}

void integ_element(elementri *E) {
    float J1, J2, J3, J4, dJ;
    J1 = E->x[1] - E->x[0];
    J2 = E->y[1] - E->y[0];
    J3 = E->x[2] - E->x[0];
    J4 = E->y[2] - E->y[0];
    dJ = 2*(J1*J4 - J3*J2);

    // Calcula a matriz de contribuições do elemento.
    E->matriz[0] = (pow(J2-J4, 2) + pow(J3-J1, 2))/dJ;   // C11
    E->matriz[1] = (pow(J4, 2) + pow(J3, 2))/dJ;         // C22
    E->matriz[2] = (pow(J2, 2) + pow(J1, 2))/dJ;         // C33
    E->matriz[3] = ((J2-J4)*J4 - (J3-J1)*J3)/dJ;         // C12 C21
    E->matriz[4] = ((J2-J4)*-1*J2 + (J3-J1)*J1)/dJ;      // C13 C31
    E->matriz[5] = (J4*-1*J2 - J3*J1)/dJ;                // C23 C32
}

void calc_node(const node N, const float errmin, const float R, float *V,
    elementri *elements, bool *run) {
    int e;
    float diag_sum = 0.0f, right_sum = 0.0f, Vi, Vo, diff;
    elementri E;
    Vo = V[N.i];

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

    Vi = diag_sum == 0 ? 0.0 : right_sum/diag_sum;
    diff = Vi - Vo;
    Vi += R*diff;
    *run |= (fabs(diff) > errmin*fabs(Vi));
    V[N.i] = Vi;
}

extern "C" int runCPU(const int ng, const int nn, const int kmax, const float R,
    const float errmin, group *groups, float *V, bool verbose, float *bench) {
    // Inicia cronômetro do benchmark.
    clock_t t = clock();
    // Aloca variáveis.
    int i, j, k = 1;

    // Loop principal das iterações.
    bool run = true;
    while (run && k < kmax) {
        run = false;
        // Loop de grupo emulado.
        for (i = 0; i < ng; i++) {
            group G = groups[i];
            // Integra os elementos do Grupo
            for (j = 0; j < G.ne; j++)
                integ_element(&G.elements[j]);
            // Calcula os potenciais nos nós do Grupo.
            for (j = 0; j < G.nn; j++)
                calc_node(G.nodes[j], errmin, R, V, G.elements, &run);
        }
        k++;
    }

    t = clock() - t;
    bench[0] = cast(float, t)/CLOCKS_PER_SEC;
    return k;
}

extern "C" void test_group(const int ng, group *groups) {
    unsigned int i;

    for (i = 0; i < ng; i++) {
        group G = groups[i];
        printf("Group %i has %i nodes and %i elements.\n", i, G.nn, G.ne);
        printf("Nodes: %p\t Elements: %p\n\n", G.nodes, G.elements);
    }
}
