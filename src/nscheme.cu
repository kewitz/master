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
#define BSIZE 256
// Macros
#define cma(a, b, c, d, e) CudaSafeCall(cudaMemcpyAsync(a, b, c, d, e))

// vec[i] = 0.0f
__global__ void kernel_util_zero(int nn, float *vec) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nn) return;

    vec[i] = 0.0f;
}

// Kernel responsável por uma iteração.
__global__ void kernel_node(int nn, float R, float errmin, elementri *elements,
    node *nodes, float *V, float *S, int *conv) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nn) return;

    node N = nodes[i];

    int e, c;
    float diag_sum = 0.0, right_sum = S[N.i], Vo = V[N.i], Vi, diff;
    elementri Element;

    for (e = 0; e < N.ne; e++) {
        Element = elements[N.elements[e]];
        if (N.i == Element.nodes[0]) {
            diag_sum  += Element.matriz[0];
            right_sum -= Element.matriz[3]*V[Element.nodes[1]];
            right_sum -= Element.matriz[4]*V[Element.nodes[2]];
        }
        if (N.i == Element.nodes[1]) {
            diag_sum += Element.matriz[1];
            right_sum -= Element.matriz[3]*V[Element.nodes[0]];
            right_sum -= Element.matriz[5]*V[Element.nodes[2]];
        }
        if (N.i == Element.nodes[2]) {
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
    V[N.i] = Vi;
}

// Kernel de pre-processamento responsável por calcular as matrizes de contribu-
// ição de todos os elementos.
__global__ void kernel_element(int ne, elementri *elements, float *S) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ne) return;

    elementri *E = &elements[i];
    float mat = E->mat;
    float f = E->f;

    // Calcula argumentos necessários
    float J1, J2, J3, J4, dJ;
    J1 = E->x[1] - E->x[0];
    J2 = E->y[1] - E->y[0];
    J3 = E->x[2] - E->x[0];
    J4 = E->y[2] - E->y[0];
    dJ = 2*(J1*J4 - J3*J2);

    // Calcula a matriz de contribuições do elemento.
    E->matriz[0] = mat*(pow(J2-J4, 2) + pow(J3-J1, 2))/dJ;   // C11
    E->matriz[1] = mat*(pow(J4, 2) + pow(J3, 2))/dJ;         // C22
    E->matriz[2] = mat*(pow(J2, 2) + pow(J1, 2))/dJ;         // C33
    E->matriz[3] = mat*((J2-J4)*J4 - (J3-J1)*J3)/dJ;       // C12 C21
    E->matriz[4] = mat*((J2-J4)*-1*J2 + (J3-J1)*J1)/dJ;    // C13 C31
    E->matriz[5] = mat*(J4*-1*J2 - J3*J1)/dJ;              // C23 C32


    if (f > 0.0f) {
        f = f*dJ*0.5f*0.333333f;
        S[E->nodes[0]] += f;
        S[E->nodes[1]] += f;
        S[E->nodes[2]] += f;
    }
}

// Calcula espaço teórico máximo de nós e elementos que cabem na memória da GPU.
extern "C" unsigned int alloc(const int nn) {
    cudaDeviceProp prop = getInfo();
    unsigned int gm = prop.totalGlobalMem*.9 - sizeof(float)*nn*2;
    cudaDeviceReset();
    return cast(unsigned int, gm / (sizeof(node) + 6*sizeof(elementri)));
}

// Função externa que processa o problema, responsável por alocar a memória no
// device e invocar todas os kernels necessários.
extern "C" int runGPU(int ng, int nn, int kmax, float R, float errmin,
    group *groups, float *V, float *S, bool verbose, float *bench) {
    // Inicia cronômetro do benchmark.
    clock_t t = clock();
    cudaDeviceReset();
    // Aloca variáveis.
    int k = 1, g, conv, *d_conv;
    float *d_V, *d_S;
    group G;
    elementri *d_elements;
    node *d_nodes;

    unsigned int maxn = alloc(nn);

    // Malloc e Memcpy de variáveis globais.
    smalloc(&d_V, sizeof(float)*nn);
    smalloc(&d_S, sizeof(float)*nn);
    smalloc(&d_conv, sizeof(int));
    smalloc(&d_nodes, sizeof(node)*maxn);
    smalloc(&d_elements, sizeof(elementri)*maxn*6);
    smemcpy(d_V, V, sizeof(float)*nn, cudaMemcpyHostToDevice);
    smemcpy(d_S, S, sizeof(float)*nn, cudaMemcpyHostToDevice);

    // Iterações
    conv = 1;
    while (conv == 1 && k < kmax) {
        conv = 0;
        smemcpy(d_conv, &conv, sizeof(int), cudaMemcpyHostToDevice);
        for (g = 0; g < ng; g++) {
            cudaDeviceSynchronize();
            G = groups[g];
            // Zera vetor S.
            kernel_util_zero<<<(1 + nn/BSIZE), BSIZE>>>(nn, d_S);
            // Memcpy e processamento dos elementos.
            smemcpy(d_elements, G.elements, sizeof(elementri)*G.ne,
                cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            kernel_element<<<(1 + G.ne/BSIZE), BSIZE>>>(G.ne, d_elements, d_S);
            // Memcpy dos nós enquanto se processa os elementos.
            smemcpy(d_nodes, G.nodes, sizeof(node)*G.nn,
                cudaMemcpyHostToDevice);
            kernel_node<<<(1 + G.nn/BSIZE), BSIZE>>>(G.nn, R, errmin,
                d_elements, d_nodes, d_V, d_S, d_conv);
        }
        cudaDeviceSynchronize();
        smemcpy(&conv, d_conv, sizeof(int), cudaMemcpyDeviceToHost);
        k++;
    }

    smemcpy(V, d_V, sizeof(float)*nn, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_elements); cudaFree(d_nodes);
    cudaFree(d_V); cudaFree(d_S); cudaFree(d_conv);

    t = clock() - t;
    bench[0] = cast(float, t)/CLOCKS_PER_SEC;
    return k;
}



extern "C" int streamGPU(int ng, int nn, int kmax, float R, float errmin,
    group *groups, float *V, float *S, bool verbose, float *bench) {
    // Inicia cronômetro do benchmark.
    clock_t t = clock();
    cudaDeviceReset();
    // Aloca variáveis.
    int k = 1, g, conv, *d_conv;
    bool copye = true;
    float *d_V, *d_S;
    elementri *d_elements, *pe;
    node *d_nodes;
    group *G;

    unsigned int maxn = alloc(nn);
    size_t sN = sizeof(node)*maxn;
    size_t sE = sizeof(elementri)*maxn*6;

    // Malloc e Memcpy de variáveis globais.
    smalloc(&d_V, sizeof(float)*nn);
    smalloc(&d_S, sizeof(float)*nn);
    smalloc(&d_conv, sizeof(int));
    smalloc(&d_nodes, sN);
    smalloc(&d_elements, sE);
    // Inicia a cópia do vetor V.
    smemcpy(d_V, V, sizeof(float)*nn, cudaMemcpyHostToDevice);

    // Cria streams.
    cudaStream_t stream[2];
    for (int i = 0; i < 2; ++i)
        cudaStreamCreate(&stream[i]);

    // Define ponteiros temporários para streaming.
    pe = d_elements;

    // Iterações
    conv = 1;
    while (conv == 1 && k < kmax) {
        conv = 0;
        smemcpy(d_conv, &conv, sizeof(int), cudaMemcpyHostToDevice);

        for (g = 0; g < ng; g++) {
            G = &groups[g];
            if (copye) {
                pe = d_elements;
                cma(pe, G->elements, sizeof(elementri)*G->ne,
                    cudaMemcpyHostToDevice, stream[0]);
            }
            kernel_util_zero<<<(1 + nn/BSIZE), BSIZE, 0, stream[0]>>>(nn, d_S);
            cma(d_nodes, G->nodes, sizeof(node)*G->nn,
                cudaMemcpyHostToDevice, stream[1]);
            kernel_element<<<(1 + G->ne/BSIZE), BSIZE, 0, stream[0]>>>(G->ne,
                pe, d_S);
            cudaDeviceSynchronize();

            kernel_node<<<(1 + G->nn/BSIZE), BSIZE, 0, stream[0]>>>(G->nn, R,
                errmin, pe, d_nodes, d_V, d_S, d_conv);
            // Se não for o último grupo, já copia novos elementos.
            if (g < ng-1 && sE - sizeof(elementri)*G->ne >
                sizeof(elementri)*groups[g+1].ne) {
                pe += G->ne;
                cma(pe, groups[g+1].elements, sizeof(elementri)*groups[g+1].ne,
                    cudaMemcpyHostToDevice, stream[1]);
                copye = false;
            } else {
                copye = true;
            }
            cudaDeviceSynchronize();
        }
        cudaDeviceSynchronize();
        smemcpy(&conv, d_conv, sizeof(int), cudaMemcpyDeviceToHost);
        k++;
    }

    smemcpy(V, d_V, sizeof(float)*nn, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (int i = 0; i < 2; ++i)
        cudaStreamDestroy(stream[i]);

    cudaFree(d_V); cudaFree(d_S); cudaFree(d_conv);
    cudaFree(d_elements); cudaFree(d_nodes);

    t = clock() - t;
    bench[0] = cast(float, t)/CLOCKS_PER_SEC;

    return k;
}

void integ_element(elementri *E, float *S) {
    float mat = E->mat, f = E->f;
    // Calcula [J]
    float J[2][2];
    J[0][0] = E->x[1] - E->x[0];
    J[0][1] = E->y[1] - E->y[0];
    J[1][0] = E->x[2] - E->x[0];
    J[1][1] = E->y[2] - E->y[0];
    // Calcula |J|
    float dJ = J[0][0]*J[1][1] - J[1][0]*J[1][1];
    assert(!isnan(dJ));
    // Calcula [J]^-1
    float iJ[2][2];
    iJ[0][0] = J[1][1]/dJ;
    iJ[0][1] = -J[0][1]/dJ;
    iJ[1][0] = -J[1][0]/dJ;
    iJ[1][1] = J[0][0]/dJ;
    if isnan(iJ[0][0]) iJ[0][0] = 0.0f;
    if isnan(iJ[0][1]) iJ[0][1] = 0.0f;
    if isnan(iJ[1][0]) iJ[1][0] = 0.0f;
    if isnan(iJ[1][1]) iJ[1][1] = 0.0f;
    // Calcula [J]^-1.GradN
    float q1 = -iJ[0][0]+iJ[0][1], q2 = iJ[0][0], q3 = iJ[0][1];
    float r1 = -iJ[1][0]+iJ[1][1], r2 = iJ[1][0], r3 = iJ[1][1];
    assert(!isnan(q1) && !isnan(q2) && !isnan(q3));
    assert(!isnan(r1) && !isnan(r2) && !isnan(r3));
    // Calcula a matriz de contribuições do elemento.
    E->matriz[0] = (q1*q1 + r1*r1)*mat/(dJ*2);
    E->matriz[1] = (q2*q2 + r2*r2)*mat/(dJ*2);
    E->matriz[2] = (q3*q3 + r3*r3)*mat/(dJ*2);
    E->matriz[3] = (q1*q2 + r1*r2)*mat/(dJ*2);
    E->matriz[4] = (q1*q3 + r1*r3)*mat/(dJ*2);
    E->matriz[5] = (q2*q3 + r2*r3)*mat/(dJ*2);

    if (fabs(f) > 0) {
        f = f*dJ*0.5f*0.333333f;
        S[E->nodes[0]] += f;
        S[E->nodes[1]] += f;
        S[E->nodes[2]] += f;
    }
}

void calc_node(node N, float errmin, float R, float *V, float *S,
    elementri *elements, bool *run) {
    int e;
    float diag_sum = 0.0f, right_sum = S[N.i], Vo = V[N.i], Vi, diff;
    elementri E;

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

    Vi = diag_sum == 0.0 ? 0.0 : right_sum/diag_sum;
    diff = Vi - Vo;
    Vi += R*diff;
    *run |= (fabs(diff/Vi) > errmin);
    V[N.i] = Vi;
}

extern "C" int runCPU(int ng, int nn, int kmax, float R, float errmin,
    group *groups, float *V, float *S, bool verbose, float *bench) {
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
            // Zera vetor S para integração.
            for (j = 0; j < nn; j++)
                S[j] = 0.0f;
            // Integra os elementos do Grupo.
            for (j = 0; j < G.ne; j++)
                integ_element(&G.elements[j], S);
            // Calcula os potenciais nos nós do Grupo.
            for (j = 0; j < G.nn; j++)
                calc_node(G.nodes[j], errmin, R, V, S, G.elements, &run);
        }
        k++;
    }

    t = clock() - t;
    bench[0] = cast(float, t)/CLOCKS_PER_SEC;
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
