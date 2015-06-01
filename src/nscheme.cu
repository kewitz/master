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
#define smalloc(a, b) CudaSafeCall(cudaMalloc(a, b))
#define smemcpy(a, b, c, d) CudaSafeCall(cudaMemcpy(a, b, c, d))
#define cast(t, v) static_cast<t>(v)

// Kernel responsável por uma iteração.
__global__ void kernel_iter(const int nn, const int k, const int cn,
    const float R, const float errmin, elementri *elements, node *nodes,
    color *colors, float *V, int *conv) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    color co = colors[cn];

    if (i >= co.len) return;

    #if DEBUG
    if (i == 0 && k == 1) {
        printf("\nColor %i:\n", cn);
        for (int n = 0; n < co.len; n++) {
            printf("%i, ", co.nodes[n]);
        }
    }
    #endif

    node Node = nodes[co.nodes[i]];
    if (Node.calc == false) return;

    int e, c;
    float diag_sum = 0.0, right_sum = 0.0, Vi, diff, Vo;
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
    Vo = V[Node.i];
    diff = Vi - Vo;
    Vi += R*diff;
    c = fabs(diff) > errmin*fabs(Vi);
    atomicOr(conv, c);
    V[Node.i] = Vi;
}

// Kernel de pre-processamento responsável por calcular as matrizes de contribu-
// ição de todos os elementos.
__global__ void kernel_pre(int ne, elementri *elements, node *nodes) {
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
    elements[i].matriz[0] = (pow(J2-J4, 2) + pow(J3-J1, 2))/dJ;   // C11
    elements[i].matriz[1] = (pow(J4, 2) + pow(J3, 2))/dJ;         // C22
    elements[i].matriz[2] = (pow(J2, 2) + pow(J1, 2))/dJ;         // C33
    elements[i].matriz[3] = ((J2-J4)*J4 - (J3-J1)*J3)/dJ;       // C12 C21
    elements[i].matriz[4] = ((J2-J4)*-1*J2 + (J3-J1)*J1)/dJ;    // C13 C31
    elements[i].matriz[5] = (J4*-1*J2 - J3*J1)/dJ;              // C23 C32
}

// Função externa que processa o problema, responsável por alocar a memória no
// device e invocar todas os kernels necessários.
extern "C" int runGPU(const int ne, const int nn, const int nc, const int kmax,
    const float R, const float errmin, elementri *elements, node *nodes,
    color *colors, float *V, bool verbose, float *bench) {
    clock_t t = clock();
    int k = 1, conv, *d_conv;
    unsigned int c;
    float *d_V;
    node *d_nodes;
    elementri *d_elements;
    color *d_colors;
    const dim3 threads(BSIZE);
    const dim3 preblocks(1 + ne/BSIZE);
    const dim3 iterblocks(1 + nn/BSIZE);
    size_t s_Elements = sizeof(elementri)*ne,
           s_Nodes = sizeof(node)*nn,
           s_V = sizeof(float)*nn;

    // Malloc
    smalloc(&d_elements, s_Elements);
    smalloc(&d_nodes, s_Nodes);
    smalloc(&d_V, s_V);
    smalloc(&d_conv, sizeof(int));
    // Memcpy
    smemcpy(d_elements, elements, s_Elements, cudaMemcpyHostToDevice);
    smemcpy(d_nodes, nodes, s_Nodes, cudaMemcpyHostToDevice);
    smemcpy(d_V, V, s_V, cudaMemcpyHostToDevice);
    // Malloc e Memcpy do Coloring.
    smalloc(&d_colors, sizeof(color)*nc);
    for (c = 0; c < nc; c++) {
        color co = colors[c];
        unsigned int *d_cnodes;
        smalloc(&d_cnodes, sizeof(unsigned int)*co.len);
        smemcpy(d_cnodes, co.nodes, sizeof(unsigned int)*co.len,
            cudaMemcpyHostToDevice);
        co.nodes = d_cnodes;
        smemcpy(d_colors+c, &co, sizeof(color), cudaMemcpyHostToDevice);
    }

    // Pre-processamento
    kernel_pre<<<preblocks, threads>>>(ne, d_elements, d_nodes);
    cudaDeviceSynchronize();

    // Iterações
    conv = 1;
    while (conv == 1 && k < kmax) {
        conv = 0;
        cudaMemcpy(d_conv, &conv, sizeof(int), cudaMemcpyHostToDevice);
        for (c = 1; c < nc; c++) {
            color co = colors[c];
            kernel_iter<<<(1 + co.len/BSIZE), threads>>>(nn, k, c, R, errmin,
                d_elements, d_nodes, d_colors, d_V, d_conv);
            cudaDeviceSynchronize();
        }
        cudaMemcpy(&conv, d_conv, sizeof(int), cudaMemcpyDeviceToHost);
        k++;
    }

    CudaSafeCall(cudaMemcpy(V, d_V, s_V, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    cudaFree(d_V); cudaFree(d_nodes); cudaFree(d_elements); cudaFree(d_conv);

    t -= clock();
    bench[0] = cast(float, t)/CLOCKS_PER_SEC;
    return k;
}

// Função externa que processa o problema no CPU.
extern "C" int runCPU(const int ne, const int nn, const int nc, const int kmax,
    const float R, const float errmin, elementri *elements, node *nodes,
    color *colors, float *V, bool verbose, float *bench) {
    clock_t t = clock();
    int i, k = 1;

    // Pre-processamento. Calcula as matrizes de contribuição dos elementos.
    for (i = 0; i < ne; i++) {
        elementri E = elements[i];
        node N1 = nodes[E.nodes[0]], N2 = nodes[E.nodes[1]],
             N3 = nodes[E.nodes[2]];

        // Calcula argumentos necessários
        float J1, J2, J3, J4, dJ;
        J1 = N2.x - N1.x;
        J2 = N2.y - N1.y;
        J3 = N3.x - N1.x;
        J4 = N3.y - N1.y;
        dJ = 2*(J1*J4 - J3*J2);

        // Calcula a matriz de contribuições do elemento.
        elements[i].matriz[0] = (pow(J2-J4, 2) + pow(J3-J1, 2))/dJ;   // C11
        elements[i].matriz[1] = (pow(J4, 2) + pow(J3, 2))/dJ;         // C22
        elements[i].matriz[2] = (pow(J2, 2) + pow(J1, 2))/dJ;         // C33
        elements[i].matriz[3] = ((J2-J4)*J4 - (J3-J1)*J3)/dJ;       // C12 C21
        elements[i].matriz[4] = ((J2-J4)*-1*J2 + (J3-J1)*J1)/dJ;    // C13 C31
        elements[i].matriz[5] = (J4*-1*J2 - J3*J1)/dJ;              // C23 C32
    }

    float diff;
    bool run = true;
    while (run && k < kmax) {
        run = false;
        for (i = 0; i < nn; i++) {
            int e;
            float diag_sum = 0.0f, right_sum = 0.0f, Vi, Vo;
            node Node = nodes[i];
            if (Node.calc == true) {
                Vo = V[Node.i];
                elementri E;
                for (e = 0; e < Node.ne; e++) {
                    E = elements[Node.elements[e]];
                    if (Node.i == E.nodes[0]) {
                        diag_sum  += E.matriz[0];
                        right_sum -= E.matriz[3]*V[E.nodes[1]];
                        right_sum -= E.matriz[4]*V[E.nodes[2]];
                    }
                    if (Node.i == E.nodes[1]) {
                        diag_sum += E.matriz[1];
                        right_sum -= E.matriz[3]*V[E.nodes[0]];
                        right_sum -= E.matriz[5]*V[E.nodes[2]];
                    }
                    if (Node.i == E.nodes[2]) {
                        diag_sum += E.matriz[2];
                        right_sum -= E.matriz[4]*V[E.nodes[0]];
                        right_sum -= E.matriz[5]*V[E.nodes[1]];
                    }
                }
                Vi = diag_sum == 0 ? 0.0 : right_sum/diag_sum;
                diff = Vi - Vo;
                Vi += R*diff;
                run |= fabs(diff) > errmin*fabs(Vi);
                V[Node.i] = Vi;
            }
        }
        k++;
    }

    t -= clock();
    bench[0] = cast(float, t)/CLOCKS_PER_SEC;
    return k;
}

extern "C" int runCPUColor(const int ne, const int nn, const int nc,
    const int kmax, const float R, const float errmin, elementri *elements,
    node *nodes, color *colors, float *V, bool verbose, float *bench) {
    clock_t t = clock();
    int i, k = 1;
    unsigned int c;

    // Pre-processamento. Calcula as matrizes de contribuição dos elementos.
    for (i = 0; i < ne; i++) {
        elementri E = elements[i];
        node N1 = nodes[E.nodes[0]], N2 = nodes[E.nodes[1]],
             N3 = nodes[E.nodes[2]];

        // Calcula argumentos necessários
        float J1, J2, J3, J4, dJ;
        J1 = N2.x - N1.x;
        J2 = N2.y - N1.y;
        J3 = N3.x - N1.x;
        J4 = N3.y - N1.y;
        dJ = 2*(J1*J4 - J3*J2);

        // Calcula a matriz de contribuições do elemento.
        elements[i].matriz[0] = (pow(J2-J4, 2) + pow(J3-J1, 2))/dJ;   // C11
        elements[i].matriz[1] = (pow(J4, 2) + pow(J3, 2))/dJ;         // C22
        elements[i].matriz[2] = (pow(J2, 2) + pow(J1, 2))/dJ;         // C33
        elements[i].matriz[3] = ((J2-J4)*J4 - (J3-J1)*J3)/dJ;       // C12 C21
        elements[i].matriz[4] = ((J2-J4)*-1*J2 + (J3-J1)*J1)/dJ;    // C13 C31
        elements[i].matriz[5] = (J4*-1*J2 - J3*J1)/dJ;              // C23 C32
    }

    float diff;
    bool run = true;
    while (run && k < kmax) {
        run = false;
        for (c = 0; c < nc; c++) {
            color co = colors[c];
            for (i = 0; i < co.len; i++) {
                int e;
                float diag_sum = 0.0f, right_sum = 0.0f, Vi, Vo;
                node Node = nodes[co.nodes[i]];
                if (Node.calc == true) {
                    Vo = V[Node.i];
                    elementri E;
                    for (e = 0; e < Node.ne; e++) {
                        E = elements[Node.elements[e]];
                        if (Node.i == E.nodes[0]) {
                            diag_sum  += E.matriz[0];
                            right_sum -= E.matriz[3]*V[E.nodes[1]];
                            right_sum -= E.matriz[4]*V[E.nodes[2]];
                        }
                        if (Node.i == E.nodes[1]) {
                            diag_sum += E.matriz[1];
                            right_sum -= E.matriz[3]*V[E.nodes[0]];
                            right_sum -= E.matriz[5]*V[E.nodes[2]];
                        }
                        if (Node.i == E.nodes[2]) {
                            diag_sum += E.matriz[2];
                            right_sum -= E.matriz[4]*V[E.nodes[0]];
                            right_sum -= E.matriz[5]*V[E.nodes[1]];
                        }
                    }
                    Vi = diag_sum == 0 ? 0.0 : right_sum/diag_sum;
                    diff = Vi - Vo;
                    Vi += R*diff;
                    run |= fabs(diff) > errmin*fabs(Vi);
                    V[Node.i] = Vi;
                }
            }
        }
        k++;
    }

    t -= clock();
    bench[0] = cast(float, t)/CLOCKS_PER_SEC;
    return k;
}

extern "C" void test_colors(int numcolors, color *colors) {
    for (unsigned int i = 0; i < numcolors; i++) {
        color c = colors[i];
        printf("Color %i have %i nodes:\n", i, c.len);
        for (unsigned int j = 0; j < c.len; j++) {
            printf(" %i,", c.nodes[j]);
        }
        printf("\n\n");
    }
}

__global__ void kernel_iter_test(const int n, const float R, const float errmin,
    elementri *elements, node *nodes, float *V, int *conv) {

    node Node = nodes[n];
    if (Node.calc == false) return;

    int e, c;
    float diag_sum = 0.0, right_sum = 0.0, Vi, diff, Vo;
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
    Vo = V[Node.i];
    diff = Vi - Vo;
    Vi += R*diff;
    c = fabs(diff) > errmin*fabs(Vi);
    atomicOr(conv, c);
    V[Node.i] = Vi;
}

extern "C" int test_gpu(const int ne, const int nn, const int nc,
    const int kmax, const float R, const float errmin, elementri *elements,
    node *nodes, color *colors, float *V, bool verbose, float *bench) {

    clock_t t = clock();
    int k = 1, conv, *d_conv;
    float *d_V;
    node *d_nodes;
    elementri *d_elements;
    const dim3 threads(512);
    const dim3 preblocks(1 + ne/512);
    const dim3 iterblocks(1 + nn/512);
    size_t s_Elements = sizeof(elementri)*ne,
           s_Nodes = sizeof(node)*nn,
           s_V = sizeof(float)*nn;

    // Malloc
    smalloc(&d_elements, s_Elements);
    smalloc(&d_nodes, s_Nodes);
    smalloc(&d_V, s_V);
    smalloc(&d_conv, sizeof(int));
    // Memcpy
    smemcpy(d_elements, elements, s_Elements, cudaMemcpyHostToDevice);
    smemcpy(d_nodes, nodes, s_Nodes, cudaMemcpyHostToDevice);
    smemcpy(d_V, V, s_V, cudaMemcpyHostToDevice);

    // Pre-processamento
    kernel_pre<<<preblocks, threads>>>(ne, d_elements, d_nodes);
    cudaDeviceSynchronize();

    // Iterações
    conv = 1;
    while (conv == 1 && k < kmax) {
        conv = 0;
        cudaMemcpy(d_conv, &conv, sizeof(int), cudaMemcpyHostToDevice);
        for (int i = 0; i < nn; i++) {
            kernel_iter_test<<<1, 1>>>(i, R, errmin, d_elements, d_nodes,
                d_V, d_conv);
            cudaDeviceSynchronize();
        }
        cudaMemcpy(&conv, d_conv, sizeof(int), cudaMemcpyDeviceToHost);
        k++;
    }

    CudaSafeCall(cudaMemcpy(V, d_V, s_V, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    cudaFree(d_V); cudaFree(d_nodes); cudaFree(d_elements); cudaFree(d_conv);

    t -= clock();
    bench[0] = cast(float, t)/CLOCKS_PER_SEC;
    return k;
}
