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


// Kernel responsável por uma iteração.
__global__ void kernel_iter(const int nn, const int k, const float R,
    const float errmin, elementri *elements, node *nodes, float *V, int *conv) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nn) return;

    node Node = nodes[i];
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

    Vi = diag_sum == 0 ? 0.0 : fdividef(right_sum, diag_sum);
    Vo = V[Node.i];
    diff = Vi - Vo;
    Vi = fmaf(R, diff, Vi);
    c = fabs(diff) > errmin;
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
extern "C" int runGPU(const int ne, const int nn, const int kmax, const float R,
                   const float errmin, elementri *elements,
                   node *nodes, float *V, bool verbose, float *bench) {
    clock_t t;
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
    t = clock();
    CudaSafeCall(cudaMalloc(&d_elements, s_Elements));
    CudaSafeCall(cudaMalloc(&d_nodes, s_Nodes));
    CudaSafeCall(cudaMalloc(&d_V, s_V));
    CudaSafeCall(cudaMalloc(&d_conv, sizeof(int)));
    // Memcpy
    CudaSafeCall(cudaMemcpy(d_elements, elements, s_Elements,
                            cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_nodes, nodes, s_Nodes, cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_V, V, s_V, cudaMemcpyHostToDevice));
    t = clock() - t;
    if (verbose)
        printf("[!] GPU Malloc and MemCpy: %d clicks or %f seconds.\n",
               (int)t, ((float)t)/CLOCKS_PER_SEC);
    bench[0] = ((float)t)/CLOCKS_PER_SEC;

    // Pre-processamento
    t = clock();
    kernel_pre<<<preblocks, threads>>>(ne, d_elements, d_nodes);
    cudaDeviceSynchronize();
    t = clock() - t;
    if (verbose)
        printf("[!] GPU Element integration: %d clicks or %f seconds.\n",
               (int)t, ((float)t)/CLOCKS_PER_SEC);
    bench[1] = ((float)t)/CLOCKS_PER_SEC;

    // Iterações
    t = clock();
    conv = 1;
    while (conv == 1 && k < kmax) {
        conv = 0;
        cudaMemcpy(d_conv, &conv, sizeof(int), cudaMemcpyHostToDevice);
        kernel_iter<<<iterblocks, threads>>>(nn, k, R, errmin, d_elements,
                                             d_nodes, d_V, d_conv);
        cudaDeviceSynchronize();
        cudaMemcpy(&conv, d_conv, sizeof(int), cudaMemcpyDeviceToHost);
        k++;
    }
    t = clock() - t;
    if (verbose)
        printf("[!] GPU Convergence: %d clicks or %f seconds.\n", (int)t,
               ((float)t)/CLOCKS_PER_SEC);
    bench[2] = ((float)t)/CLOCKS_PER_SEC;

    CudaSafeCall(cudaMemcpy(V, d_V, s_V, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    cudaFree(d_V); cudaFree(d_nodes); cudaFree(d_elements); cudaFree(d_conv);

    return k;
}

// Função externa que processa o problema no CPU.
extern "C" int runCPU(int ne, int nn, int kmax, float R, float errmin,
                      elementri *elements, node *nodes, float *V, bool verbose,
                      float *bench) {
    int i, k = 1;
    clock_t t;

    // Pre-processamento. Calcula as matrizes de contribuição dos elementos.
    t = clock();
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
    t = clock() - t;
    if (verbose)
        printf("[!] CPU Element integration: %d clicks or %f seconds.\n",
               (int)t, ((float)t)/CLOCKS_PER_SEC);
    bench[0] = ((float)t)/CLOCKS_PER_SEC;

    t = clock();
    float diff;
    bool run = true;
    while (run && k < kmax) {
        run = false;
        for (i = 0; i < nn; i++) {
            int e;
            float diag_sum = 0.0, right_sum = 0.0, Vi, Vo;
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
    t = clock() - t;
    if (verbose)
        printf("[!] CPU Convergence: %d clicks or %f seconds.\n", (int)t,
               ((float)t)/CLOCKS_PER_SEC);
    bench[1] = ((float)t)/CLOCKS_PER_SEC;

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
