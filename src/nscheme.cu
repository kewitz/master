/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2014 Leonardo Kewitz
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_snippets.h"
#include "nscheme.h"

// Kernel responsável por uma iteração.
__global__ void kernel_iter(int nn, elementri *elements, node *nodes, double *V, double *R) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nn) return;

    int e;
    double diag_sum = 0.0, right_sum = 0.0, Vi;

    node Node = nodes[i];
    if (Node.calc == false) return;

    elementri Element;
    for (e = 0; e < Node.ne; e++) {
        Element = elements[Node.elements[e]];
        if (Node.i == Element.nodes[0]) {
            diag_sum  += Element.matriz[0];                     // A11
            right_sum -= Element.matriz[3]*V[Element.nodes[1]]; // A12
            right_sum -= Element.matriz[4]*V[Element.nodes[2]]; // A13
        }
        if (Node.i == Element.nodes[1]) {
            diag_sum += Element.matriz[1];                      // A22
            right_sum -= Element.matriz[3]*V[Element.nodes[1]]; // A21
            right_sum -= Element.matriz[5]*V[Element.nodes[2]]; // A23
        }
        if (Node.i == Element.nodes[2]) {
            diag_sum += Element.matriz[2];                      // A33
            right_sum -= Element.matriz[4]*V[Element.nodes[0]]; // A31
            right_sum -= Element.matriz[5]*V[Element.nodes[1]]; // A32
        }
    }
    Vi = diag_sum == 0 ? 0.0 : right_sum/diag_sum;
    R[Node.i] = fabs(V[Node.i] - Vi);
    V[Node.i] = Vi;

    return;
}

// Kernel de pre-processamento responsável por calcular as matrizes de contribu-
// ição de todos os elementos.
__global__ void kernel_pre(int ne, elementri *elements, node *nodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ne) return;

    elementri E = elements[i];
    node N1 = nodes[E.nodes[0]], N2 = nodes[E.nodes[1]], N3 = nodes[E.nodes[2]];

    // Calcula argumentos necessários
    double J1, J2, J3, J4, dJ;
    J1 = (double)N2.x - (double)N1.x;
    J2 = (double)N2.y - (double)N1.y;
    J3 = (double)N3.x - (double)N1.x;
    J4 = (double)N3.y - (double)N1.y;
    dJ = 2*(J1*J4 - J3*J2);

    // Calcula a matriz de contribuições do elemento.
    elements[i].matriz[0] = (pow(J2-J4,2) + pow(J3-J1,2))/dJ;   // C11
    elements[i].matriz[1] = (pow(J4,2) + pow(J3,2))/dJ;         // C22
    elements[i].matriz[2] = (pow(J2,2) + pow(J1,2))/dJ;         // C33
    elements[i].matriz[3] = ((J2-J4)*J4 - (J3-J1)*J3)/dJ;       // C12 C21
    elements[i].matriz[4] = ((J2-J4)*-1*J2 + (J3-J1)*J1)/dJ;    // C13 C31
    elements[i].matriz[5] = (J4*-1*J2 - J3*J1)/dJ;              // C23 C32

    return;
}

// Função externa que processa o problema, responsável por alocar a memória no
// device e invocar todas os kernels necessários.
extern "C" int run(int ne, int nn, double alpha, elementri *elements, node *nodes, double *V, bool verbose) {
    if (verbose) {
        cudaDeviceProp prop;
        CudaSafeCall(cudaGetDeviceProperties(&prop, 0) );
        printf("[!] Device Name: %s\n", prop.name);
        printf("[!] %s compiled in %s %s\n", __FILE__, __DATE__, __TIME__);
    }

    int k = 1, i;
    double *d_V, *d_R, *R;
    node *d_nodes;
    elementri *d_elements;
    const dim3 threads(512);
    const dim3 preblocks(1 + ne/512);
    const dim3 iterblocks(1 + nn/512);
    size_t s_Elements = sizeof(elementri)*ne,
           s_Nodes = sizeof(node)*nn,
           s_V = sizeof(double)*nn;

    // Malloc
    R = (double*)malloc(s_V);
    CudaSafeCall(cudaMalloc(&d_elements, s_Elements));
    CudaSafeCall(cudaMalloc(&d_nodes, s_Nodes));
    CudaSafeCall(cudaMalloc(&d_V, s_V));
    CudaSafeCall(cudaMalloc(&d_R, s_V));
    // Memcpy
    CudaSafeCall(cudaMemcpy(d_elements, elements, s_Elements, cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_nodes, nodes, s_Nodes, cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_V, V, s_V, cudaMemcpyHostToDevice));

    // Pre-processamento
    kernel_pre<<<preblocks, threads>>>(ne, d_elements, d_nodes);
    cudaDeviceSynchronize();

    // Iterações
    double r = 10*alpha;
    while (r > alpha) {
        kernel_iter<<<iterblocks, threads>>>(nn, d_elements, d_nodes, d_V, d_R);
        cudaDeviceSynchronize();

        // Calcula a convergência à cada 300 iterações.
        if (k % 300 == 0) {
            r = 0.0;
            CudaSafeCall(cudaMemcpy(R, d_R, s_V, cudaMemcpyDeviceToHost));
            for (i = 0; i < nn; i++) {
                node N = nodes[i];
                if (N.calc) {
                    r = (R[N.i] > r) ? R[N.i] : r;
                }
            }
        }

        k++;
    }

    CudaSafeCall(cudaMemcpy(V, d_V, s_V, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    cudaFree(d_V);
    cudaFree(d_R);
    cudaFree(d_nodes);
    cudaFree(d_elements);

    return k;
}

// Função externa que processa o problema no CPU.
extern "C" int runCPU(int ne, int nn, double alpha, elementri *elements, node *nodes, double *V, bool verbose) {
    int i, k = 0;

    // Pre-processamento. Calcula as matrizes de contribuição dos elementos.
    for (i = 0; i < ne; i++) {
        elementri E = elements[i];
        node N1 = nodes[E.nodes[0]], N2 = nodes[E.nodes[1]], N3 = nodes[E.nodes[2]];

        // Calcula argumentos necessários
        double J1, J2, J3, J4, dJ;
        J1 = (double)N2.x - (double)N1.x;
        J2 = (double)N2.y - (double)N1.y;
        J3 = (double)N3.x - (double)N1.x;
        J4 = (double)N3.y - (double)N1.y;
        dJ = 2*(J1*J4 - J3*J2);

        // Calcula a matriz de contribuições do elemento.
        elements[i].matriz[0] = (pow(J2-J4,2) + pow(J3-J1,2))/dJ;   // C11
        elements[i].matriz[1] = (pow(J4,2) + pow(J3,2))/dJ;         // C22
        elements[i].matriz[2] = (pow(J2,2) + pow(J1,2))/dJ;         // C33
        elements[i].matriz[3] = ((J2-J4)*J4 - (J3-J1)*J3)/dJ;       // C12 C21
        elements[i].matriz[4] = ((J2-J4)*-1*J2 + (J3-J1)*J1)/dJ;    // C13 C31
        elements[i].matriz[5] = (J4*-1*J2 - J3*J1)/dJ;              // C23 C32
    }

    double r = 10*alpha, diff;
    while (r > alpha) {
        if (k % 300 == 0) r = 0.0;
        for (i = 0; i < nn; i++) {
            int e;
            double diag_sum = 0.0, right_sum = 0.0, Vi;

            node Node = nodes[i];
            if (Node.calc == true) {
                elementri Element;
                for (e = 0; e < Node.ne; e++) {
                    Element = elements[Node.elements[e]];
                    if (Node.i == Element.nodes[0]) {
                        diag_sum  += Element.matriz[0];                     // A11
                        right_sum -= Element.matriz[3]*V[Element.nodes[1]]; // A12
                        right_sum -= Element.matriz[4]*V[Element.nodes[2]]; // A13
                    }
                    if (Node.i == Element.nodes[1]) {
                        diag_sum += Element.matriz[1];                      // A22
                        right_sum -= Element.matriz[3]*V[Element.nodes[1]]; // A21
                        right_sum -= Element.matriz[5]*V[Element.nodes[2]]; // A23
                    }
                    if (Node.i == Element.nodes[2]) {
                        diag_sum += Element.matriz[2];                      // A33
                        right_sum -= Element.matriz[4]*V[Element.nodes[0]]; // A31
                        right_sum -= Element.matriz[5]*V[Element.nodes[1]]; // A32
                    }
                }
                Vi = diag_sum == 0 ? 0.0 : right_sum/diag_sum;

                if (k % 300 == 0) {
                    diff = fabs(V[Node.i] - Vi);
                    r = (diff > r) ? diff : r;
                }

                V[Node.i] = Vi;
            }
        }

        k++;
    }

    return k;
}

extern "C" void teste_Arrays(int ne, int nn, elementri *elements, node *nodes) {
    int i, k;
    printf("\nStarting Node Test...\n\n");
    for (i = 0; i < nn; i++) {
        if (i%100 == 0) {
            printf("\tNode %i (%.3f, %.3f):\n", i, nodes[i].x, nodes[i].y);
            printf("\t\tElements:");
            for (k = 0; k < nodes[i].ne; k++) {
                printf(" %i", (int)nodes[i].elements[k]);
            }
            printf("\n");
        }
    }

    printf("\nStarting Elements Test...\n\n");
    for (i = 0; i < nn; i++) {
        if (i%100 == 0) {
            printf("\tElement %i:\n", i);
            printf("\t\tNodes:");
            for (k = 0; k < 3; k++) {
                printf(" %i", elements[i].nodes[k]);
            }
            printf("\n\t\tMatriz:");
            printf("\n\t\t\t%.3f %.3f %.3f",
                   elements[i].matriz[0],
                   elements[i].matriz[3],
                   elements[i].matriz[4]);
            printf("\n\t\t\t%.3f %.3f %.3f",
                   elements[i].matriz[3],
                   elements[i].matriz[1],
                   elements[i].matriz[5]);
            printf("\n\t\t\t%.3f %.3f %.3f",
                   elements[i].matriz[4],
                   elements[i].matriz[5],
                   elements[i].matriz[2]);
            printf("\n");
        }
    }
}
