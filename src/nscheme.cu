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
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_snippets.h"
#include "nscheme.h"


// Kernel responsável por uma iteração.
__global__ void kernel_iter(const int nn, const int k, const float alpha, float R, elementri *elements, node *nodes, float *V, int *conv) {
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
            right_sum -= Element.matriz[3]*V[Element.nodes[1]];
            right_sum -= Element.matriz[5]*V[Element.nodes[2]];
        }
        if (Node.i == Element.nodes[2]) {
            diag_sum += Element.matriz[2];
            right_sum -= Element.matriz[4]*V[Element.nodes[0]];
            right_sum -= Element.matriz[5]*V[Element.nodes[1]];
        }
    }
    Vi = diag_sum == 0 ? 0.0 : fdividef(right_sum*Element.eps, diag_sum*Element.eps);
    Vo = V[Node.i];
    diff = Vi - Vo;
    Vi = fmaf(R, diff, Vi);
    c = fabsf(diff) > fabsf(Vi*alpha);
    atomicOr(conv, c);
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
    float J1, J2, J3, J4, dJ;
    J1 = (float)N2.x - (float)N1.x;
    J2 = (float)N2.y - (float)N1.y;
    J3 = (float)N3.x - (float)N1.x;
    J4 = (float)N3.y - (float)N1.y;
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
extern "C" int run(const int ne, const int nn, const float alpha, const float Rf, const float T, elementri *elements, node *nodes, float *V, bool verbose, float *bench) {
    clock_t t;
    int k = 1, conv, *d_conv;
    float *d_V;
    float R;
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
    CudaSafeCall(cudaMemcpy(d_elements, elements, s_Elements, cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_nodes, nodes, s_Nodes, cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_V, V, s_V, cudaMemcpyHostToDevice));
    t = clock() - t;
    if (verbose) printf ("[!] GPU Malloc and MemCpy: %d clicks or %f seconds.\n", (int)t, ((float)t)/CLOCKS_PER_SEC);
    bench[0] = ((float)t)/CLOCKS_PER_SEC;

    // Pre-processamento
    t = clock();
    kernel_pre<<<preblocks, threads>>>(ne, d_elements, d_nodes);
    cudaDeviceSynchronize();
    t = clock() - t;
    if (verbose) printf ("[!] GPU Element integration: %d clicks or %f seconds.\n", (int)t, ((float)t)/CLOCKS_PER_SEC);
    bench[1] = ((float)t)/CLOCKS_PER_SEC;

    // Iterações
    t = clock();
    conv = 1;
    while (conv == 1) {
        conv = 0;
        R = Rf*(1-expf(-k/T));
        cudaMemcpy(d_conv, &conv, sizeof(int), cudaMemcpyHostToDevice);
        kernel_iter<<<iterblocks, threads>>>(nn, k, alpha, R, d_elements, d_nodes, d_V, d_conv);
        cudaDeviceSynchronize();
        cudaMemcpy(&conv, d_conv, sizeof(int), cudaMemcpyDeviceToHost);
        k++;
    }
    t = clock() - t;
    if (verbose) printf ("[!] GPU Convergence: %d clicks or %f seconds.\n", (int)t, ((float)t)/CLOCKS_PER_SEC);
    bench[2] = ((float)t)/CLOCKS_PER_SEC;

    CudaSafeCall(cudaMemcpy(V, d_V, s_V, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    cudaFree(d_V);
    cudaFree(d_nodes);
    cudaFree(d_elements);
    cudaFree(d_conv);

    return k;
}

// Função externa que processa o problema no CPU.
extern "C" int runCPU(int ne, int nn, float alpha, float Rf, float T, elementri *elements, node *nodes, float *V, bool verbose, float *bench) {
    int i, k = 0;
    float R;
    clock_t t;

    float *Vos = (float*) malloc(nn*sizeof(float));
    memcpy(Vos, V, nn*sizeof(float));

    // Pre-processamento. Calcula as matrizes de contribuição dos elementos.
    t = clock();
    for (i = 0; i < ne; i++) {
        elementri E = elements[i];
        node N1 = nodes[E.nodes[0]], N2 = nodes[E.nodes[1]], N3 = nodes[E.nodes[2]];

        // Calcula argumentos necessários
        float J1, J2, J3, J4, dJ;
        J1 = (float)N2.x - (float)N1.x;
        J2 = (float)N2.y - (float)N1.y;
        J3 = (float)N3.x - (float)N1.x;
        J4 = (float)N3.y - (float)N1.y;
        dJ = 2*(J1*J4 - J3*J2);

        // Calcula a matriz de contribuições do elemento.
        elements[i].matriz[0] = (pow(J2-J4,2) + pow(J3-J1,2))/dJ;   // C11
        elements[i].matriz[1] = (pow(J4,2) + pow(J3,2))/dJ;         // C22
        elements[i].matriz[2] = (pow(J2,2) + pow(J1,2))/dJ;         // C33
        elements[i].matriz[3] = ((J2-J4)*J4 - (J3-J1)*J3)/dJ;       // C12 C21
        elements[i].matriz[4] = ((J2-J4)*-1*J2 + (J3-J1)*J1)/dJ;    // C13 C31
        elements[i].matriz[5] = (J4*-1*J2 - J3*J1)/dJ;              // C23 C32
    }
    t = clock() - t;
    if (verbose) printf ("[!] CPU Element integration: %d clicks or %f seconds.\n", (int)t, ((float)t)/CLOCKS_PER_SEC);
    bench[0] = ((float)t)/CLOCKS_PER_SEC;

    t = clock();
    float diff;
    bool run = true;
    while (run) {
        run = false;
        R = Rf*(1-expf(-k/T));
        for (i = 0; i < nn; i++) {
            int e;
            float diag_sum = 0.0, right_sum = 0.0, Vi, Vo;
            node Node = nodes[i];
            if (Node.calc == true) {
                Vo = V[Node.i];
                elementri Element;
                for (e = 0; e < Node.ne; e++) {
                    Element = elements[Node.elements[e]];
                    if (Node.i == Element.nodes[0]) {
                        diag_sum  += Element.matriz[0];
                        right_sum -= Element.matriz[3]*Vos[Element.nodes[1]];
                        right_sum -= Element.matriz[4]*Vos[Element.nodes[2]];
                    }
                    if (Node.i == Element.nodes[1]) {
                        diag_sum += Element.matriz[1];
                        right_sum -= Element.matriz[3]*Vos[Element.nodes[1]];
                        right_sum -= Element.matriz[5]*Vos[Element.nodes[2]];
                    }
                    if (Node.i == Element.nodes[2]) {
                        diag_sum += Element.matriz[2];
                        right_sum -= Element.matriz[4]*Vos[Element.nodes[0]];
                        right_sum -= Element.matriz[5]*Vos[Element.nodes[1]];
                    }
                }
                Vi = diag_sum == 0 ? 0.0 : right_sum/diag_sum;
                diff = Vi - Vo;
                Vi += R*diff;
                run |= fabs(diff) > fabs(Vi*alpha);
                V[Node.i] = Vi;
            }
        }
        for (i = 0; i < nn; i++) {
            node Node = nodes[i];
            Vos[Node.i] = V[Node.i];
        }

        k++;
    }
    t = clock() - t;
    if (verbose) printf ("[!] CPU Convergence: %d clicks or %f seconds.\n", (int)t, ((float)t)/CLOCKS_PER_SEC);
    bench[1] = ((float)t)/CLOCKS_PER_SEC;

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
