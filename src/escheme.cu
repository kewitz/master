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
    int i;
    // clock_t t;

    float *rsum = static_cast<float*>(malloc(nn*sizeof(float)));
    float *dsum = static_cast<float*>(malloc(nn*sizeof(float)));
    float *r = static_cast<float*>(malloc(nn*sizeof(float)));
    float *z = static_cast<float*>(malloc(nn*sizeof(float)));
    float *p = static_cast<float*>(malloc(nn*sizeof(float)));
    float *q = static_cast<float*>(malloc(nn*sizeof(float)));
    // float *Vos = (float*) malloc(nn*sizeof(float));
    // memcpy(Vos, V, nn*sizeof(float));

    // Inicialização dos vetores.
    for (i = 0; i < nn; i++) {
        rsum[i] = 0.0;
        dsum[i] = 0.0;
    }

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
            (pow(J2-J4, 2) + pow(J3-J1, 2))*E.eps/dJ : 0.0;
        elements[i].matriz[1] = dJ != 0.0 ?
            (pow(J4, 2) + pow(J3, 2))*E.eps/dJ : 0.0;
        elements[i].matriz[2] = dJ != 0.0 ?
            (pow(J2, 2) + pow(J1, 2))*E.eps/dJ : 0.0;
        elements[i].matriz[3] = dJ != 0.0 ?
            ((J2-J4)*J4 - (J3-J1)*J3)*E.eps/dJ : 0.0;
        elements[i].matriz[4] = dJ != 0.0 ?
            ((J2-J4)*-1*J2 + (J3-J1)*J1)*E.eps/dJ : 0.0;
        elements[i].matriz[5] = dJ != 0.0 ?
            (J4*-1*J2 - J3*J1)*E.eps/dJ : 0.0;
    }

    // Calcula dsum e rsum.
    int n1, n2, n3;
    for (i = 0; i < ne; i++) {
        E = elements[i];
        n1 = E.nodes[0]; n2 = E.nodes[1]; n3 = E.nodes[2];

        dsum[n1] += E.matriz[0];
        dsum[n2] += E.matriz[1];
        dsum[n3] += E.matriz[2];

        rsum[n1] -= E.matriz[3]*V[n2] - E.matriz[4]*V[n3];
        rsum[n2] -= E.matriz[3]*V[n1] - E.matriz[5]*V[n3];
        rsum[n3] -= E.matriz[4]*V[n1] - E.matriz[5]*V[n2];
    }

    // Inicializa vetor de resíduos
    // r = b - Ax
    float ri, erri = 0.0;
    for (i = 0; i < nn; i++) {
        ri = nodes[i].calc ? rsum[i] - dsum[i]*V[i] : 0.0;
        r[i] = ri;
        if (ri != 0)
            erri += pow(ri, 2);
    }
    erri = sqrt(erri);

    // Iterações.
    int k = 1;
    float rho, rhop, alpha, beta, somaPQ, errf=1, errlat = 10*errmin;
    while (errlat > errmin && k < kmax) {
        rho = 0.0;
        // Pré-condicionador Jacobi e calcula Rho.
        for (i = 0; i < nn; i++) {
            z[i] = r[i]/dsum[i];
            rho += z[i]*r[i];
        }

        // Calcula P = Z + BETA*P
        if (k == 1) {
            for (i = 0; i < nn; i++)
                p[i] = z[i];
        } else {
            beta = rho/rhop;
            for (i = 0; i < nn; i++)
                p[i] = z[i] + beta*p[i];
        }

        // Calcula Q = A*P
        for (i = 0; i < nn; i++)
            q[i] = 0.0;
        for (i = 0; i < ne; i++) {
            E = elements[i];
            n1 = E.nodes[0]; n2 = E.nodes[1]; n3 = E.nodes[2];

            q[n1] += E.matriz[0]*p[n1] + E.matriz[3]*p[n2] + E.matriz[4]*p[n3];
            q[n2] += E.matriz[3]*p[n1] + E.matriz[1]*p[n2] + E.matriz[5]*p[n3];
            q[n3] += E.matriz[4]*p[n1] + E.matriz[5]*p[n2] + E.matriz[2]*p[n3];
        }

        for (i = 0; i < nn; i++)
            if (!nodes[i].calc)
                q[i] = p[i];

        // Calcula Alpha
        somaPQ = 0.0;
        for (i = 0; i < nn; i++)
            somaPQ += p[i]*q[i];
        alpha = rho/somaPQ;

        // Atualiza 'x' e calcula o novo resíduo.
        errf = 0.0;
        for (i = 0; i < nn; i++) {
            V[i] += alpha*p[i];
            r[i] -= alpha*q[i];
            errf += pow(r[i], 2);
        }
        errf = sqrt(errf);
        errlat = errf/erri;

        rhop = rho;
        k++;
    }


    free(rsum);
    free(dsum);
    free(r);
    free(z);
    free(p);
    free(q);
    return k;
}

extern "C" int runCPUCG(int ne, int nn, int kmax, float errmin,
                      elementri *elements, node *nodes, float *V, bool verbose,
                      float *bench) {
    int i, k;
    float *rsum = (float*)malloc(nn*sizeof(float));
    float *dsum = (float*)malloc(nn*sizeof(float));

    // Inicialização dos vetores.
    for (i = 0; i < nn; i++) {
        rsum[i] = 0.0;
        dsum[i] = 0.0;
    }

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

    // Calcula dsum e rsum.
    int n1, n2, n3;
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
    float *r = (float*)malloc(nn*sizeof(float));
    float *p = (float*)malloc(nn*sizeof(float));
    float *u = (float*)malloc(nn*sizeof(float));

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

    return k;
}

// Sadiku's Numerical Techniques in Electromagnetics. pg.712
extern "C" int testeCG(int n, int kmax, float err, float* A, float* x,
                       float* b) {
    int i, j, k = 1;
    float alpha, beta, sum1, sum2, sum3 = 1, sum4;
    float *r = (float*)malloc(n*sizeof(float));
    float *p = (float*)malloc(n*sizeof(float));
    float *u = (float*)malloc(n*sizeof(float));

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
