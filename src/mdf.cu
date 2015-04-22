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
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_snippets.h"

#define BSIZE 32

__global__ void kernel_iter(int w, int h, double alpha, double *X, const double *bound) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    double vt, vb, vl, vr;

    vt = y == 0 ? bound[0] : X[(y-1)*w + x];
    vb = y == h-1 ? bound[1] : X[(y+1)*w + x];
    vl = x == 0 ? bound[2] : X[y*w + x-1];
    vr = x == w-1 ? bound[3] : X[y*w + x+1];
    X[y*w + x] = ((vt+vb+vl+vr)/4.0);

    return;
}

extern "C" void run(int w, int h, int ks, double alpha, const double *bound, double *X) {
    int k;

    cudaDeviceProp prop;
    CudaSafeCall(cudaGetDeviceProperties(&prop, 0) );
    printf("[!] %s compiled in %s %s\n", __FILE__, __DATE__, __TIME__);
    printf("[!] Device Name: %s\n", prop.name);

    double *d_X, *d_B;
    cudaMalloc(&d_B, sizeof(double)*4);
    cudaMalloc(&d_X, sizeof(double)*w*h);
    cudaMemcpy(d_B, bound, sizeof(double)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, X, sizeof(double)*w*h, cudaMemcpyHostToDevice);

    const dim3 threads(BSIZE, BSIZE);
    const dim3 blocks(1 + w/BSIZE, 1 + h/BSIZE);
    for (k = 0; k < ks; k++) {
        kernel_iter<<<blocks, threads>>>(w, h, alpha, d_X, d_B);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(X, d_X, sizeof(double)*w*h, cudaMemcpyDeviceToHost);

    cudaFree(d_B);
    cudaFree(d_X);
    return;
}

extern "C" void runCPU(int w, int h, int ks, double alpha, double *bound, double *X) {
    int k, x, y;
    double vt, vb, vl, vr;

    for (k = 0; k < ks; k++) {
        for (x = 0; x < w; x++) {
            for (y = 0; y < h; y++) {
                vt = y == 0 ? bound[0] : X[(y-1)*w + x];
                vb = y == h-1 ? bound[1] : X[(y+1)*w + x];
                vl = x == 0 ? bound[2] : X[y*w + x-1];
                vr = x == w-1 ? bound[3] : X[y*w + x+1];

                X[y*w + x] = ((vt+vb+vl+vr)/double(4.0));
            }
        }
    }

    return;
}
