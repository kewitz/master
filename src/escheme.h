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

typedef struct {
    int nodes[3];
    float matriz[6]; // C11 C22 C33 C12 C13 C23
    float eps;
} elementri;

typedef struct {
    float x;
    float y;
    bool calc;
} node;

// HEADER

// Snippets
extern "C" void hello() {
    printf("[!] %s compiled in %s %s\n", __FILE__, __DATE__, __TIME__);
}

extern "C" void getInfo() {
    cudaDeviceProp prop;
    CudaSafeCall(cudaGetDeviceProperties(&prop, 0) );
    printf("[!] Device Name: %s\n", prop.name);
}

extern "C" int getCUDAdevices() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    return deviceCount;
}
