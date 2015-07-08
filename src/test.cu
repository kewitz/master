// Sadiku's Numerical Techniques in Electromagnetics. pg.712
extern "C" int testeCG(int n, int kmax, float err, float* A, float* x,
                       float* b) {
    int i, j, k = 1;
    float alpha, beta, sum1, sum2, sum3 = 1.0f, sum4;
    float *r = cast(float*, malloc(n*sizeof(float)));
    float *p = cast(float*, malloc(n*sizeof(float)));
    float *u = cast(float*, malloc(n*sizeof(float)));

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

        sum1 = 0.0f; sum2 = 0.0f;
        for (i = 0; i < n; i++) {
            sum1 += p[i]*r[i];
            sum2 += p[i]*u[i];
        }

        alpha = sum2 != 0.0f ? sum1/sum2 : 0.0f;
        alpha = isnan(alpha) ? 0.0f : alpha;

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

#if CUDA
extern "C" float teste_sum_reduction(int size, float *a, float *b) {
    const dim3 threads(BSIZE);
    const dim3 blocks(1 + size/BSIZE);

    float *_a, *_b, *_sum, sum = 0;
    smalloc(&_a, sizeof(float)*size);
    smalloc(&_b, sizeof(float)*size);
    smalloc(&_sum, sizeof(float));

    smemcpy(_a, a, sizeof(float)*size, cudaMemcpyHostToDevice);
    smemcpy(_b, b, sizeof(float)*size, cudaMemcpyHostToDevice);
    smemcpy(_sum, &sum, sizeof(float), cudaMemcpyHostToDevice);

    kernel_util_vecsummult<<<blocks, threads>>>(size, _a, _b, _sum);
    cudaDeviceSynchronize();
    smemcpy(&sum, _sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    return sum;
}
#endif
