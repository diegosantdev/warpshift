#include <iostream>
#include <vector>
#include <chrono>

#ifndef __HIP_PLATFORM_AMD__
#include <cuda_runtime.h>
#else
#include <hip/hip_runtime.h>
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaFree hipFree
#define cudaDeviceSynchronize hipDeviceSynchronize
#endif

// SAXPY: Y[i] = A * X[i] + Y[i]
__global__
void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main(void) {
    int N = 1 << 20; // 1M elements
    float *x, *y, *d_x, *d_y;
    x = (float*)malloc(N * sizeof(float));
    y = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Warmup
    saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
    cudaDeviceSynchronize();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 100; iter++) {
        saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> diff = end - start;
    double avg_time = diff.count() / 100.0;

    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Validation
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        // Expected: 2.0 * 1.0 + 2.0 = 4.0 for first iter... 
        // We ran it 101 times actually (warmup + 100).
        // 2.0f * 101 + 2.0f = 204.0f
        maxError = max(maxError, abs(y[i] - 204.0f));
    }

    std::cout << "[WARPSHIFT_BENCHMARK] time_ms=" << avg_time << std::endl;
    
    if (maxError < 1e-5) {
        std::cout << "[WARPSHIFT_VALIDATION] status=SUCCESS" << std::endl;
    } else {
        std::cout << "[WARPSHIFT_VALIDATION] status=FAILED maxError=" << maxError << std::endl;
    }

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);

    return 0;
}
