#include <iostream>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

int main() {
    // Matrix dimensions
    const int M = 4096; // Number of rows in matrix A
    const int K = 65536; // Number of columns in matrix A, Number of rows in matrix B
    const int N = 4096; // Number of columns in matrix B

    // Host matrices
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];

    // Initialize matrices A and B
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = 2.0f; // You can initialize your values here
    }

    for (int i = 0; i < K * N; ++i) {
        h_B[i] = 1.0f; // You can initialize your values here
    }

    // Device matrices
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    hipMalloc((void**)&d_A, M * K * sizeof(float));
    hipMalloc((void**)&d_B, K * N * sizeof(float));
    hipMalloc((void**)&d_C, M * N * sizeof(float));

    // Copy host matrices to device
    hipMemcpy(d_A, h_A, M * K * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, K * N * sizeof(float), hipMemcpyHostToDevice);

    // cuBLAS handle
    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // CUDA events for timing
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // Record start event
    hipEventRecord(start);

    // Matrix multiplication using cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;

    hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);

    // Record stop event
    hipEventRecord(stop);

    // Synchronize to wait for completion of all tasks
    hipDeviceSynchronize();

    // Calculate elapsed time
    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);

    // Copy result matrix from device to host
    hipMemcpy(h_C, d_C, M * N * sizeof(float), hipMemcpyDeviceToHost);

    // Print the result matrix
    std::cout << "Result matrix:" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Print kernel execution time
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // Clean up
    hipblasDestroy(handle);
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}

