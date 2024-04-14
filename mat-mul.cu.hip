#include "hip/hip_runtime.h"
#include <iostream>
#include <hip/hip_runtime.h>

// Define matrix dimensions
const int M = 4096; // Number of rows in matrix A
const int K = 4096; // Number of columns in matrix A, Number of rows in matrix B
const int N = 4096; // Number of columns in matrix B
const int BLOCK_SIZE = 2;

// Kernel function to perform matrix multiplication
__global__ void matrixMul(int *a, int *b, int *c, int m, int k, int n) {
    // Calculate row and column indices of the element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform the multiplication for the element in C
    if (row < m && col < n) {
        int sum = 0;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main() {
    int a[M][K], b[K][N], c[M][N];
    int *dev_a, *dev_b, *dev_c;

    // Initialize matrices a and b
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            a[i][j] = 2;
        }
    }

    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            b[i][j] = 1;
        }
    }

    // Allocate memory on the device
    hipMalloc((void**)&dev_a, M * K * sizeof(int));
    hipMalloc((void**)&dev_b, K * N * sizeof(int));
    hipMalloc((void**)&dev_c, M * N * sizeof(int));

    // Copy matrices a and b from host to device memory
    hipMemcpy(dev_a, a, M * K * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(dev_b, b, K * N * sizeof(int), hipMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    matrixMul<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_c, M, K, N);

    // Copy result matrix c from device to host memory
    hipMemcpy(c, dev_c, M * N * sizeof(int), hipMemcpyDeviceToHost);

    // Print the result matrix
//    std::cout << "Result matrix:" << std::endl;
//    for (int i = 0; i < M; ++i) {
//        for (int j = 0; j < N; ++j) {
//            std::cout << c[i][j] << " ";
//        }
//        std::cout << std::endl;
//    }

    // Free device memory
    hipFree(dev_a);
    hipFree(dev_b);
    hipFree(dev_c);

    return 0;
}

