#include <iostream>
#include <cuda_runtime.h>

// Define matrix dimensions
const int M = 4096; // Number of rows in matrix A
const int K = 65536; // Number of columns in matrix A, Number of rows in matrix B
const int N = 4096; // Number of columns in matrix B
const int BLOCK_SIZE = 32;

// Kernel function to perform matrix multiplication
__global__ void matrixMul(float *a, float *b, float *c, int m, int k, int n) {
    // Calculate row and column indices of the element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform the multiplication for the element in C
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main() {
    float a[M][K], b[K][N], c[M][N];
    float *dev_a, *dev_b, *dev_c;
    cudaEvent_t start, stop;
    float elapsedTime=1.0f;

    // Initialize matrices a and b
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            a[i][j] = 2.0f;
        }
    }

    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            b[i][j] = 1.0f;
        }
    }

    // Allocate memory on the device
    cudaMalloc((void**)&dev_a, M * K * sizeof(float));
    cudaMalloc((void**)&dev_b, K * N * sizeof(float));
    cudaMalloc((void**)&dev_c, M * N * sizeof(float));

    // Copy matrices a and b from host to device memory
    cudaMemcpy(dev_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Create CUDA events for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Launch kernel
    matrixMul<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_c, M, K, N);

    // Record the stop event
    cudaEventRecord(stop, 0);

    // Synchronize to make sure stop event is recorded
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    cudaEventElapsedTime(&elapsedTime, start, stop);

     // Print the elapsed time
    std::cout << "Kernel execution time: " << elapsedTime << " milliseconds" << std::endl;


    // Copy result matrix c from device to host memory
    cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result matrix
//     std::cout << "Result matrix:" << std::endl;
//     for (int i = 0; i < M; ++i) {
//         for (int j = 0; j < N; ++j) {
//             std::cout << c[i][j] << " ";
//         }
//         std::cout << std::endl;
//     }

    // Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}

