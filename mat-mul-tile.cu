#include <iostream>
#include <cuda_runtime.h>

// Define matrix dimensions
const int M = 20000; // Number of rows in matrix A
const int K = 3333; // Number of columns in matrix A, Number of rows in matrix B
const int N = 7777; // Number of columns in matrix B
//const int BLOCK_SIZE = 2;
const int TILE_SIZE = 16;

// Kernel function to perform matrix multiplication
template<typename T>
__global__ void matrixMul(T *a, T *b, T *c, int m, int k, int n) {
    // Shared memory for tiles
    __shared__ T tileA[TILE_SIZE][TILE_SIZE];
    __shared__ T tileB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    T tmp = 0.0;

    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (t * TILE_SIZE + tx < k && row < m) {
            tileA[ty][tx] = a[row * k + t * TILE_SIZE + tx];
        } else {
            tileA[ty][tx] = 0.0;
        }

        if (t * TILE_SIZE + ty < k && col < n) {
            tileB[ty][tx] = b[(t * TILE_SIZE + ty) * n + col];
        } else {
            tileB[ty][tx] = 0.0;
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            tmp += tileA[ty][i] * tileB[i][tx];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = tmp;
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
    cudaMalloc((void**)&dev_a, M * K * sizeof(int));
    cudaMalloc((void**)&dev_b, K * N * sizeof(int));
    cudaMalloc((void**)&dev_c, M * N * sizeof(int));

    // Copy matrices a and b from host to device memory
    cudaMemcpy(dev_a, a, M * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, K * N * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Launch kernel
    matrixMul<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_c, M, K, N);

    // Copy result matrix c from device to host memory
    cudaMemcpy(c, dev_c, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result matrix
//    std::cout << "Result matrix:" << std::endl;
//    for (int i = 0; i < M; ++i) {
//        for (int j = 0; j < N; ++j) {
//            std::cout << c[i][j] << " ";
//        }
//        std::cout << std::endl;
//    }

    // Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}

