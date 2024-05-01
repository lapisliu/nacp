#include <cuda.h>
#include <mma.h>
#include <cublasLt.h>
#include <iostream>

#include "cuda_runtime.h"
#include "cublas_v2.h"

using namespace nvcuda;

#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

#define CUDA_KERNEL_CHECK() \
do { \
    cudaError_t error = cudaGetLastError(); \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(error) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) \
do { \
    cublasStatus_t error = call; \
    if (error != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "CUBLAS error: " << cublasGetStatusString(error) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;
//constexpr int WARP_SIZE = 16;
constexpr int TILE_SIZE = 32; // Tile size, divisible by WARP_SIZE for simplicity
// Fragment size for WMMA
constexpr int FRAG_SIZE = 16; // Should be a power of 2 and ideally a multiple of WARP_SIZE for optimal performance

__host__ void InitMatrix(half *A, half *B) {
    for (int i = 0; i < M * M; i++) {
        A[i] = __float2half(1.0f);
        B[i] = __float2half(2.0f);
    }
}

// Kernel function to perform matrix multiplication
__global__ void matrixMul(half *a, half *b, float *c, int m, int k, int n) {
    // Shared memory for tiles
    __shared__ half tileA[TILE_SIZE][TILE_SIZE];
    __shared__ half tileB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    // Define fragments for the output matrix
    wmma::fragment<wmma::matrix_a, FRAG_SIZE, FRAG_SIZE, FRAG_SIZE, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, FRAG_SIZE, FRAG_SIZE, FRAG_SIZE, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, FRAG_SIZE, FRAG_SIZE, FRAG_SIZE, float> c_frag;

    // Initialize accumulator fragment
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over tiles
    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile into shared memory
        if (ty < TILE_SIZE && t * TILE_SIZE + tx < k && row < m) {
            tileA[ty][tx] = a[row * k + t * TILE_SIZE + tx];
        } else {
            tileA[ty][tx] = 0.0f;
        }

        if (tx < TILE_SIZE && t * TILE_SIZE + ty < k && col < n) {
            tileB[ty][tx] = b[(t * TILE_SIZE + ty) * n + col];
        } else {
            tileB[ty][tx] = 0.0f;
        }

        __syncthreads();

      // Load tile fragments into WMMA fragments
        for (int i = 0; i < TILE_SIZE / FRAG_SIZE; ++i) {
            if (i*FRAG_SIZE>=TILE_SIZE) break;
            wmma::load_matrix_sync(a_frag, &tileA[i * FRAG_SIZE][0], TILE_SIZE);
            wmma::load_matrix_sync(b_frag, &tileB[0][i * FRAG_SIZE], TILE_SIZE);

            // Perform fragment-wise matrix multiplication
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        } 
    }

    // Store the result
    if (row < m && col < n) {
        // Store the result from the accumulator fragment to global memory
        wmma::store_matrix_sync(c, c_frag, n, wmma::mem_row_major);
    }
}

int main() {
    half *a, *a_d;
    half *b, *b_d;
    float *c, *c_d; 

    // Initialize timing infra
    cudaEvent_t wmma_start, wmma_stop;
    CUDA_CHECK(cudaEventCreate(&wmma_start));
    CUDA_CHECK(cudaEventCreate(&wmma_stop));
    float wmma_eventMs = 1.0f;

    // Allocate host memory
    a = (half*)malloc(M * K * sizeof(half));
    b = (half*)malloc(K * N * sizeof(half));
    c = (float*)malloc(M * N * sizeof(float));

    InitMatrix(a, b);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&a_d, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&b_d, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&c_d, M * N * sizeof(float)));

    // Copy data to the device
    CUDA_CHECK(cudaMemcpy(a_d, a, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_d, b, K * N * sizeof(half), cudaMemcpyHostToDevice));

    // Launch the kernel
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    CUDA_CHECK(cudaEventRecord(wmma_start));
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(a_d, b_d, c_d, M, K, N);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaEventRecord(wmma_stop));
    CUDA_CHECK(cudaEventSynchronize(wmma_stop));

    // Copy the result back to the host
    CUDA_CHECK(cudaMemcpy(c, c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventElapsedTime(&wmma_eventMs, wmma_start, wmma_stop));
    std::cout << "wmma: " << wmma_eventMs << std::endl;
    for (int i = 0; i < 5; i ++) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    free(a);
    free(b);
    free(c);

    return 0;
}

