#include <cuda.h>
#include <mma.h>
#include <iostream>

#include "cuda_runtime.h"
#include "cublas_v2.h"

// Define the dimensions of the matrices.
// WMMA supports various dimensions, here we use 16x16x16 for simplicity.

using namespace nvcuda;

__host__ void InitMatrix(half *A, half *B) {
    for (int i = 0; i < 4096 * 4096; i++) {
        A[i] = __float2half(1.0f);
        B[i] = __float2half(2.0f);
    }
}

__global__ void matMulWMMA(half *a, half *b, float *c) {
    int blockRow = blockIdx.y * blockDim.y + threadIdx.y;
    int blockCol = blockIdx.x * blockIdx.x + threadIdx.x;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Initialize the output fragment
    wmma::fill_fragment(c_frag, 0.0f);

    int row = threadIdx.y;
    int col = threadIdx.x;
    for (int k = 0; k < 256; k++) {
        if (blockRow < 4096 && k * 16 + col < 4096 && blockCol < 4096 && k * 16 + row < 4096) {
            wmma::load_matrix_sync(a_frag, a, 16);
            wmma::load_matrix_sync(b_frag, b, 16);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
        }
    }
    // Load the inputs
    // wmma::load_matrix_sync(a_frag, a, K);
    // wmma::load_matrix_sync(b_frag, b, K);

    // Perform the matrix multiplication
    // wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the output
    // wmma::store_matrix_sync(c, c_frag, N, wmma::mem_row_major);
}


int main() {
    half *a, *a_d;
    half *b, *b_d;
    float *c, *c_d, *blas_a, *blas_b, *blas_c;

    int m = 4096;
    int n = 4096;
    int k = 4096;

    // Initialize timing infra
    cudaEvent_t wmma_start, wmma_stop, blas_start, blas_stop;
    cudaEventCreate(&wmma_start);
    cudaEventCreate(&wmma_stop);
    float wmma_eventMs = 1.0f;
    cudaEventCreate(&blas_start);
    cudaEventCreate(&blas_stop);
    float blas_eventMs = 1.0f;

    // Allocate host memory
    a = (half*)malloc(m * k * sizeof(half));
    b = (half*)malloc(k * n * sizeof(half));
    c = (float*)malloc(m * n * sizeof(float));

    InitMatrix(a, b);

    // Allocate device memory
    cudaMalloc((void**)&a_d, m * k * sizeof(half));
    cudaMalloc((void**)&b_d, k * n * sizeof(half));
    cudaMalloc((void**)&c_d, m * n * sizeof(float));

    // Copy data to the device
    cudaMemcpy(a_d, a, m * k * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, k * n * sizeof(half), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(n/16, m/16);
    cudaEventRecord(wmma_start);
    matMulWMMA<<<blocksPerGrid, threadsPerBlock>>>(a_d, b_d, c_d);
    cudaEventRecord(wmma_stop);
    cudaEventSynchronize(wmma_stop);

    // Copy the result back to the host
    cudaMemcpy(c, c_d, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify the result by comparing it with cuBLAS's result
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMallocManaged((void**)&blas_a, m * k * sizeof(float));
    cudaMallocManaged((void**)&blas_b, k * n * sizeof(float));
    cudaMallocManaged((void**)&blas_c, m * n * sizeof(float));

    for (int i = 0; i < m * k; i++) blas_a[i] = 1.0f;
    for (int i = 0; i < k * n; i++) blas_b[i] = 2.0f;
    for (int i = 0; i < m * n; i++) blas_c[i] = 0.0f;

    float alpha = 1.0f, beta = 0.0f;
    cudaEventRecord(blas_start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, blas_a, m, blas_b, k, &beta, blas_c, m);
    cudaEventRecord(blas_stop);
    cudaEventSynchronize(blas_stop);

    cudaDeviceSynchronize();

    float *blas_c_host;
    blas_c_host = (float*)malloc(m * n * sizeof(float));
    cudaMemcpy(blas_c_host, blas_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventElapsedTime(&wmma_eventMs, wmma_start, wmma_stop);
    cudaEventElapsedTime(&blas_eventMs, blas_start, blas_stop);
    std::cout << "cublas: " << blas_eventMs << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << blas_c_host[i] << " ";
    }
    std::cout << std::endl << "wmma: " << wmma_eventMs << std::endl;
    for (int i = 0; i < 5; i ++) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    cudaFree(blas_a);
    cudaFree(blas_b);
    cudaFree(blas_c);
    free(a);
    free(b);
    free(c);
    free(blas_c_host);
    cublasDestroy(handle);

    return 0;
}
