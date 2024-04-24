#include <cuda.h>
#include <mma.h>
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

__host__ void InitMatrix(half *A, half *B) {
    for (int i = 0; i < 40960 * 40960; i++) {
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
    for (int k = 0; k < 2560; k++) {
        if (blockRow < 40960 && k * 16 + col < 40960 && blockCol < 40960 && k * 16 + row < 40960) {
            wmma::load_matrix_sync(a_frag, a, 16);
            wmma::load_matrix_sync(b_frag, b, 16);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
        }
    }
}

int main() {
    half *a, *a_d;
    half *b, *b_d;
    float *c, *c_d, *blas_a, *blas_b, *blas_c;

    int m = 40960;
    int n = 40960;
    int k = 40960;

    // Initialize timing infra
    cudaEvent_t wmma_start, wmma_stop, blas_start, blas_stop;
    CUDA_CHECK(cudaEventCreate(&wmma_start));
    CUDA_CHECK(cudaEventCreate(&wmma_stop));
    float wmma_eventMs = 1.0f;
    CUDA_CHECK(cudaEventCreate(&blas_start));
    CUDA_CHECK(cudaEventCreate(&blas_stop));
    float blas_eventMs = 1.0f;

    // Allocate host memory
    a = (half*)malloc(m * k * sizeof(half));
    b = (half*)malloc(k * n * sizeof(half));
    c = (float*)malloc(m * n * sizeof(float));

    InitMatrix(a, b);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&a_d, m * k * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&b_d, k * n * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&c_d, m * n * sizeof(float)));

    // Copy data to the device
    CUDA_CHECK(cudaMemcpy(a_d, a, m * k * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_d, b, k * n * sizeof(half), cudaMemcpyHostToDevice));

    // Launch the kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n+16-1)/16, (m+16-1)/16);
    CUDA_CHECK(cudaEventRecord(wmma_start));
    matMulWMMA<<<blocksPerGrid, threadsPerBlock>>>(a_d, b_d, c_d);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaEventRecord(wmma_stop));
    CUDA_CHECK(cudaEventSynchronize(wmma_stop));

    // Copy the result back to the host
    CUDA_CHECK(cudaMemcpy(c, c_d, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify the result by comparing it with cuBLAS's result
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate separate host and device memory for blas_a, blas_b, and blas_c
    blas_a = (float*)malloc(m * k * sizeof(float));
    blas_b = (float*)malloc(k * n * sizeof(float));
    blas_c = (float*)malloc(m * n * sizeof(float));

    float *blas_a_d, *blas_b_d, *blas_c_d;
    CUDA_CHECK(cudaMalloc((void**)&blas_a_d, m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&blas_b_d, k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&blas_c_d, m * n * sizeof(float)));

    for (int i = 0; i < m * k; i++) blas_a[i] = 1.0f;
    for (int i = 0; i < k * n; i++) blas_b[i] = 2.0f;
    for (int i = 0; i < m * n; i++) blas_c[i] = 0.0f;

    // Copy data to the device
    CUDA_CHECK(cudaMemcpy(blas_a_d, blas_a, m * k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(blas_b_d, blas_b, k * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(blas_c_d, blas_c, m * n * sizeof(float), cudaMemcpyHostToDevice));

    float alpha = 1.0f, beta = 0.0f;
    CUDA_CHECK(cudaEventRecord(blas_start));
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, blas_a_d, m, blas_b_d, k, &beta, blas_c_d, m);
    CUDA_CHECK(cudaEventRecord(blas_stop));
    CUDA_CHECK(cudaEventSynchronize(blas_stop));

    CUDA_CHECK(cudaDeviceSynchronize());

    float *blas_c_host;
    blas_c_host = (float*)malloc(m * n * sizeof(float));
    CUDA_CHECK(cudaMemcpy(blas_c_host, blas_c_d, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventElapsedTime(&wmma_eventMs, wmma_start, wmma_stop));
    CUDA_CHECK(cudaEventElapsedTime(&blas_eventMs, blas_start, blas_stop));
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
    cudaFree(blas_a_d);
    cudaFree(blas_b_d);
    cudaFree(blas_c_d);
    free(a);
    free(b);
    free(c);
    free(blas_a);
    free(blas_b);
    free(blas_c);
    free(blas_c_host);
    cublasDestroy(handle);

    return 0;
}
