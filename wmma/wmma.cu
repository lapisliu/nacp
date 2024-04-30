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

constexpr int M = 40960;
constexpr int N = 40960;
constexpr int K = 40960;
constexpr int WARP_SIZE = 16;

__host__ void InitMatrix(half *A, half *B) {
    for (int i = 0; i < M * M; i++) {
        A[i] = __float2half(1.0f);
        B[i] = __float2half(2.0f);
    }
}

__global__ void matMulWMMA(half *a, half *b, float *c) {
    int blockRow = blockIdx.y * blockDim.y + threadIdx.y;
    int blockCol = blockIdx.x * blockIdx.x + threadIdx.x;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WARP_SIZE, WARP_SIZE, WARP_SIZE, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WARP_SIZE, WARP_SIZE, WARP_SIZE, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WARP_SIZE, WARP_SIZE, WARP_SIZE, float> c_frag;

    // Initialize the output fragment
    wmma::fill_fragment(c_frag, 0.0f);

    int row = threadIdx.y;
    int col = threadIdx.x;
    for (int k = 0; k < K / WARP_SIZE; k++) {
        if (blockRow < M && k * WARP_SIZE + col < K && blockCol < N && k * WARP_SIZE + row < K) {
            wmma::load_matrix_sync(a_frag, a, WARP_SIZE);
            wmma::load_matrix_sync(b_frag, b, WARP_SIZE);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            wmma::store_matrix_sync(c, c_frag, WARP_SIZE, wmma::mem_row_major);
        }
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
    dim3 threadsPerBlock(WARP_SIZE, WARP_SIZE);
    dim3 blocksPerGrid((N + WARP_SIZE - 1) / WARP_SIZE, (M + WARP_SIZE - 1) / WARP_SIZE);
    CUDA_CHECK(cudaEventRecord(wmma_start));
    matMulWMMA<<<blocksPerGrid, threadsPerBlock>>>(a_d, b_d, c_d);
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

    //cublas part
    half *a_blas, *a_d_blas;
    half *b_blas, *b_d_blas;
    float *c_blas, *c_d_blas;

    // Initialize timing infra
    cudaEvent_t blas_start, blas_stop;
    CUDA_CHECK(cudaEventCreate(&blas_start));
    CUDA_CHECK(cudaEventCreate(&blas_stop));
    float blas_eventMs = 1.0f;

    // Allocate host memory
    a_blas = (half*)malloc(M * K * sizeof(half));
    b_blas = (half*)malloc(K * N * sizeof(half));
    c_blas = (float*)malloc(M * N * sizeof(float));

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&a_d_blas, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&b_d_blas, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&c_d_blas, M * N * sizeof(float)));

    for (int i = 0; i < M * K; i++) a_blas[i] = 1.0f;
    for (int i = 0; i < K * N; i++) b_blas[i] = 2.0f;
    for (int i = 0; i < M * N; i++) c_blas[i] = 0.0f;

    CUDA_CHECK(cudaMemcpy(a_d_blas, a_blas, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_d_blas, b_blas, K * N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c_d_blas, c_blas, M * N * sizeof(float), cudaMemcpyHostToDevice));

    cublasLtHandle_t handle_blas;
    cublasLtCreate(&handle_blas);

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, M, K, M));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, K, N, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, N, M));

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;

    cublasLtMatmulDesc_t operationDesc;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F_FAST_16F, CUDA_R_32F));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    cublasLtMatmulPreference_t preference = NULL;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle_blas, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
        CUBLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    float alpha = 1.0f, beta = 0.0f;
    CUDA_CHECK(cudaEventRecord(blas_start));
    CUBLAS_CHECK(cublasLtMatmul(handle_blas, operationDesc, &alpha, a_d_blas, Adesc, b_d_blas, Bdesc, &beta, c_d_blas, Cdesc, c_d_blas, Cdesc, &heuristicResult.algo, nullptr, 0, 0));
    CUDA_CHECK(cudaEventRecord(blas_stop));
    CUDA_CHECK(cudaEventSynchronize(blas_stop));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(c_blas, c_d_blas, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventElapsedTime(&blas_eventMs, blas_start, blas_stop));
    std::cout << "cublas: " << blas_eventMs << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << c_blas[i] << " ";
    }
    std::cout << std::endl;

    free(a_blas);
    free(b_blas);
    free(c_blas);
    cudaFree(a_d_blas);
    cudaFree(b_d_blas);
    cudaFree(c_d_blas);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtDestroy(handle_blas);

    return 0;
}

