#pragma once

#include <cuda.h>
#include <mma.h>
#include <cublasLt.h>
#include <iostream>

#include "cuda_runtime.h"
#include "cublas_v2.h"

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

__host__ void InitMatrix(int M, int N, int K, half *A, half *B, float *C) {
    for (int i = 0; i < M * K; i++) A[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; i++) B[i] = __float2half(2.0f);
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;
}

__host__ void InitMatrix(int M, int N, int K, float *A, float *B, float *C) {
    for (int i = 0; i < M * K; i++) A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) B[i] = 2.0f;
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;
}