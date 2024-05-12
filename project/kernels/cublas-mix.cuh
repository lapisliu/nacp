#pragma once

#include "util.h"

float run_cublas(int M, int N, int K) {
    half *a, *a_d;
    half *b, *b_d;
    float *c, *c_d; 

    // Initialize timing infra
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float eventMs = 1.0f;

    // Allocate host memory
    a = (half*)malloc(M * K * sizeof(half));
    b = (half*)malloc(K * N * sizeof(half));
    c = (float*)malloc(M * N * sizeof(float));

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&a_d, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&b_d, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&c_d, M * N * sizeof(float)));

    InitMatrix(M, N, K, a_blas, b_blas, c_blas);

    CUDA_CHECK(cudaMemcpy(a_d, a, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_d, b, K * N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c_d, c, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch the kernel
    cublasLtHandle_t handle;
    cublasLtCreate(&handle);

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
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
        CUBLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    float alpha = 1.0f, beta = 0.0f;
    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasLtMatmul(handle, operationDesc, &alpha, a_d, Adesc, b_d, Bdesc, &beta, c_d, Cdesc, c_d, Cdesc, &heuristicResult.algo, nullptr, 0, 0));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the result back to the host
    CUDA_CHECK(cudaMemcpy(c, c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventElapsedTime(&eventMs, start, stop));
    for (int i = 0; i < 5; i++) {
        std::cout << c_blas[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    free(a);
    free(b);
    free(c);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtDestroy(handle);

    return eventMs;
}