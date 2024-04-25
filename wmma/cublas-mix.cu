#include <cublasLt.h>
#include <iostream>

#include "cublas_v2.h"

#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at line " << __LINE__ << std::endl; \
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

void LtHalfMatmul(cublasLtHandle_t handle,
		int m,
		int n,
		int k,
		const float *alpha,
		// const float *a_scale,
		const __half *A,
		int lda,
		// const float *b_scale,
		const __half *B,
		int ldb,
		// const float *c_scale,
		float *D,
		int ldc,
		// const float *d_scale,
		// float *amax_d,
		void *workspace,
		size_t workspaceSize) {
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    float beta = 0.0f;

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_16F));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

//    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
//    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));
//    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, &c_scale, sizeof(c_scale)));
//    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_scale, sizeof(d_scale)));
//    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &amax_d, sizeof(amax_d)));

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, m, k, lda));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, k, n, ldb));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_32F, m, n, ldc));

    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1, &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
        CUBLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    CUBLAS_CHECK(cublasLtMatmul(handle,
		operationDesc,
		alpha,
		A,
		Adesc,
		B,
		Bdesc,
		&beta,
		nullptr,
		Cdesc,
		D,
		Ddesc,
		&heuristicResult.algo,
		workspace,
		workspaceSize,
		0));

    if (Cdesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) CUBLAS_CHECK(cublasLtMatmulDescDestroy(operationDesc));
}


int main() {
    half *a, *a_d;
    half *b, *b_d;
    float *c, *c_d;

    int m = 4096;
    int n = 4096;
    int k = 4096;

    // Initialize timing infra
    cudaEvent_t blas_start, blas_stop;
    CUDA_CHECK(cudaEventCreate(&blas_start));
    CUDA_CHECK(cudaEventCreate(&blas_stop));
    float blas_eventMs = 1.0f;

    // Allocate host memory
    a = (half*)malloc(m * k * sizeof(half));
    b = (half*)malloc(k * n * sizeof(half));
    c = (float*)malloc(m * n * sizeof(float));

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&a_d, m * k * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&b_d, k * n * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&c_d, m * n * sizeof(float)));

    for (int i = 0; i < m * k; i++) a[i] = 1.0f;
    for (int i = 0; i < k * n; i++) b[i] = 2.0f;
    for (int i = 0; i < m * n; i++) c[i] = 0.0f;

    CUDA_CHECK(cudaMemcpy(a_d, a, m * k * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_d, b, k * n * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c_d, c, m * n * sizeof(half), cudaMemcpyHostToDevice));

    cublasLtHandle_t handle;
    cublasLtCreate(&handle);

    // Create workspace
    void *workspace;
    CUDA_CHECK(cudaMalloc((void**)&workspace, 4194304));

    float alpha = 1.0f;
    CUDA_CHECK(cudaEventRecord(blas_start));
    LtHalfMatmul(handle, m, n, k, &alpha, a, m, b, k, c, m, workspace, 4194304);
    CUDA_CHECK(cudaEventRecord(blas_stop));
    CUDA_CHECK(cudaEventSynchronize(blas_stop));

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(c, c_d, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventElapsedTime(&blas_eventMs, blas_start, blas_stop));
    std::cout << "cublas: " << blas_eventMs << std::endl;
    for (int i = 0; i < 5; i++) {
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
    cublasLtDestroy(handle);

    return 0;
}
