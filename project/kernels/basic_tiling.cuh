#pragma once

#include "util.h"

constexpr int TILE_SIZE = 16;
//Index is a MACRO that returns the element of t tensor at row, col coordinate
#define Index(t, row, col, stride_h, stride_w) (((t)[(row) * (stride_h) + (col) * (stride_w)]))

__global__ void op_mm_kernel(float *a, float *b, float *c, int M, int N, int K) {
    int blockRow = blockIdx.y * blockDim.y + threadIdx.y;
    int blockCol = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    float r = 0;
    int row = threadIdx.y;
    int col = threadIdx.x;
    for (int k = 0; k < (K + TILE_SIZE - 1) / TILE_SIZE; k++) {
	    if (blockRow < M && k * TILE_SIZE + col < K)
	        As[row][col] = Index(a, blockRow, k * TILE_SIZE + col, K, 1);
	    else
            As[row][col] = 0;
	    if (k * TILE_SIZE + row < K && blockCol < N)
            Bs[row][col] = Index(b, k * TILE_SIZE + row, blockCol, N, 1);
	    else
            Bs[row][col] = 0;
        __syncthreads();
        for (int e = 0; e < TILE_SIZE; e++)
            r += As[row][e] * Bs[e][col];
        __syncthreads();
    }
    if (blockRow < M && blockCol < N)
        Index(c, blockRow, blockCol, N, 1) = r;
}

float run_basic_tiling(int M, int N, int K) {
    float *a, *a_d;
    float *b, *b_d;
    float *c, *c_d; 

    // Initialize timing infra
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float eventMs = 1.0f;

    // Allocate host memory
    a = (float*)malloc(M * K * sizeof(float));
    b = (float*)malloc(K * N * sizeof(float));
    c = (float*)malloc(M * N * sizeof(float));

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&a_d, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&b_d, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&c_d, M * N * sizeof(float)));

    InitMatrix(M, N, K, a, b, c);

    CUDA_CHECK(cudaMemcpy(a_d, a, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_d, b, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c_d, c, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch the kernel
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    CUDA_CHECK(cudaEventRecord(start));
    op_mm_kernel<<<dimGrid, dimBlock>>>(a_d, b_d, c_d, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    cudaDeviceSynchronize();

    // Copy the result back to the host
    CUDA_CHECK(cudaMemcpy(c, c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventElapsedTime(&eventMs, start, stop));
    for (int i = 0; i < 5; i ++) {
        std::cout << c[i] << " ";
    }

    // Cleanup
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    free(a);
    free(b);
    free(c);

    return eventMs;
}