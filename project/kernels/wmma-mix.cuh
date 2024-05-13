#pragma once

#include "util.h"

using namespace nvcuda;
constexpr int WARP_SIZE = 16;

__global__ void matMulWMMA(half *a, half *b, float *c, int M, int N, int K, float alpha, float beta) {
    // int blockRow = blockIdx.y * blockDim.y + threadIdx.y;
    // int blockCol = blockIdx.x * blockIdx.x + threadIdx.x;

    // // Declare the fragments
    // wmma::fragment<wmma::matrix_a, WARP_SIZE, WARP_SIZE, WARP_SIZE, half, wmma::col_major> a_frag;
    // wmma::fragment<wmma::matrix_b, WARP_SIZE, WARP_SIZE, WARP_SIZE, half, wmma::row_major> b_frag;
    // wmma::fragment<wmma::accumulator, WARP_SIZE, WARP_SIZE, WARP_SIZE, float> c_frag;

    // // Initialize the output fragment
    // wmma::fill_fragment(c_frag, 0.0f);

    // int row = threadIdx.y;
    // int col = threadIdx.x;
    // for (int k = 0; k < K / WARP_SIZE; k++) {
    //     if (blockRow < M && k * WARP_SIZE + col < K && blockCol < N && k * WARP_SIZE + row < K) {
    //         wmma::load_matrix_sync(a_frag, a, WARP_SIZE);
    //         wmma::load_matrix_sync(b_frag, b, WARP_SIZE);
    //         wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    //         wmma::store_matrix_sync(c, c_frag, WARP_SIZE, wmma::mem_row_major);
    //     }
    // }
    // Leading dimensions. Packed with no transpositions.
    int lda = M;
    int ldb = K;
    int ldc = M;
    
    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(acc_frag, 0.0f);
    // Loop over the K-dimension
    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;
        
        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
            wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
    // Load in current value of c, scale by beta, and add to result scaled by alpha
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    
    if (cRow < M && cCol < N) {
        wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);
        
        for(int i=0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }
        // Store the output
        wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
    }
}

float run_wmma(int M, int N, int K) {
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

    InitMatrix(M, N, K, a, b, c);

    CUDA_CHECK(cudaMemcpy(a_d, a, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_d, b, K * N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c_d, c, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch the kernel
    dim3 threadsPerBlock(WARP_SIZE, WARP_SIZE);
    dim3 blocksPerGrid((N + WARP_SIZE - 1) / WARP_SIZE, (M + WARP_SIZE - 1) / WARP_SIZE);
    CUDA_CHECK(cudaEventRecord(start));
    matMulWMMA<<<blocksPerGrid, threadsPerBlock>>>(a_d, b_d, c_d, M, N, K, 1.0f, 0.0f);
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
