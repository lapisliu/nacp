#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/gemm.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <iostream>

//  using Gemm = cutlass::gemm::device::Gemm<
//    cutlass::half_t,                           // ElementA
//    cutlass::layout::ColumnMajor,              // LayoutA
//    cutlass::half_t,                           // ElementB
//    cutlass::layout::ColumnMajor,              // LayoutB
//    float,                                     // ElementOutput
//    cutlass::layout::ColumnMajor,              // LayoutOutput
//    float,                                     // ElementAccumulator
//    cutlass::arch::OpClassTensorOp,            // tag indicating Tensor Cores
//    cutlass::arch::Sm70                        // tag indicating target GPU compute architecture
//  >;

cudaError_t cutlass_sgemm_nn(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {
  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible compositions
  // including the following example for single-precision GEMM. Typical values are used as
  // default template arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see `cutlass/gemm/device/gemm.h`
  using ColumnMajor = cutlass::layout::ColumnMajor;
  using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                  ColumnMajor,  // Layout of A matrix
                                                  float,        // Data-type of B matrix
                                                  ColumnMajor,  // Layout of B matrix
                                                  float,        // Data-type of C matrix
                                                  ColumnMajor,  // Layout of C matrix
                                                  float,
                                                  cutlass::arch::OpClassTensorOp,
                                                  cutlass::arch::Sm70,
                                                  cutlass::gemm::GemmShape<128, 128, 32>,
                                                  cutlass::gemm::GemmShape<64, 64, 32>,
                                                  cutlass::gemm::GemmShape<8, 8, 4>,
                                                  cutlass::epilogue::threado>;
  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;
  // Construct the CUTLASS GEMM arguments object.
  //
  // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
  // in host code and passed to kernels by value. These may include pointers, strides, scalars,
  // and other arguments needed by Gemm and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
  // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
  //
  CutlassGemm::Arguments args({M, N, K},   // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              {B, ldb},    // Tensor-ref for source matrix B
                              {C, ldc},    // Tensor-ref for source matrix C
                              {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {alpha, beta}); // Scalars used in the Epilogue
  //
  // Launch the CUTLASS GEMM kernel.
  //
  cutlass::Status status = gemm_operator(args);
  //
  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
  //
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  // Return success, if no errors were encountered.
  return cudaSuccess;
}

int main() {
    // Dimensions of the GEMM problem
    int m = 4096;
    int n = 4096;
    int k = 4096;

    float *a, *a_d;
    float *b, *b_d;
    float *c, *c_d;

    a = (float*)malloc(m * k * sizeof(float));
    b = (float*)malloc(k * n * sizeof(float));
    c = (float*)malloc(m * n * sizeof(float));

    cudaMalloc((void**)&a_d, m * k * sizeof(float));
    cudaMalloc((void**)&b_d, k * n * sizeof(float));
    cudaMalloc((void**)&c_d, m * n * sizeof(float));

    for (int i = 0; i < m * k; i++) a[i] = 1.0f;
    for (int i = 0; i < k * n; i++) b[i] = 2.0f;
    for (int i = 0; i < m * n; i++) c[i] = 0.0f;

    cudaMemcpy(a_d, a, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c, m * n * sizeof(float), cudaMemcpyHostToDevice);

    cudaError_t e = cutlass_sgemm_nn(m, n, k, 1.0f, a_d, m, b_d, k, 0.0f, c_d, m);

//    // Check if the configuration is supported
//    cutlass::Status status = gemm_op.can_implement(args);
//    if (status != cutlass::Status::kSuccess) {
//        std::cerr << "This problem size is not supported!" << std::endl;
//        return -1;
//    }
//
//    // Run the GEMM operation
//    status = gemm_op(args);
//    if (status != cutlass::Status::kSuccess) {
//        std::cerr << "Failed to perform GEMM operation!" << std::endl;
//        return -1;
//    }

    cudaMemcpy(c, c_d, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 5; i++) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

