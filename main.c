#include "include/gemm.h"
#include "include/gemv.h"
#include "include/spmm.h"
#include "include/spmv.h"

int main(int argc, char *argv[]) {
  char *gpu_enabled_str = (GPU_ENABLED) ? "True" : "False";
  printf("GPU BLAS Offloading Benchmark:\n");
  printf("\tIterations per Kernel: %d\n", ITERATIONS);
  printf("\tMax Problem Dimension: %d\n", UPPER_LIMIT);
  printf("\tCPU Tests Enabled: True\n");
  printf("\tCPU Library: %s\n", CPU_LIB_NAME);
  printf("\tGPU Tests Enabled: %s\n", gpu_enabled_str);
  printf("\tGPU Library: %s\n", GPU_LIB_NAME);

  // GEMM Comparison - Square
  for (uint64_t dim = 1; dim <= UPPER_LIMIT; dim++) {
    uint64_t M = dim, N = dim, K = dim;
    // Perform CPU
    struct dT_Times cpuTimes = {gemm_cpu(_fp16_, ITERATIONS, M, N, K),
                                gemm_cpu(_fp32_, ITERATIONS, M, N, K),
                                gemm_cpu(_fp64_, ITERATIONS, M, N, K)};
    // Perform GPU
    struct dT_Times gpuTimes_OffloadOnce = {
        gemm_gpu(_fp16_, ITERATIONS, M, N, K, true),
        gemm_gpu(_fp32_, ITERATIONS, M, N, K, true),
        gemm_gpu(_fp64_, ITERATIONS, M, N, K, true)};
    struct dT_Times gpuTimes_OffloadAlways = {
        gemm_gpu(_fp16_, ITERATIONS, M, N, K, false),
        gemm_gpu(_fp32_, ITERATIONS, M, N, K, false),
        gemm_gpu(_fp64_, ITERATIONS, M, N, K, false)};

    // Compare results: when GPU > CPU record value

    // Save values to CSV (dimensions, time taken, GFLOP/s, etc)

    // Increment Dimensions
  }

  return 0;
}