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

  // Perform all CPU kernels & record results

  // Perform all GPU kernels & record results
  if (GPU_ENABLED) {
  }

  return 0;
}