#include "include/gemm.h"
#include "include/gemv.h"
#include "include/spmm.h"
#include "include/spmv.h"

int main(int argc, char *argv[]) {
  char *gpu_enabled_str = (GPU_ENABLED) ? "True" : "False";
  unsigned int omp_threads = (getenv("OMP_NUM_THREADS") != NULL) ? atoi(getenv("OMP_NUM_THREADS")) : 1;
  const char* omp_proc_bind = (getenv("OMP_PROC_BIND") != NULL) ? getenv("OMP_PROC_BIND") : "Not Set";
  const char* omp_places = (getenv("OMP_PLACES") != NULL) ? getenv("OMP_PLACES") : "Not Set";
  printf("\nGPU BLAS Offloading Benchmark:\n");
  printf("\tIterations per Kernel: %d\n", ITERATIONS);
  printf("\tMax Problem Dimension: %d\n", UPPER_LIMIT);
  printf("\tCPU Tests Enabled: True\n");
  printf("\tCPU Library: %s\n", CPU_LIB_NAME);
  printf("\tGPU Tests Enabled: %s\n", gpu_enabled_str);
  printf("\tGPU Library: %s\n", GPU_LIB_NAME);
  printf("\tOMP_NUM_THREADS: %u\n", omp_threads);
  printf("\tOMP_PROC_BIND: %s\n", omp_proc_bind);
  printf("\tOMP_PLACES: %s\n", omp_places);
  printf("\n\n");

  // SGEMM Comparison - Square
  printf("Comparing SGEMM Kernels...\n");
  fprintf(stderr, "M, N, K, Kernel, seconds, GFLOP/s\n");
  for (uint64_t dim = 1; dim <= UPPER_LIMIT; dim*=2) {
    uint64_t M = dim, N = dim, K = dim;
    // Perform CPU
    double cpuTime = gemm_cpu(_fp32_, ITERATIONS, M, N, K);
    fprintf(stderr, "%lu, %lu, %lu, SGEMM, %f, %f\n", M, N, K, cpuTime, calcGflops(gemmFlops(M, N, K), cpuTime));
    // Perform GPU
    // gemm_gpu(_fp32_, ITERATIONS, M, N, K, true);
    // gemm_gpu(_fp32_, ITERATIONS, M, N, K, false);
  }
  printf("Done!\n");

  // DGEMM Comparison - Square
  printf("Comparing DGEMM Kernels...\n");
  fprintf(stderr, "M, N, K, Kernel, seconds, GFLOP/s\n");
  for (uint64_t dim = 1; dim <= UPPER_LIMIT; dim*=2) {
    uint64_t M = dim, N = dim, K = dim;
    // Perform CPU
    double cpuTime = gemm_cpu(_fp64_, ITERATIONS, M, N, K);
    fprintf(stderr, "%lu, %lu, %lu, DGEMM, %f, %f\n", M, N, K, cpuTime, calcGflops(gemmFlops(M, N, K), cpuTime));
    // Perform GPU
    // gemm_gpu(_fp64_, ITERATIONS, M, N, K, true);
    // gemm_gpu(_fp64_, ITERATIONS, M, N, K, false);
  }
  printf("Done!\n");


  return 0;
}