#include "main.h"

int main(int argc, char *argv[]) {
  char *gpu_enabled_str = (GPU_ENABLED) ? "True" : "False";
  unsigned int omp_threads =
      (getenv("OMP_NUM_THREADS") != NULL) ? atoi(getenv("OMP_NUM_THREADS")) : 1;
  const char *omp_proc_bind =
      (getenv("OMP_PROC_BIND") != NULL) ? getenv("OMP_PROC_BIND") : "Not Set";
  const char *omp_places =
      (getenv("OMP_PLACES") != NULL) ? getenv("OMP_PLACES") : "Not Set";
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

  // Ensure CSV file directory exists.
  struct stat st = {0};
  if (stat(CSV_DIR, &st) == -1) {
    mkdir(CSV_DIR, 0700);
  }

  // SGEMM Comparison - Square
  printf("Comparing SGEMM Kernels...\n");
  FILE *fptr;
  fptr = newCSV(CSV_DIR "/sgemm_square.csv");
  for (uint64_t dim = 1; dim <= UPPER_LIMIT; dim++) {
    uint64_t M = dim, N = dim, K = dim;
    // Perform CPU
    double cpuTime = gemm_cpu(_fp32_, ITERATIONS, M, N, K);
    uint64_t gemmProbSize = (M * K) + (K * N) + (M * N);
    writeLineToCsv(fptr, "cpu", "sgemm", M, N, K, calcKib(gemmProbSize, 4),
                   ITERATIONS, cpuTime,
                   calcGflops(gemmFlops(M, N, K), ITERATIONS, cpuTime));

    // Perform GPU
    // gemm_gpu(_fp32_, ITERATIONS, M, N, K, true);
    // gemm_gpu(_fp32_, ITERATIONS, M, N, K, false);
  }
  // Close file
  fclose(fptr);
  printf("Finished!\n");

  // DGEMM Comparison - Square
  printf("Comparing DGEMM Kernels...\n");
  fptr = newCSV(CSV_DIR "/dgemm_square.csv");
  for (uint64_t dim = 1; dim <= UPPER_LIMIT; dim++) {
    uint64_t M = dim, N = dim, K = dim;
    // Perform CPU
    double cpuTime = gemm_cpu(_fp64_, ITERATIONS, M, N, K);
    uint64_t gemmProbSize = (M * K) + (K * N) + (M * N);
    writeLineToCsv(fptr, "cpu", "dgemm", M, N, K, calcKib(gemmProbSize, 8),
                   ITERATIONS, cpuTime,
                   calcGflops(gemmFlops(M, N, K), ITERATIONS, cpuTime));
    // Perform GPU
    // gemm_gpu(_fp64_, ITERATIONS, M, N, K, true);
    // gemm_gpu(_fp64_, ITERATIONS, M, N, K, false);
  }
  printf("Done!\n");

  return 0;
}