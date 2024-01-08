#include "main.h"

int main(int argc, char *argv[]) {
  printBenchmarkConfig();

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
    const uint64_t M = dim, N = dim, K = dim;
    const uint64_t gemmProbSize = (M * K) + (K * N) + (M * N);
    // Perform CPU
    double cpuTime = gemm_cpu(_fp32_, ITERATIONS, M, N, K);
    writeLineToCsv(fptr, "cpu", "sgemm", M, N, K, calcKib(gemmProbSize, 4),
                   ITERATIONS, cpuTime,
                   calcGflops(gemmFlops(M, N, K), ITERATIONS, cpuTime));

    // Perform GPU
    // - Offload to/from GPU once before all iterations and once after
    double gpuTime_once = gemm_gpu(_fp32_, ITERATIONS, M, N, K, true);
    writeLineToCsv(fptr, "gpu_offloadOnce", "sgemm", M, N, K,
                   calcKib(gemmProbSize, 4), ITERATIONS, gpuTime_once,
                   calcGflops(gemmFlops(M, N, K), ITERATIONS, gpuTime_once));
    // - Offload to/from GPU every iteration
    double gpuTime_every = gemm_gpu(_fp32_, ITERATIONS, M, N, K, false);
    writeLineToCsv(fptr, "gpu_offloadAlways", "sgemm", M, N, K,
                   calcKib(gemmProbSize, 4), ITERATIONS, gpuTime_every,
                   calcGflops(gemmFlops(M, N, K), ITERATIONS, gpuTime_every));
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
    // - Offload to/from GPU once before all iterations and once after
    double gpuTime_once = gemm_gpu(_fp64_, ITERATIONS, M, N, K, true);
    writeLineToCsv(fptr, "gpu_offloadOnce", "dgemm", M, N, K,
                   calcKib(gemmProbSize, 8), ITERATIONS, gpuTime_once,
                   calcGflops(gemmFlops(M, N, K), ITERATIONS, gpuTime_once));
    // - Offload to/from GPU every iteration
    double gpuTime_every = gemm_gpu(_fp64_, ITERATIONS, M, N, K, false);
    writeLineToCsv(fptr, "gpu_offloadAlways", "dgemm", M, N, K,
                   calcKib(gemmProbSize, 8), ITERATIONS, gpuTime_every,
                   calcGflops(gemmFlops(M, N, K), ITERATIONS, gpuTime_every));
  }
  printf("Finished!\n");

  return 0;
}