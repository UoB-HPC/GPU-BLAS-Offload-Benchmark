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