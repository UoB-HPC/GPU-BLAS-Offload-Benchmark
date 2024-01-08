#include "main.h"

int iters = 10;
int upperLimit = 128;

int parseInt(const char *str);
void getParameters(int argc, char *argv[]);

int main(int argc, char *argv[]) {
  getParameters(argc, argv);
  printBenchmarkConfig(iters, upperLimit);

  // Ensure CSV file directory exists.
  struct stat st = {0};
  if (stat(CSV_DIR, &st) == -1) {
    mkdir(CSV_DIR, 0700);
  }

  // SGEMM Comparison - Square
  printf("Comparing SGEMM Kernels...\n");
  FILE *fptr;
  fptr = newCSV(CSV_DIR "/sgemm_square.csv");
  for (int dim = 1; dim <= upperLimit; dim++) {
    const int M = dim, N = dim, K = dim;
    const uint64_t gemmProbSize = (M * K) + (K * N) + (M * N);
    // Perform CPU
    double cpuTime = gemm_cpu(_fp32_, iters, M, N, K);
    writeLineToCsv(fptr, "cpu", "sgemm", M, N, K, calcKib(gemmProbSize, 4),
                   iters, cpuTime,
                   calcGflops(gemmFlops(M, N, K), iters, cpuTime));

    // Perform GPU
    // - Offload to/from GPU once before all iterations and once after
    double gpuTime_once = gemm_gpu(_fp32_, iters, M, N, K, true);
    writeLineToCsv(fptr, "gpu_offloadOnce", "sgemm", M, N, K,
                   calcKib(gemmProbSize, 4), iters, gpuTime_once,
                   calcGflops(gemmFlops(M, N, K), iters, gpuTime_once));
    // - Offload to/from GPU every iteration
    double gpuTime_every = gemm_gpu(_fp32_, iters, M, N, K, false);
    writeLineToCsv(fptr, "gpu_offloadAlways", "sgemm", M, N, K,
                   calcKib(gemmProbSize, 4), iters, gpuTime_every,
                   calcGflops(gemmFlops(M, N, K), iters, gpuTime_every));
  }
  // Close file
  fclose(fptr);
  printf("Finished!\n");

  // DGEMM Comparison - Square
  printf("Comparing DGEMM Kernels...\n");
  fptr = newCSV(CSV_DIR "/dgemm_square.csv");
  for (int dim = 1; dim <= upperLimit; dim++) {
    int M = dim, N = dim, K = dim;
    // Perform CPU
    double cpuTime = gemm_cpu(_fp64_, iters, M, N, K);
    uint64_t gemmProbSize = (M * K) + (K * N) + (M * N);
    writeLineToCsv(fptr, "cpu", "dgemm", M, N, K, calcKib(gemmProbSize, 8),
                   iters, cpuTime,
                   calcGflops(gemmFlops(M, N, K), iters, cpuTime));
    // Perform GPU
    // - Offload to/from GPU once before all iterations and once after
    double gpuTime_once = gemm_gpu(_fp64_, iters, M, N, K, true);
    writeLineToCsv(fptr, "gpu_offloadOnce", "dgemm", M, N, K,
                   calcKib(gemmProbSize, 8), iters, gpuTime_once,
                   calcGflops(gemmFlops(M, N, K), iters, gpuTime_once));
    // - Offload to/from GPU every iteration
    double gpuTime_every = gemm_gpu(_fp64_, iters, M, N, K, false);
    writeLineToCsv(fptr, "gpu_offloadAlways", "dgemm", M, N, K,
                   calcKib(gemmProbSize, 8), iters, gpuTime_every,
                   calcGflops(gemmFlops(M, N, K), iters, gpuTime_every));
  }
  printf("Finished!\n");

  return 0;
}

int parseInt(const char *str) {
  char *next;
  int value = strtoul(str, &next, 10);
  return strlen(next) ? -1 : value;
}

void getParameters(int argc, char *argv[]) {
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--iterations") || !strcmp(argv[i], "-i")) {
      if (++i >= argc || (iters = parseInt(argv[i])) < 0) {
        printf("ERROR - Invalid number of iterations\n");
        exit(1);
      }
    } else if (!strcmp(argv[i], "--dimension_limit") ||
               !strcmp(argv[i], "-d")) {
      if (++i >= argc || (upperLimit = parseInt(argv[i])) < 0) {
        printf("ERROR - Invalid dimension limit\n");
        exit(1);
      }
    } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
      printf("\n");
      printf("Usage: ./gpu-blob [OPTIONS]\n\n");
      printf("Options:\n");
      printf("  -h  --help                   Print this message\n");
      printf("  -i  --iterations I           Repeat each kernel I times "
             "(default: %d)\n",
             iters);
      printf("  -d  --dimension_limit D      Max value of M, N, K is D "
             "(default: %d)\n",
             upperLimit);
      printf("\n");
      exit(0);
    } else {
      printf("Unrecognized argument '%s' (try '--help')\n", argv[i]);
      exit(1);
    }
  }
}