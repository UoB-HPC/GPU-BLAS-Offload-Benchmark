#include "../include/main.h"

int iters = 10;
int upperLimit = 128;
FILE *fptr;

int main(int argc, char *argv[]) {
  getParameters(argc, argv);
  printBenchmarkConfig(iters, upperLimit);

  // Ensure CSV file directory exists.
  struct stat st = {0};
  if (stat(CSV_DIR, &st) == -1) {
    mkdir(CSV_DIR, 0700);
  }

  // SGEMM Comparison
  printf("Comparing SGEMM Kernels:\n");
  printf("\tSquare Problem Sizes...\n");
  fptr = newCSV(CSV_DIR "/sgemm_square.csv");
  for (int dim = 1; dim <= upperLimit; dim++) {
    const int M = dim, N = dim, K = dim;
    const uint64_t gemmProbSize = gemmKib(_fp32_, M, N, K);
    // Perform CPU
    double cpuTime = gemm_cpu(_fp32_, iters, M, N, K);
    writeLineToCsv(fptr, "cpu", "sgemm", M, N, K, gemmProbSize, iters, cpuTime,
                   calcGflops(gemmFlops(M, N, K), iters, cpuTime));

    // Perform GPU
    // - Offload to/from GPU once before all iterations and once after
    double gpuTime_once = gemm_gpu(_fp32_, iters, M, N, K, true);
    writeLineToCsv(fptr, "gpu_offloadOnce", "sgemm", M, N, K, gemmProbSize,
                   iters, gpuTime_once,
                   calcGflops(gemmFlops(M, N, K), iters, gpuTime_once));
    // - Offload to/from GPU every iteration
    double gpuTime_every = gemm_gpu(_fp32_, iters, M, N, K, false);
    writeLineToCsv(fptr, "gpu_offloadAlways", "sgemm", M, N, K, gemmProbSize,
                   iters, gpuTime_every,
                   calcGflops(gemmFlops(M, N, K), iters, gpuTime_every));
  }
  // Close file
  fclose(fptr);

  printf("\tRectangular Problem Sizes:\n");
  printf("\t\tTall and thin (16M x K)...\n");
  fptr = newCSV(CSV_DIR "/sgemm_rectangular_16MxK.csv");
  for (int dim = 16; dim <= upperLimit; dim += 16) {
    const int M = dim, N = dim, K = (dim / 16);
    const uint64_t gemmProbSize = gemmKib(_fp32_, M, N, K);
    // Perform CPU
    double cpuTime = gemm_cpu(_fp32_, iters, M, N, K);
    writeLineToCsv(fptr, "cpu", "sgemm", M, N, K, gemmProbSize, iters, cpuTime,
                   calcGflops(gemmFlops(M, N, K), iters, cpuTime));

    // Perform GPU
    // - Offload to/from GPU once before all iterations and once after
    double gpuTime_once = gemm_gpu(_fp32_, iters, M, N, K, true);
    writeLineToCsv(fptr, "gpu_offloadOnce", "sgemm", M, N, K, gemmProbSize,
                   iters, gpuTime_once,
                   calcGflops(gemmFlops(M, N, K), iters, gpuTime_once));
    // - Offload to/from GPU every iteration
    double gpuTime_every = gemm_gpu(_fp32_, iters, M, N, K, false);
    writeLineToCsv(fptr, "gpu_offloadAlways", "sgemm", M, N, K, gemmProbSize,
                   iters, gpuTime_every,
                   calcGflops(gemmFlops(M, N, K), iters, gpuTime_every));
  }
  // Close file
  fclose(fptr);

  printf("\t\tTall and thin (M x 32)...\n");
  fptr = newCSV(CSV_DIR "/sgemm_rectangular_Mx32.csv");
  for (int dim = 1; dim <= upperLimit; dim++) {
    const int M = dim, N = dim, K = 32;
    const uint64_t gemmProbSize = gemmKib(_fp32_, M, N, K);
    // Perform CPU
    double cpuTime = gemm_cpu(_fp32_, iters, M, N, K);
    writeLineToCsv(fptr, "cpu", "sgemm", M, N, K, gemmProbSize, iters, cpuTime,
                   calcGflops(gemmFlops(M, N, K), iters, cpuTime));

    // Perform GPU
    // - Offload to/from GPU once before all iterations and once after
    double gpuTime_once = gemm_gpu(_fp32_, iters, M, N, K, true);
    writeLineToCsv(fptr, "gpu_offloadOnce", "sgemm", M, N, K, gemmProbSize,
                   iters, gpuTime_once,
                   calcGflops(gemmFlops(M, N, K), iters, gpuTime_once));
    // - Offload to/from GPU every iteration
    double gpuTime_every = gemm_gpu(_fp32_, iters, M, N, K, false);
    writeLineToCsv(fptr, "gpu_offloadAlways", "sgemm", M, N, K, gemmProbSize,
                   iters, gpuTime_every,
                   calcGflops(gemmFlops(M, N, K), iters, gpuTime_every));
  }
  // Close file
  fclose(fptr);

  printf("\t\tShort and wide (M x 16K)...\n");
  fptr = newCSV(CSV_DIR "/sgemm_rectangular_Mx16K.csv");
  for (int dim = 16; dim <= upperLimit; dim += 16) {
    const int M = (dim / 16), N = (dim / 16), K = dim;
    const uint64_t gemmProbSize = gemmKib(_fp32_, M, N, K);
    // Perform CPU
    double cpuTime = gemm_cpu(_fp32_, iters, M, N, K);
    writeLineToCsv(fptr, "cpu", "sgemm", M, N, K, gemmProbSize, iters, cpuTime,
                   calcGflops(gemmFlops(M, N, K), iters, cpuTime));

    // Perform GPU
    // - Offload to/from GPU once before all iterations and once after
    double gpuTime_once = gemm_gpu(_fp32_, iters, M, N, K, true);
    writeLineToCsv(fptr, "gpu_offloadOnce", "sgemm", M, N, K, gemmProbSize,
                   iters, gpuTime_once,
                   calcGflops(gemmFlops(M, N, K), iters, gpuTime_once));
    // - Offload to/from GPU every iteration
    double gpuTime_every = gemm_gpu(_fp32_, iters, M, N, K, false);
    writeLineToCsv(fptr, "gpu_offloadAlways", "sgemm", M, N, K, gemmProbSize,
                   iters, gpuTime_every,
                   calcGflops(gemmFlops(M, N, K), iters, gpuTime_every));
  }
  // Close file
  fclose(fptr);

  printf("\t\tShort and wide (32 x K)...\n");
  fptr = newCSV(CSV_DIR "/sgemm_rectangular_32xK.csv");
  for (int dim = 1; dim <= upperLimit; dim++) {
    const int M = 32, N = 32, K = dim;
    const uint64_t gemmProbSize = gemmKib(_fp32_, M, N, K);
    // Perform CPU
    double cpuTime = gemm_cpu(_fp32_, iters, M, N, K);
    writeLineToCsv(fptr, "cpu", "sgemm", M, N, K, gemmProbSize, iters, cpuTime,
                   calcGflops(gemmFlops(M, N, K), iters, cpuTime));

    // Perform GPU
    // - Offload to/from GPU once before all iterations and once after
    double gpuTime_once = gemm_gpu(_fp32_, iters, M, N, K, true);
    writeLineToCsv(fptr, "gpu_offloadOnce", "sgemm", M, N, K, gemmProbSize,
                   iters, gpuTime_once,
                   calcGflops(gemmFlops(M, N, K), iters, gpuTime_once));
    // - Offload to/from GPU every iteration
    double gpuTime_every = gemm_gpu(_fp32_, iters, M, N, K, false);
    writeLineToCsv(fptr, "gpu_offloadAlways", "sgemm", M, N, K, gemmProbSize,
                   iters, gpuTime_every,
                   calcGflops(gemmFlops(M, N, K), iters, gpuTime_every));
  }
  // Close file
  fclose(fptr);
  printf("Finished!\n");

  // DGEMM Comparison - Square
  printf("Comparing DGEMM Kernels:\n");
  printf("\tSquare Problem Sizes...\n");
  fptr = newCSV(CSV_DIR "/dgemm_square.csv");
  for (int dim = 1; dim <= upperLimit; dim++) {
    int M = dim, N = dim, K = dim;
    // Perform CPU
    double cpuTime = gemm_cpu(_fp64_, iters, M, N, K);
    uint64_t gemmProbSize = gemmKib(_fp32_, M, N, K);
    writeLineToCsv(fptr, "cpu", "dgemm", M, N, K, gemmProbSize, iters, cpuTime,
                   calcGflops(gemmFlops(M, N, K), iters, cpuTime));
    // Perform GPU
    // - Offload to/from GPU once before all iterations and once after
    double gpuTime_once = gemm_gpu(_fp64_, iters, M, N, K, true);
    writeLineToCsv(fptr, "gpu_offloadOnce", "dgemm", M, N, K, gemmProbSize,
                   iters, gpuTime_once,
                   calcGflops(gemmFlops(M, N, K), iters, gpuTime_once));
    // - Offload to/from GPU every iteration
    double gpuTime_every = gemm_gpu(_fp64_, iters, M, N, K, false);
    writeLineToCsv(fptr, "gpu_offloadAlways", "dgemm", M, N, K, gemmProbSize,
                   iters, gpuTime_every,
                   calcGflops(gemmFlops(M, N, K), iters, gpuTime_every));
  }
  printf("Finished!\n");

  return 0;
}

double calcGflops(const uint64_t flops, const int iters, const double seconds) {
  return (seconds == 0.0 || seconds == INFINITY)
             ? 0.0
             : ((double)(flops * iters) / seconds) * 1e-9;
}

double calcKib(const uint64_t probSize, const uint64_t bytesPerElem) {
  return ((double)(probSize * bytesPerElem) / 1024);
}

uint64_t gemmFlops(const int M, const int N, const int K) {
  return ((uint64_t)M * (uint64_t)N * (uint64_t)K * 2);
}

double gemmKib(const dataTypes type, const int M, const int N, const int K) {
  uint8_t bytes = 0;
  switch (type) {
  case _fp32_:
    bytes = 4;
    break;
  case _fp64_:
    bytes = 8;
    break;
  default:
    break;
  }
  uint64_t M_ = (uint64_t)M, N_ = (uint64_t)N, K_ = (uint64_t)K;
  uint64_t probSize = (M_ * K_) + (K_ * N_) + (M_ * N_);
  return calcKib(probSize, bytes);
}

FILE *newCSV(const char *filename) {
  FILE *newFile;
  newFile = fopen(filename, "w");
  if (newFile == NULL) {
    perror("ERROR - File failed to open: ");
    exit(1);
  }
  // Write headers to file.
  fprintf(newFile,
          "Device,Kernel,M,N,K,Total Problem Size (KiB),Iterations,Total "
          "Seconds,GFLOP/s\n");
  return newFile;
}

void writeLineToCsv(FILE *fptr, const char *device, const char *kernel,
                    const int M, const int N, const int K,
                    const double totalProbSize, const int iters,
                    const double totalTime, const double gflops) {
  if (fptr == NULL) {
    printf("ERROR - Attempted to write line to invalid file pointer.\n");
    exit(1);
  }
  fprintf(fptr, "%s,%s,%d,%d,%d,%.2lf,%d,%lf,%lf\n", device, kernel, M, N, K,
          totalProbSize, iters, totalTime, gflops);
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