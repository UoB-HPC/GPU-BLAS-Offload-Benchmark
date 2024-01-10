#include "../include/commonHeadersDo.h"

void callSgemms(FILE *fptr, const int iters, const int M, const int N,
                const int K);

void doSgemm(const int iters, const int upperLimit) {
  FILE *fptr;
  printf("\tSquare Problem Sizes...\n");
  // Open new CSV file
  fptr = newCSV(CSV_DIR "/sgemm_square.csv");
  for (int dim = 1; dim <= upperLimit; dim++) {
    const int M = dim, N = dim, K = dim;
    callSgemms(fptr, iters, M, N, K);
  }
  // Close file
  fclose(fptr);

  printf("\tRectangular Problem Sizes:\n");
  printf("\t\tTall and thin (16M x K)...\n");
  fptr = newCSV(CSV_DIR "/sgemm_rectangular_16MxK.csv");
  for (int dim = 16; dim <= upperLimit; dim += 16) {
    const int M = dim, N = dim, K = (dim / 16);
    callSgemms(fptr, iters, M, N, K);
  }
  // Close file
  fclose(fptr);

  printf("\t\tTall and thin (M x 32)...\n");
  fptr = newCSV(CSV_DIR "/sgemm_rectangular_Mx32.csv");
  if (upperLimit >= 32) {
    for (int dim = 1; dim <= upperLimit; dim++) {
      const int M = dim, N = dim, K = 32;
      callSgemms(fptr, iters, M, N, K);
    }
  }
  // Close file
  fclose(fptr);

  printf("\t\tShort and wide (M x 16K)...\n");
  fptr = newCSV(CSV_DIR "/sgemm_rectangular_Mx16K.csv");
  for (int dim = 16; dim <= upperLimit; dim += 16) {
    const int M = (dim / 16), N = (dim / 16), K = dim;
    callSgemms(fptr, iters, M, N, K);
  }
  // Close file
  fclose(fptr);

  printf("\t\tShort and wide (32 x K)...\n");
  fptr = newCSV(CSV_DIR "/sgemm_rectangular_32xK.csv");
  if (upperLimit >= 32) {
    for (int dim = 1; dim <= upperLimit; dim++) {
      const int M = 32, N = 32, K = dim;
      callSgemms(fptr, iters, M, N, K);
    }
  }
  // Close file
  fclose(fptr);
}

void callSgemms(FILE *fptr, const int iters, const int M, const int N,
                const int K) {
  const uint64_t gemmProbSize = gemmKib(_fp32_, M, N, K);
  // Perform CPU
  double cpuTime = gemm_cpu(_fp32_, iters, M, N, K);
  writeLineToCsv(fptr, "cpu", "sgemm", M, N, K, gemmProbSize, iters, cpuTime,
                 calcGflops(gemmFlops(M, N, K), iters, cpuTime));

  // Perform GPU
  // - Offload to/from GPU once before all iterations and once after
  double gpuTime_once = gemm_gpu(_fp32_, iters, M, N, K, true);
  writeLineToCsv(fptr, "gpu_offloadOnce", "sgemm", M, N, K, gemmProbSize, iters,
                 gpuTime_once,
                 calcGflops(gemmFlops(M, N, K), iters, gpuTime_once));
  // - Offload to/from GPU every iteration
  double gpuTime_every = gemm_gpu(_fp32_, iters, M, N, K, false);
  writeLineToCsv(fptr, "gpu_offloadAlways", "sgemm", M, N, K, gemmProbSize,
                 iters, gpuTime_every,
                 calcGflops(gemmFlops(M, N, K), iters, gpuTime_every));
}