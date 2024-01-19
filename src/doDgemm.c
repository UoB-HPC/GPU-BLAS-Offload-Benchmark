#include "../include/commonHeadersDo.h"

void callDgemms(FILE *fptr, const int iters, const int M, const int N,
                const int K);

void doDgemm(const int iters, const int upperLimit) {
  FILE *fptr;
  // Square Problem Sizes...
  // Open new CSV file
  fptr = newCSV(CSV_DIR "/dgemm_square.csv");
  for (int dim = 1; dim <= upperLimit; dim++) {
    const int M = dim, N = dim, K = dim;
    callDgemms(fptr, iters, M, N, K);
  }
  // Close file
  fclose(fptr);

  // Rectangular Problem Sizes:
  // Tall and thin (16M x K)...
  fptr = newCSV(CSV_DIR "/dgemm_rectangular_16MxK.csv");
  for (int dim = 16; dim <= upperLimit; dim += 16) {
    const int M = dim, N = dim, K = (dim / 16);
    callDgemms(fptr, iters, M, N, K);
  }
  // Close file
  fclose(fptr);

  // Tall and thin (M x 32)...
  fptr = newCSV(CSV_DIR "/dgemm_rectangular_Mx32.csv");
  if (upperLimit >= 32) {
    for (int dim = 1; dim <= upperLimit; dim++) {
      const int M = dim, N = dim, K = 32;
      callDgemms(fptr, iters, M, N, K);
    }
  }
  // Close file
  fclose(fptr);

  // Short and wide (M x 16K)...
  fptr = newCSV(CSV_DIR "/dgemm_rectangular_Mx16K.csv");
  for (int dim = 16; dim <= upperLimit; dim += 16) {
    const int M = (dim / 16), N = (dim / 16), K = dim;
    callDgemms(fptr, iters, M, N, K);
  }
  // Close file
  fclose(fptr);

  // Short and wide (32 x K)...
  fptr = newCSV(CSV_DIR "/dgemm_rectangular_32xK.csv");
  if (upperLimit >= 32) {
    for (int dim = 1; dim <= upperLimit; dim++) {
      const int M = 32, N = 32, K = dim;
      callDgemms(fptr, iters, M, N, K);
    }
  }
  // Close file
  fclose(fptr);
}

void callDgemms(FILE *fptr, const int iters, const int M, const int N,
                const int K) {
  const uint64_t gemmProbSize = gemmKib(_fp64_, M, N, K);
  // Perform CPU
  double cpuTime = gemm_cpu(_fp64_, iters, M, N, K);
  writeLineToCsv(fptr, "cpu", "dgemm", M, N, K, gemmProbSize, iters, cpuTime,
                 calcGflops(gemmFlops(M, N, K), iters, cpuTime));

  // Perform GPU
  // - Offload to/from GPU once before all iterations and once after
  double gpuTime_once = gemm_gpu(_fp64_, iters, M, N, K, true);
  writeLineToCsv(fptr, "gpu_offloadOnce", "dgemm", M, N, K, gemmProbSize, iters,
                 gpuTime_once,
                 calcGflops(gemmFlops(M, N, K), iters, gpuTime_once));
  // - Offload to/from GPU every iteration
  double gpuTime_every = gemm_gpu(_fp64_, iters, M, N, K, false);
  writeLineToCsv(fptr, "gpu_offloadAlways", "dgemm", M, N, K, gemmProbSize,
                 iters, gpuTime_every,
                 calcGflops(gemmFlops(M, N, K), iters, gpuTime_every));
}