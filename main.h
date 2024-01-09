#include "utilities.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "cpuKernels.h"
#include "gpuKernels.h"

/** A function which prints standard configuration information to stdout. */
void printBenchmarkConfig(const int iters, const int upperLimit) {
  char *gpuEnabledStr = (GPU_ENABLED) ? "True" : "False";
  unsigned int ompThreads =
      (getenv("OMP_NUM_THREADS") != NULL) ? atoi(getenv("OMP_NUM_THREADS")) : 1;
  const char *ompProcBind =
      (getenv("OMP_PROC_BIND") != NULL) ? getenv("OMP_PROC_BIND") : "Not Set";
  const char *ompPlaces =
      (getenv("OMP_PLACES") != NULL) ? getenv("OMP_PLACES") : "Not Set";
  printf("GPU BLAS Offload Benchmark:\n");
  printf("\tIterations per Kernel: %d\n", iters);
  printf("\tMax Problem Dimension: %d\n", upperLimit);
  printf("\tCPU Kernels Enabled: True\n");
  printf("\tCPU Library: %s\n", CPU_LIB_NAME);
  printf("\tGPU Kernels Enabled: %s\n", gpuEnabledStr);
  printf("\tGPU Library: %s\n", GPU_LIB_NAME);
  printf("\tOMP_NUM_THREADS: %u\n", ompThreads);
  printf("\tOMP_PROC_BIND: %s\n", ompProcBind);
  printf("\tOMP_PLACES: %s\n", ompPlaces);
#ifdef CPU_DEFAULT
  printf("\nWARNING - No CPU BLAS library selected. Results will be collected "
         "from a single threaded naive implementation.\n");
#endif
#ifdef GPU_DEFAULT
  printf("\nWARNING - No GPU BLAS Library selected. All results will be based "
         "off of a time of infinity.\n");
#endif
  printf("\n");
}

/** A function to open a new csv file in WRITE mode and write the standard
 * headers to the file.
 * Returns the file pointer. */
FILE *newCSV(const char *filename) {
  FILE *newFile;
  newFile = fopen(filename, "w");
  // Write headers to file.
  fprintf(newFile,
          "Device,Kernel,M,N,K,Total Problem Size (KiB),Iterations,Total "
          "Seconds,GFLOP/s\n");
  return newFile;
}

/** A function to write a new line to an open CSV file. */
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

/** A function to calculate GFLOPs. */
double calcGflops(const uint64_t flops, const int iters, const double seconds) {
  return (seconds == 0.0 || seconds == INFINITY)
             ? 0.0
             : ((double)(flops * iters) / seconds) * 1e-9;
}

/** A function to calculate KiB from a data-structur's dimensions. */
double calcKib(const uint64_t probSize, const uint64_t bytesPerElem) {
  return ((double)(probSize * bytesPerElem) / 1024);
}

/** A function for calculating FLOPs performed by a GEMM. */
uint64_t gemmFlops(const int M, const int N, const int K) {
  return (M * N * K * 2);
}