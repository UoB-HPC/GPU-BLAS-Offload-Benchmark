#include "../include/helpers.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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

double calcGflops(const uint64_t flops, const int iters, const double seconds) {
  return (seconds == 0.0 || seconds == INFINITY)
             ? 0.0
             : ((double)(flops * iters) / seconds) * 1e-9;
}

double calcKib(const uint64_t probSize, const uint64_t bytesPerElem) {
  return ((double)(probSize * bytesPerElem) / 1024);
}

/* ------------------------------ GEMM -------------------------------------- */

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

/* -------------------------------------------------------------------------- */