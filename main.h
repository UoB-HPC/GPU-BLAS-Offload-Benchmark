#include "utilities.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// Select which CPU Library to use
#if defined CPU_DEFAULT
#include "DefaultCPU/default.h"
#elif defined CPU_ARMPL
#include "ArmPL/armpl.h"
#elif defined CPU_ONEMKL
// #include "OneMKL/onemkl.h"
#elif defined CPU_OPENBLAS
// #include "OpenBLAS/openblas.h"
#elif defined CPU_AOCL
// #include "AOCL/aocl.h"
#endif

// Select which GPU Library to use
#if defined GPU_DEFAULT
#include "DefaultGPU/default.h"
#elif defined GPU_CUBLAS
// #include "cuBLAS/cublas.h"
#elif defined GPU_ONEMKL
// #include "OneMKL/onemkl.h"
#elif defined GPU_ROCBLAS
// #include "rocBLAS/rocblas.h"
#endif

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

void writeLineToCsv(FILE *fptr, const char *device, const char *kernel,
                    const uint64_t M, const uint64_t N, const uint64_t K,
                    const double totalProbSize, const uint64_t iters,
                    const double totalTime, const double gflops) {
  fprintf(fptr, "%s,%s,%llu,%llu,%llu,%.2lf,%llu,%lf,%lf\n", device, kernel, M,
          N, K, totalProbSize, iters, totalTime, gflops);
}