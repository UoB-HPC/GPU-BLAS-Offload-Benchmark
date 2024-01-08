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

/** A function which prints standard configuration information to stdout. */
void printBenchmarkConfig() {
  char *gpu_enabled_str = (GPU_ENABLED) ? "True" : "False";
  unsigned int omp_threads =
      (getenv("OMP_NUM_THREADS") != NULL) ? atoi(getenv("OMP_NUM_THREADS")) : 1;
  const char *omp_proc_bind =
      (getenv("OMP_PROC_BIND") != NULL) ? getenv("OMP_PROC_BIND") : "Not Set";
  const char *omp_places =
      (getenv("OMP_PLACES") != NULL) ? getenv("OMP_PLACES") : "Not Set";
  printf("GPU BLAS Offload Benchmark:\n");
  printf("\tIterations per Kernel: %d\n", ITERATIONS);
  printf("\tMax Problem Dimension: %d\n", UPPER_LIMIT);
  printf("\tCPU Tests Enabled: True\n");
  printf("\tCPU Library: %s\n", CPU_LIB_NAME);
  printf("\tGPU Tests Enabled: %s\n", gpu_enabled_str);
  printf("\tGPU Library: %s\n", GPU_LIB_NAME);
  printf("\tOMP_NUM_THREADS: %u\n", omp_threads);
  printf("\tOMP_PROC_BIND: %s\n", omp_proc_bind);
  printf("\tOMP_PLACES: %s\n", omp_places);
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

void writeLineToCsv(FILE *fptr, const char *device, const char *kernel,
                    const uint64_t M, const uint64_t N, const uint64_t K,
                    const double totalProbSize, const uint64_t iters,
                    const double totalTime, const double gflops) {
  fprintf(fptr, "%s,%s,%llu,%llu,%llu,%.2lf,%llu,%lf,%lf\n", device, kernel, M,
          N, K, totalProbSize, iters, totalTime, gflops);
}